"""Module 2: Encoder — reasoning arm.

Calls Cosmos-Reason2 via the NVIDIA NIM OpenAI-compatible API, extracts
structured JSON from the response, validates it against the locked vocabulary,
and returns a fields dict.

Known assumptions / accepted risks (Phase 1):
- Retry prompt always uses the "invalid JSON" correction even for vocabulary
  violations. Both failure types use the same stricter prompt. Acceptable
  because vocabulary violations are rare with Cosmos-Reason2 in practice.
- No concurrency: annotate() is synchronous and not thread-safe. The Encoder
  is designed for sequential batch processing (process_batch). For parallel
  annotation, instantiate one Encoder per thread/process.
- VLM response is assumed ≤ 4096 tokens. Longer responses may be truncated
  silently by the API; extract_json() will still try all extraction patterns.

The VLMClient protocol is the seam. CosmosReason2Client is the production
implementation. StubVLMClient is drop-in for tests — same output type,
no network call.

Standalone usage (production):
    from pipeline.modules.encoder.reasoning_arm import ReasoningArm, CosmosReason2Client
    from pipeline.modules.encoder.vocabulary import DEFAULT_VOCABULARY
    import os

    client = CosmosReason2Client(api_key=os.environ["NVIDIA_API_KEY"])
    arm = ReasoningArm(vlm=client, vocabulary=DEFAULT_VOCABULARY)
    fields, raw = arm.annotate(video_url="https://...", pose_summary="...",
                               prompt_template_id="v1_describe")

Standalone usage (test / offline):
    from pipeline.modules.encoder.reasoning_arm import ReasoningArm, StubVLMClient
    arm = ReasoningArm(vlm=StubVLMClient(), vocabulary=DEFAULT_VOCABULARY)
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Protocol, runtime_checkable

from pipeline.modules.encoder.vocabulary import Vocabulary


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ReasoningArmError(Exception):
    """Base class for reasoning-arm failures."""


class VLMUnavailableError(ReasoningArmError):
    """Raised when the VLM endpoint cannot be reached at all.

    This is a transient infrastructure failure — propagate to the caller,
    do not silently record as a schema failure.
    """
    def __init__(self, model: str, detail: str) -> None:
        self.model = model
        self.detail = detail
        super().__init__(
            f"\n{'='*70}\n"
            f"ENCODER ERROR: VLMUnavailableError\n"
            f"  Model  : {model}\n"
            f"  Detail : {detail}\n"
            f"  → Check NVIDIA_API_KEY and network access.\n"
            f"{'='*70}\n"
        )
        print(str(self), file=sys.stderr)


# ---------------------------------------------------------------------------
# VLMClient Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class VLMClient(Protocol):
    """Interface every VLM backend must implement.

    `complete` receives a public video URL and a text prompt, and returns
    the raw text content of the model's response. Parsing, JSON extraction,
    and validation happen in ReasoningArm — not here.
    """
    model_id: str

    def complete(self, video_url: str, prompt: str) -> str:
        """Call the model and return its raw text output."""
        ...


# ---------------------------------------------------------------------------
# Production client — Cosmos-Reason2 via NVIDIA NIM
# ---------------------------------------------------------------------------

class CosmosReason2Client:
    """Calls Cosmos-Reason2 via the NVIDIA NIM OpenAI-compatible API.

    Environment variables (loaded from .env by the caller):
      NVIDIA_API_KEY     — required
      NVIDIA_BASE_URL    — default: https://integrate.api.nvidia.com/v1
      COSMOS_REASON2_MODEL_ID — default: nvidia/cosmos-reason2-7b
    """

    model_id: str

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model_id: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        timeout: float | None = None,
    ) -> None:
        self.model_id = (
            model_id
            or os.environ.get("COSMOS_REASON2_MODEL_ID", "nvidia/cosmos-reason2-7b")
        )
        self._api_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
        self._base_url = base_url or os.environ.get(
            "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
        )
        self._max_tokens = max_tokens
        self._temperature = temperature
        # 10-minute ceiling by default. A stuck NIM call must not hang a
        # worker thread indefinitely (would block ThreadPoolExecutor's
        # shutdown, including Ctrl-C in pipeline.run analyze).
        self._timeout = float(
            timeout
            if timeout is not None
            else os.environ.get("NVIDIA_NIM_TIMEOUT_SECONDS", "600")
        )

        if not self._api_key:
            print(
                "\n[Encoder/ReasoningArm] WARNING: NVIDIA_API_KEY is not set. "
                "Calls to CosmosReason2Client will fail.\n"
                "  Set it in .env or pass api_key= to the constructor.\n",
                file=sys.stderr,
            )

    def _get_client(self) -> Any:
        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError:
            print(
                "\n[Encoder/ReasoningArm] MISSING DEPENDENCY: openai\n"
                "  Install it with:  pip install openai\n",
                file=sys.stderr,
            )
            raise
        return OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self._timeout,
        )

    def _resolve_video_url(self, video_url: str) -> str:
        """Return a URL the NIM can ingest.

        A local NIM cannot fetch a gs:// URI (and user-ADC can't sign one), so
        when storage hands back gs:// (its no-signing fallback), download the clip
        in-process via ADC and inline it as a base64 data URI. https/data URLs
        pass through unchanged.
        """
        if not video_url.startswith("gs://"):
            return video_url
        import base64  # noqa: PLC0415
        from google.cloud import storage  # noqa: PLC0415
        no_scheme = video_url[len("gs://"):]
        bucket_name, _, blob_name = no_scheme.partition("/")
        project = os.environ.get("GCS_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project:
            try:
                import google.auth  # noqa: PLC0415
                creds, project = google.auth.default()
                project = project or getattr(creds, "quota_project_id", None)
            except Exception:  # noqa: BLE001
                project = None
        data = storage.Client(project=project).bucket(bucket_name).blob(blob_name).download_as_bytes()
        return "data:video/mp4;base64," + base64.b64encode(data).decode()

    def complete(self, video_url: str, prompt: str) -> str:
        """Send a video URL + text prompt to Cosmos-Reason2. Return raw text."""
        client = self._get_client()
        video_ref = self._resolve_video_url(video_url)
        try:
            response = client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video_url",
                                "video_url": {"url": video_ref},
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            raise VLMUnavailableError(self.model_id, f"{type(exc).__name__}: {exc}") from exc


# ---------------------------------------------------------------------------
# Stub client — same output type, no network call
# ---------------------------------------------------------------------------

class StubVLMClient:
    """Drop-in replacement for CosmosReason2Client for tests and offline runs.

    Returns a deterministic, schema-valid JSON response. The output shape
    is identical to what Cosmos-Reason2 would return after reasoning.
    """

    model_id: str = "stub/cosmos-reason2"

    def complete(self, video_url: str, prompt: str) -> str:
        """Return a hardcoded Cosmos-Reason2-shaped response with valid JSON."""
        # Cosmos-Reason2 typically emits chain-of-thought wrapped in <think>
        # tags, followed by the structured answer. We replicate that shape.
        payload = {
            "agents": ["car", "pedestrian"],
            "environment": {
                "weather": "clear",
                "time_of_day": "day",
                "lighting_condition": "well_lit",
            },
            "road": {
                "geometry": "intersection",
                "lane_count": 4,
            },
            "traffic_control": "traffic_light",
            "ego_task": "cruising",
            "conditions": [],
        }
        json_block = json.dumps(payload, indent=2)
        return (
            "<think>\n"
            "The video shows a daytime urban intersection. "
            "The ego vehicle is proceeding straight through a signalized intersection. "
            "Several cars and a pedestrian are visible.\n"
            "</think>\n\n"
            f"```json\n{json_block}\n```"
        )


# ---------------------------------------------------------------------------
# JSON extraction from VLM response
# ---------------------------------------------------------------------------

def extract_json(text: str) -> dict[str, Any]:
    """Extract the first valid JSON object from a raw VLM response.

    Handles:
      1. Direct JSON string
      2. ```json ... ``` code fences
      3. Bare ``` ... ``` fences
      4. <json>...</json> tags
      5. First {...} block in the response

    Raises ValueError if no valid JSON object is found.
    """
    text = text.strip()

    # 1. Full response is valid JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # 2. ```json ... ``` fences
    fence_match = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        try:
            obj = json.loads(fence_match.group(1).strip())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # 3. Bare ``` ... ``` fences
    bare_fence = re.search(r"```\s*([\s\S]*?)```", text, re.DOTALL)
    if bare_fence:
        try:
            obj = json.loads(bare_fence.group(1).strip())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # 4. <json>...</json> tags
    tag_match = re.search(r"<json>\s*(.*?)\s*</json>", text, re.DOTALL | re.IGNORECASE)
    if tag_match:
        try:
            obj = json.loads(tag_match.group(1).strip())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # 5. First {...} block (greedy from last closing brace)
    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        try:
            obj = json.loads(brace_match.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"No valid JSON object found in VLM response.\n"
        f"Response preview: {text[:500]!r}"
    )


# ---------------------------------------------------------------------------
# Reasoning arm
# ---------------------------------------------------------------------------

_PROMPT_DIR = __file__.replace("reasoning_arm.py", "prompts")


def _load_prompt_template(template_id: str) -> str:
    """Load a versioned prompt template from the prompts/ directory.

    Raises FileNotFoundError with a clear message if the template is missing.
    """
    import pathlib  # noqa: PLC0415
    path = pathlib.Path(_PROMPT_DIR) / f"{template_id}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"[Encoder/ReasoningArm] Prompt template {template_id!r} not found.\n"
            f"  Expected: {path}\n"
            f"  Available: {list(pathlib.Path(_PROMPT_DIR).glob('*.txt'))}"
        )
    return path.read_text(encoding="utf-8")


class ReasoningArm:
    """Annotates one window using Cosmos-Reason2.

    Parameters
    ----------
    vlm         Any VLMClient — CosmosReason2Client in production,
                StubVLMClient in tests.
    vocabulary  Locked Vocabulary instance used to validate field values.
    max_retries Max JSON parse + vocabulary retry attempts (default 3).
    camera      Camera to pull video from (default "FRONT").
    """

    def __init__(
        self,
        vlm: VLMClient,
        vocabulary: Vocabulary,
        max_retries: int = 3,
        camera: str = "FRONT",
    ) -> None:
        self._vlm = vlm
        self._vocab = vocabulary
        self._max_retries = max_retries
        self._camera = camera

    def annotate(
        self,
        video_url: str,
        pose_summary: str | None,
        prompt_template_id: str = "v1_describe",
    ) -> tuple[dict[str, Any], str]:
        """Annotate one window. Returns (fields_dict, raw_vlm_response).

        Retries up to max_retries times on JSON parse failure or vocabulary
        violation. On final failure, raises the last exception so the
        Encoder can record the failure_mode.

        Raises:
          VLMUnavailableError  — if the VLM endpoint cannot be reached
          ValueError           — if JSON extraction fails after all retries
          VocabularyViolation  — if fields still violate vocabulary after retries
        """
        template = _load_prompt_template(prompt_template_id)
        base_prompt = self._build_prompt(template, pose_summary)

        last_exc: Exception | None = None
        last_raw: str = ""
        violations: list[str] = []

        for attempt in range(1, self._max_retries + 1):
            prompt = base_prompt
            if attempt > 1:
                prompt = self._stricter_prompt(base_prompt, violations)
                print(
                    f"[Encoder/ReasoningArm] Retry {attempt}/{self._max_retries} "
                    f"for window (violations: {violations})",
                    file=sys.stderr,
                )

            try:
                raw = self._vlm.complete(video_url=video_url, prompt=prompt)
                last_raw = raw
            except VLMUnavailableError:
                raise  # not a retry-able failure

            try:
                fields = extract_json(raw)
            except ValueError as exc:
                last_exc = exc
                violations = [f"JSON parse failed: {exc}"]
                continue

            violations = self._vocab.validate_fields(fields)
            if not violations:
                return fields, raw

            last_exc = ValueError(
                f"Vocabulary violations after attempt {attempt}: {violations}"
            )

        # All retries exhausted
        assert last_exc is not None
        raise last_exc

    def annotate_from_storage(
        self,
        segment_id: str,
        window_idx: int,
        storage: Any,
        prompt_template_id: str = "v1_describe",
        ttl_seconds: int = 3600,
    ) -> tuple[dict[str, Any], str]:
        """Fetch video URL + pose summary from storage, then annotate.

        Raises WindowStorageError if storage lookup fails — that propagates
        to the Encoder as FAILURE_STORAGE_ERROR, not silently.
        """
        video_url = storage.get_window_video_url(
            segment_id, window_idx, camera=self._camera, ttl_seconds=ttl_seconds
        )
        try:
            manifest = storage.get_window_manifest(segment_id, window_idx)
            pose_summary = manifest.pose_summary
        except Exception as exc:
            print(
                f"[Encoder/ReasoningArm] WARNING: manifest fetch failed for "
                f"{segment_id}/{window_idx:04d} ({type(exc).__name__}: {exc}). "
                f"Proceeding without pose summary.",
                file=sys.stderr,
            )
            pose_summary = None

        return self.annotate(
            video_url=video_url,
            pose_summary=pose_summary,
            prompt_template_id=prompt_template_id,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_prompt(self, template: str, pose_summary: str | None) -> str:
        """Substitute vocabulary + pose into the prompt template."""
        for placeholder in ("{{VOCABULARY}}", "{{POSE_SUMMARY}}"):
            if placeholder not in template:
                raise ValueError(
                    f"[Encoder/ReasoningArm] Prompt template is missing placeholder "
                    f"{placeholder!r}. Check prompts/ directory."
                )
        vocab_block = self._vocab.prompt_context()
        pose_block = pose_summary or "No pose data available."
        return (
            template
            .replace("{{VOCABULARY}}", vocab_block)
            .replace("{{POSE_SUMMARY}}", pose_block)
        )

    def _stricter_prompt(self, base_prompt: str, violations: list[str]) -> str:
        """Append a correction block to the prompt on retry."""
        violation_text = "\n".join(f"  - {v}" for v in violations)
        correction = (
            "\n\n"
            "IMPORTANT — your previous response had these issues:\n"
            f"{violation_text}\n\n"
            "Fix all issues and return ONLY valid JSON using the vocabulary above. "
            "No prose. No markdown fences. Just the JSON object."
        )
        return base_prompt + correction
