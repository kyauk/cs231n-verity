"""Production model clients for the debate module.

NIMTextLLMClient — chat-completion text LLM for the four debate actors, via the
    NVIDIA NIM OpenAI-compatible API (``NVIDIA_BASE_URL`` + ``NVIDIA_API_KEY``,
    model ``DEBATE_NIM_MODEL_ID``, default meta/llama-3.1-8b-instruct).
NIMVLMClient    — VLM describe/follow-up over a clip, via ``DESCRIBE_NIM_MODEL_ID``
    (default nvidia/nemotron-nano-12b-v2-vl). The clip referenced by
    ``video_ref`` is fetched and base64-encoded into a data URI, mirroring the
    clustering NIMEmbedClient.

Both raise :class:`DebateModelUnavailableError` on transport/endpoint failure.
The Stub*/Failing* clients in config.py remain the right choice for offline runs.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import tempfile
from typing import Any

from pipeline.modules.debate.config import DebateModelUnavailableError


_DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
_DEFAULT_TEXT_MODEL_ID = "meta/llama-3.1-8b-instruct"
_DEFAULT_VLM_MODEL_ID = "nvidia/nemotron-nano-12b-v2-vl"

_VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".webm")
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
_MIME_BY_EXT: dict[str, str] = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
    ".webm": "video/webm",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


class NIMTextLLMClient:
    """Text LLM for debate turns via the NIM OpenAI-compatible chat API.

    Implements the ``TextLLMClient`` protocol (model_id + complete()).
    """

    model_id: str

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model_id: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        timeout: float | None = None,
    ) -> None:
        self.model_id = (
            model_id or os.environ.get("DEBATE_NIM_MODEL_ID", _DEFAULT_TEXT_MODEL_ID)
        )
        self._api_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
        self._base_url = base_url or os.environ.get("NVIDIA_BASE_URL", _DEFAULT_BASE_URL)
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout = float(
            timeout
            if timeout is not None
            else os.environ.get("NVIDIA_NIM_TIMEOUT_SECONDS", "600")
        )

    def _get_client(self) -> Any:
        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError as exc:
            raise DebateModelUnavailableError(
                f"missing dependency: {exc} (pip install openai)"
            ) from exc
        return OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self._timeout,
        )

    def complete(
        self, messages: list[dict[str, str]], temperature: float | None = None
    ) -> str:
        """Send OpenAI-style messages to the NIM. Return the assistant text."""
        client = self._get_client()
        try:
            response = client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=(
                    self._temperature if temperature is None else temperature
                ),
            )
            return response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            raise DebateModelUnavailableError(
                f"{type(exc).__name__}: {exc} (model={self.model_id})"
            ) from exc


def _fetch_clip_bytes(video_ref: str, timeout: float) -> tuple[bytes, str]:
    """Return (raw bytes, mime type) for a clip URL or local path."""

    lowered = video_ref.lower()
    mime = "video/mp4"
    for ext, ext_mime in _MIME_BY_EXT.items():
        if lowered.endswith(ext):
            mime = ext_mime
            break

    if video_ref.startswith(("http://", "https://")):
        try:
            import requests  # noqa: PLC0415
        except ImportError as exc:
            raise DebateModelUnavailableError(
                f"missing dependency: {exc} (pip install requests)"
            ) from exc
        resp = requests.get(video_ref, timeout=timeout)
        resp.raise_for_status()
        return resp.content, mime

    with open(os.path.abspath(video_ref), "rb") as handle:
        return handle.read(), mime


def _video_to_frame_jpegs(raw: bytes) -> list[bytes]:
    """Sample a few evenly-spaced JPEG frames from a clip — instead of the video.

    Why frames, not the video: Cosmos-Reason1 decodes a submitted clip on the
    GPU's NVDEC, and the NIM's vLLM worker pre-reserves most of the VRAM for KV
    cache, leaving little (and fluctuating) headroom for video decode. A full
    ~20s segment intermittently busts that and ``cuvidMapVideoFrame error 2``
    *segfaults the whole NIM* (exit 139) — not a recoverable per-request error.
    There is no fixed video size that's reliably safe.

    Sending a handful of still frames sidesteps NVDEC entirely (images use the
    normal, cheap image path), so it cannot OOM or crash the decoder, and it has
    no fps/pixel-budget constraints. The model still gets temporally-ordered
    coverage of the scene. Tunables: ``VLM_CLIP_FRAMES`` (default 6),
    ``VLM_CLIP_MAX_WIDTH`` (default 768). Returns [] if ffmpeg is unavailable or
    extraction fails (caller falls back to sending the raw clip).
    """
    num_frames = max(1, int(os.environ.get("VLM_CLIP_FRAMES", "6")))
    max_width = int(os.environ.get("VLM_CLIP_MAX_WIDTH", "768"))
    if shutil.which("ffmpeg") is None:
        return []
    try:
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "in.mp4")
            with open(src, "wb") as f:
                f.write(raw)
            # Total frame count (fallback to a thumbnail spread if unknown).
            nb = 0
            if shutil.which("ffprobe") is not None:
                probe = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "v:0",
                     "-count_frames", "-show_entries", "stream=nb_read_frames",
                     "-of", "default=nw=1:nk=1", src],
                    capture_output=True, text=True,
                )
                nb = int(probe.stdout.strip() or "0") if probe.returncode == 0 else 0
            step = max(1, nb // num_frames) if nb else 1
            # Evenly-spaced frames: pick every `step`-th, capped to num_frames.
            select = f"select='not(mod(n\\,{step}))'" if nb else "fps=1"
            vf = f"{select},scale='min({max_width},iw)':-2"
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vf", vf,
                 "-vsync", "0", "-frames:v", str(num_frames), "-q:v", "3",
                 os.path.join(td, "f_%03d.jpg")],
                capture_output=True, text=True,
            )
            frames: list[bytes] = []
            for name in sorted(os.listdir(td)):
                if name.startswith("f_") and name.endswith(".jpg"):
                    with open(os.path.join(td, name), "rb") as f:
                        frames.append(f.read())
            return frames
    except Exception:  # noqa: BLE001 — never let media prep break the VLM call
        return []


class NIMVLMClient:
    """VLM describe/follow-up via the NIM OpenAI-compatible chat API.

    Implements the ``VLMClient`` protocol (model_id + describe() + followup()).
    The clip referenced by ``video_ref`` is fetched and base64-encoded into a
    data URI before being attached to the chat request.
    """

    model_id: str

    _DESCRIBE_SYSTEM_PROMPT = (
        "You are a vision-language assistant describing a REAL recorded "
        "autonomous-driving clip. Use the anomaly signals as priors, then "
        "verify them against the video. "
        "Output ONLY a strict JSON object with keys scene_description (concrete, "
        "temporally ordered), anomaly_rationale (why the clip may be a valuable "
        "AV regression edge case), and confidence (low|medium|high). No markdown, "
        "no prose outside the JSON object."
    )

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model_id: str | None = None,
        max_tokens: int = 3200,
        temperature: float = 0.2,
        timeout: float | None = None,
    ) -> None:
        self.model_id = (
            model_id or os.environ.get("DESCRIBE_NIM_MODEL_ID", _DEFAULT_VLM_MODEL_ID)
        )
        self._api_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
        self._base_url = base_url or os.environ.get("NVIDIA_BASE_URL", _DEFAULT_BASE_URL)
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout = float(
            timeout
            if timeout is not None
            else os.environ.get("NVIDIA_NIM_TIMEOUT_SECONDS", "600")
        )

    def _get_client(self) -> Any:
        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError as exc:
            raise DebateModelUnavailableError(
                f"missing dependency: {exc} (pip install openai)"
            ) from exc
        return OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self._timeout,
        )

    @staticmethod
    def _image_block(jpeg: bytes) -> dict[str, Any]:
        uri = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
        return {"type": "image_url", "image_url": {"url": uri}}

    def _media_blocks(self, video_ref: str) -> list[dict[str, Any]]:
        """Build OpenAI-style media content blocks from a clip reference.

        For a video, sample a few still frames and send them as IMAGES (see
        ``_video_to_frame_jpegs`` — this avoids the NVDEC decode that segfaults
        the NIM). An already-image ref is sent as-is. If frame extraction yields
        nothing (no ffmpeg), fall back to sending the raw clip as a video block.
        """
        raw, mime = _fetch_clip_bytes(video_ref, self._timeout)
        if mime.startswith("image/"):
            return [self._image_block(raw)]
        frames = _video_to_frame_jpegs(raw)
        if frames:
            return [self._image_block(j) for j in frames]
        # Fallback: no ffmpeg — send the raw video and hope the decoder copes.
        uri = f"data:{mime};base64," + base64.b64encode(raw).decode()
        return [{"type": "video_url", "video_url": {"url": uri}}]

    def _chat(self, system_prompt: str, content: list[dict[str, Any]]) -> str:
        client = self._get_client()
        try:
            response = client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            raise DebateModelUnavailableError(
                f"{type(exc).__name__}: {exc} (model={self.model_id})"
            ) from exc

    def describe(self, video_ref: str, anomaly_priors: dict) -> str:
        """Return JSON: {scene_description, anomaly_rationale, confidence}."""
        content: list[dict[str, Any]] = []
        if video_ref:
            content.extend(self._media_blocks(video_ref))
        content.append(
            {
                "type": "text",
                "text": (
                    "The images above are time-ordered frames sampled from one "
                    "driving clip (earliest first). Anomaly signals for the clip "
                    "(use as priors, verify against the frames):\n"
                    + json.dumps(anomaly_priors, indent=2)
                ),
            }
        )
        return self._chat(self._DESCRIBE_SYSTEM_PROMPT, content)

    def followup(self, video_ref: str, prompt: str) -> str:
        """Return free-text answer to a targeted question about the clip."""
        content: list[dict[str, Any]] = []
        if video_ref:
            content.extend(self._media_blocks(video_ref))
        content.append({"type": "text", "text": prompt})
        return self._chat(
            "You are a vision-language assistant answering targeted follow-up "
            "questions about a REAL recorded autonomous-driving scene. Answer "
            "concisely (<=80 words) grounded in what is visible.",
            content,
        )
