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

    def _media_block(self, video_ref: str) -> dict[str, Any]:
        """Build an OpenAI-style media content block from a clip reference."""

        raw, mime = _fetch_clip_bytes(video_ref, self._timeout)
        b64 = base64.b64encode(raw).decode()
        data_uri = f"data:{mime};base64,{b64}"
        if mime.startswith("image/"):
            return {"type": "image_url", "image_url": {"url": data_uri}}
        return {"type": "video_url", "video_url": {"url": data_uri}}

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
            content.append(self._media_block(video_ref))
        content.append(
            {
                "type": "text",
                "text": (
                    "Anomaly signals for this clip (use as priors, verify against "
                    "the video):\n" + json.dumps(anomaly_priors, indent=2)
                ),
            }
        )
        return self._chat(self._DESCRIBE_SYSTEM_PROMPT, content)

    def followup(self, video_ref: str, prompt: str) -> str:
        """Return free-text answer to a targeted question about the clip."""
        content: list[dict[str, Any]] = []
        if video_ref:
            content.append(self._media_block(video_ref))
        content.append({"type": "text", "text": prompt})
        return self._chat(
            "You are a vision-language assistant answering targeted follow-up "
            "questions about a REAL recorded autonomous-driving scene. Answer "
            "concisely (<=80 words) grounded in what is visible.",
            content,
        )
