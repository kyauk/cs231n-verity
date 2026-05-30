"""Production TextClient for the Scorer — NVIDIA NIM OpenAI-compatible API.

Parallel to encoder.reasoning_arm.CosmosReason2Client, but text-only: both the
plausibility and difficulty arms send text prompts (no video URL), so a single
client implementation serves both.

Environment variables (loaded from .env by the caller):
  NVIDIA_API_KEY      — required
  NVIDIA_BASE_URL     — default: https://integrate.api.nvidia.com/v1
  SCORER_NIM_MODEL_ID — default: meta/llama-3.1-70b-instruct

Standalone usage:
    import os
    from pipeline.modules.scorer import NIMTextClient, Scorer

    client = NIMTextClient(api_key=os.environ["NVIDIA_API_KEY"])
    scorer = Scorer(plausibility_client=client, difficulty_client=client)

The Stub*/Failing* clients in plausibility.py and difficulty.py remain the
right choice for offline runs and failure-path tests.
"""

from __future__ import annotations

import os
import sys
from typing import Any


_DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
_DEFAULT_MODEL_ID = "meta/llama-3.1-70b-instruct"


class NIMUnavailableError(Exception):
    """Raised when the NIM endpoint cannot be reached.

    Parallels encoder.reasoning_arm.VLMUnavailableError. The Scorer's arms
    catch all exceptions from client.complete() and aggregate; this class
    exists so callers (and logs) can distinguish endpoint outages from
    application-level errors.
    """

    def __init__(self, model: str, detail: str) -> None:
        self.model = model
        self.detail = detail
        super().__init__(
            f"\n{'='*70}\n"
            f"SCORER ERROR: NIMUnavailableError\n"
            f"  Model  : {model}\n"
            f"  Detail : {detail}\n"
            f"  → Check NVIDIA_API_KEY and network access.\n"
            f"{'='*70}\n"
        )
        print(str(self), file=sys.stderr)


class NIMTextClient:
    """Calls a text-only NIM via the OpenAI-compatible chat completions API.

    Implements the TextClient protocol (model_id + complete(prompt) -> str)
    defined in pipeline.modules.scorer.config.
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
            or os.environ.get("SCORER_NIM_MODEL_ID", _DEFAULT_MODEL_ID)
        )
        self._api_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
        self._base_url = base_url or os.environ.get("NVIDIA_BASE_URL", _DEFAULT_BASE_URL)
        self._max_tokens = max_tokens
        self._temperature = temperature
        # 10-minute ceiling by default. Shared with CosmosReason2Client via
        # NVIDIA_NIM_TIMEOUT_SECONDS so operators tune both at once.
        self._timeout = float(
            timeout
            if timeout is not None
            else os.environ.get("NVIDIA_NIM_TIMEOUT_SECONDS", "600")
        )

        if not self._api_key:
            print(
                "\n[Scorer/NIMTextClient] WARNING: NVIDIA_API_KEY is not set. "
                "Calls to NIMTextClient will fail.\n"
                "  Set it in .env or pass api_key= to the constructor.\n",
                file=sys.stderr,
            )

    def _get_client(self) -> Any:
        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError:
            print(
                "\n[Scorer/NIMTextClient] MISSING DEPENDENCY: openai\n"
                "  Install it with:  pip install openai\n",
                file=sys.stderr,
            )
            raise
        return OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self._timeout,
        )

    def complete(self, prompt: str) -> str:
        """Send a text prompt to the NIM. Returns the raw response text."""
        client = self._get_client()
        try:
            response = client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            raise NIMUnavailableError(
                self.model_id, f"{type(exc).__name__}: {exc}"
            ) from exc
