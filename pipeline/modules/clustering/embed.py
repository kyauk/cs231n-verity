"""Module 8: Clustering — embedding clients (EmbedClient implementations).

NIMEmbedClient   — production: fetch the clip from its (signed) URL, base64 it,
                   POST to the Cosmos-Embed NIM /v1/embeddings endpoint.
StubEmbedClient  — offline/tests: deterministic pseudo-vector from the URL, so
                   the full embed -> UMAP -> HDBSCAN path runs with no NIM/GPU.
"""

from __future__ import annotations

import base64
import hashlib
import math
import os
from typing import Any

from pipeline.modules.clustering.config import EmbedUnavailableError

_DEFAULT_URL = "http://localhost:8080"
_DEFAULT_MODEL = "nvidia/cosmos-embed1"


class NIMEmbedClient:
    """Cosmos-Embed NIM client. Reads COSMOS_EMBED1_URL / COSMOS_EMBED1_MODEL_ID.

    Ported from the legacy waymo_embed_scenes embed call: download clip ->
    base64 -> POST /v1/embeddings -> data[0].embedding.
    """
    model_id: str

    def __init__(
        self,
        cosmos_url: str | None = None,
        model_id: str | None = None,
        timeout_seconds: int = 120,
    ) -> None:
        self._url = (cosmos_url or os.environ.get("COSMOS_EMBED1_URL", _DEFAULT_URL)).rstrip("/")
        self.model_id = model_id or os.environ.get("COSMOS_EMBED1_MODEL_ID", _DEFAULT_MODEL)
        self._timeout = timeout_seconds

    def embed(self, video_url: str) -> list[float]:
        try:
            import requests  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise EmbedUnavailableError(f"missing dependency: {exc}") from exc
        try:
            clip = requests.get(video_url, timeout=self._timeout)
            clip.raise_for_status()
            b64 = base64.b64encode(clip.content).decode()
            payload = {
                "input": [f"data:video/mp4;base64,{b64}"],
                "request_type": "query",
                "encoding_format": "float",
                "model": self.model_id,
            }
            r = requests.post(f"{self._url}/v1/embeddings", json=payload, timeout=self._timeout)
            if r.status_code != 200:
                raise EmbedUnavailableError(
                    f"Cosmos-Embed returned {r.status_code}: {r.text[:300]}"
                )
            return [float(x) for x in r.json()["data"][0]["embedding"]]
        except EmbedUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise EmbedUnavailableError(f"{type(exc).__name__}: {exc}") from exc


class StubEmbedClient:
    """Deterministic offline embedder — no network/GPU.

    Produces a stable pseudo-random unit-ish vector from the URL so the
    clustering math runs in tests and `--stub` runs. Same URL -> same vector.
    """
    model_id: str = "stub/cosmos-embed"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, video_url: str) -> list[float]:
        seed = hashlib.sha256(video_url.encode()).digest()
        vec: list[float] = []
        i = 0
        while len(vec) < self.dim:
            h = hashlib.sha256(seed + i.to_bytes(4, "big")).digest()
            for b in h:
                if len(vec) >= self.dim:
                    break
                # map byte -> [-1, 1]
                vec.append((b / 127.5) - 1.0)
            i += 1
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]
