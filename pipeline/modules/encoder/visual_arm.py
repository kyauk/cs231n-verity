"""Module 2: Encoder — visual arm.

Calls Cosmos-Embed1 via its NIM `/v1/embeddings` endpoint. Downloads each
camera's MP4 from GCS via a signed URL, base64-encodes it, and posts it to
the embedding endpoint. Concatenates per-camera 256-d vectors into a
1280-d embedding (5 cameras × 256), then L2-normalizes.

The EmbedClient protocol is the seam. CosmosEmbed1Client is the production
implementation. StubEmbedClient is drop-in for tests — same output type,
no network call.

Known assumptions / accepted risks (Phase 1):
- All configured cameras must succeed. A missing camera raises immediately
  so the failure_mode is set on the SchemaRecord rather than producing a
  silently truncated or zero-padded embedding.
- MP4 bytes are downloaded in full before encoding. For 8-second clips at
  default quality this is typically 1–5 MB per camera.
- Embedding dim (256-d per camera, 1280-d total) is fixed by Cosmos-Embed1.
  If the model is updated, both the dim constant and callers need updating.

Standalone usage (production):
    from pipeline.modules.encoder.visual_arm import CosmosEmbed1Client, VisualArm
    import os

    client = CosmosEmbed1Client(cosmos_url=os.environ["COSMOS_EMBED1_URL"])
    arm = VisualArm(client=client)
    fields, _ = arm.annotate_from_storage(
        segment_id="seg_001",
        window_idx=0,
        storage=storage,
    )
    # fields["embedding"] is a list[float] of length 1280

Standalone usage (test / offline):
    from pipeline.modules.encoder.visual_arm import StubEmbedClient, VisualArm

    arm = VisualArm(client=StubEmbedClient())
    fields, _ = arm.annotate_from_storage("seg_001", 0, mock_storage)
"""

from __future__ import annotations

import base64
import os
import sys
from typing import Any, Protocol, runtime_checkable

_EMBED_DIM_PER_CAMERA = 256
_DEFAULT_CAMERAS = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"]
_COSMOS_MODEL = "nvidia/cosmos-embed1"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class VisualArmError(Exception):
    """Base class for visual-arm failures."""


class EmbedUnavailableError(VisualArmError):
    """Raised when the Cosmos-Embed1 endpoint cannot be reached."""

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(
            f"\n{'='*70}\n"
            f"ENCODER ERROR: EmbedUnavailableError\n"
            f"  Detail : {detail}\n"
            f"  → Check COSMOS_EMBED1_URL and that the NIM container is running.\n"
            f"{'='*70}\n"
        )
        print(str(self), file=sys.stderr)


# ---------------------------------------------------------------------------
# EmbedClient Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class EmbedClient(Protocol):
    """Interface every embedding backend must implement."""

    model_id: str

    def embed(
        self,
        segment_id: str,
        window_idx: int,
        storage: Any,
        cameras: list[str],
    ) -> tuple[list[float], list[str]]:
        """Embed a window. Returns (embedding_vector, cameras_used).

        embedding_vector has length len(cameras) * _EMBED_DIM_PER_CAMERA.
        cameras_used is the subset of cameras that were actually embedded.
        """
        ...


# ---------------------------------------------------------------------------
# Production client — Cosmos-Embed1 via NIM
# ---------------------------------------------------------------------------

class CosmosEmbed1Client:
    """Calls Cosmos-Embed1 NIM to embed video windows.

    Environment variables:
      COSMOS_EMBED1_URL  — default: http://localhost:8080
      COSMOS_EMBED1_MODEL_ID — default: nvidia/cosmos-embed1
    """

    model_id: str

    def __init__(
        self,
        cosmos_url: str | None = None,
        model_id: str | None = None,
        timeout_seconds: int = 120,
    ) -> None:
        self._url = (cosmos_url or os.environ.get("COSMOS_EMBED1_URL", "http://localhost:8080")).rstrip("/")
        self.model_id = model_id or os.environ.get("COSMOS_EMBED1_MODEL_ID", _COSMOS_MODEL)
        self._timeout = timeout_seconds

    def embed(
        self,
        segment_id: str,
        window_idx: int,
        storage: Any,
        cameras: list[str],
    ) -> tuple[list[float], list[str]]:
        """Download each camera MP4 via signed URL, embed, concatenate, normalize."""
        try:
            import numpy as np  # noqa: PLC0415
            import requests  # noqa: PLC0415
        except ImportError as exc:
            raise EmbedUnavailableError(f"Missing dependency: {exc}") from exc

        per_camera: list[Any] = []
        cameras_used: list[str] = []

        for cam in cameras:
            # Let WindowStorageError propagate: a missing/unreadable window is a
            # storage failure, classified separately from an embed-endpoint outage.
            url = storage.get_window_video_url(segment_id, window_idx, camera=cam)
            try:
                resp = requests.get(url, timeout=self._timeout)
                resp.raise_for_status()
                mp4_bytes = resp.content
            except Exception as exc:
                raise EmbedUnavailableError(
                    f"Failed to download MP4 for {segment_id}/{window_idx:04d}/{cam}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

            b64 = base64.b64encode(mp4_bytes).decode()
            payload = {
                "input": [f"data:video/mp4;base64,{b64}"],
                "request_type": "query",
                "encoding_format": "float",
                "model": self.model_id,
            }
            try:
                r = requests.post(
                    f"{self._url}/v1/embeddings",
                    json=payload,
                    timeout=self._timeout,
                )
                r.raise_for_status()
                vec = np.array(r.json()["data"][0]["embedding"], dtype=np.float32)
            except Exception as exc:
                raise EmbedUnavailableError(
                    f"Cosmos-Embed1 call failed for {segment_id}/{window_idx:04d}/{cam}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

            per_camera.append(vec)
            cameras_used.append(cam)

        if not per_camera:
            raise EmbedUnavailableError(
                f"No cameras produced embeddings for {segment_id}/{window_idx:04d}"
            )

        concat = np.concatenate(per_camera)
        norm = np.linalg.norm(concat) + 1e-12
        embedding = (concat / norm).astype(float).tolist()
        return embedding, cameras_used


# ---------------------------------------------------------------------------
# Stub client — no network call
# ---------------------------------------------------------------------------

class StubEmbedClient:
    """Drop-in replacement for CosmosEmbed1Client for tests and offline runs.

    Returns a deterministic zero vector. Shape matches the real client.
    """

    model_id: str = "stub/cosmos-embed1"

    def embed(
        self,
        segment_id: str,
        window_idx: int,
        storage: Any,
        cameras: list[str],
    ) -> tuple[list[float], list[str]]:
        dim = len(cameras) * _EMBED_DIM_PER_CAMERA
        return [0.0] * dim, list(cameras)


# ---------------------------------------------------------------------------
# Visual arm
# ---------------------------------------------------------------------------

class VisualArm:
    """Embeds one window using Cosmos-Embed1.

    Parameters
    ----------
    client    Any EmbedClient — CosmosEmbed1Client in production,
              StubEmbedClient in tests.
    cameras   Camera list to embed. Default: all 5 Waymo cameras.
    """

    def __init__(
        self,
        client: EmbedClient,
        cameras: list[str] | None = None,
    ) -> None:
        self._client = client
        self._cameras = cameras or list(_DEFAULT_CAMERAS)

    @property
    def model_id(self) -> str:
        return self._client.model_id

    def annotate_from_storage(
        self,
        segment_id: str,
        window_idx: int,
        storage: Any,
    ) -> tuple[dict[str, Any], None]:
        """Embed a window. Returns (fields_dict, None).

        fields_dict keys:
          "embedding"  list[float] — L2-normalized concatenated camera embeddings
          "cameras"    list[str]   — cameras that were embedded
          "model_id"   str         — embedding model identifier

        Raises EmbedUnavailableError on network or storage failure.
        """
        embedding, cameras_used = self._client.embed(
            segment_id=segment_id,
            window_idx=window_idx,
            storage=storage,
            cameras=self._cameras,
        )
        return {
            "embedding": embedding,
            "cameras": cameras_used,
            "model_id": self._client.model_id,
        }, None
