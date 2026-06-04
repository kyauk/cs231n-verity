"""Module 8: Clustering — configuration, client protocol, and error types.

The clustering module embeds each window with a video embedder, then groups
windows by density in embedding space. Like the Scorer's TextClient and the
Encoder's VLMClient, the embedder is an injected Protocol — NIMEmbedClient is
the production (Cosmos-Embed) impl; StubEmbedClient is for offline runs/tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# EmbedClient protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class EmbedClient(Protocol):
    """Video embedder. Takes a fetchable clip URL, returns a fixed-dim vector.

    Mirrors the Encoder's VLMClient(video_url, prompt) shape — both consume a
    window via its signed URL from WindowStorageBase, so clustering reads
    windows exactly like the encoder does (over the Protocol, never the bucket).
    """
    model_id: str

    def embed(self, video_url: str) -> list[float]:
        """Embed one clip and return its vector."""
        ...


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ClustererConfig:
    """Clustering knobs. Defaults mirror the project .env (UMAP_* / HDBSCAN_*)."""
    cameras: tuple[str, ...] = ("FRONT",)   # cameras embedded + concatenated per window
    umap_n_components: int = 50
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.0
    umap_seed: int = 42
    hdbscan_min_cluster_size: int = 5
    hdbscan_min_samples: int | None = None  # None -> hdbscan default (= min_cluster_size)
    glosh_threshold: float = 0.7            # GLOSH score above this = flagged outlier
    ttl_seconds: int = 3600                 # signed-URL TTL for clip fetches

    @classmethod
    def from_env(cls, cameras: tuple[str, ...] = ("FRONT",)) -> "ClustererConfig":
        """Build from environment (the same vars the legacy waymo pipeline read)."""
        def _int(name: str, default: int) -> int:
            try:
                return int(os.environ.get(name, default))
            except (TypeError, ValueError):
                return default
        def _float(name: str, default: float) -> float:
            try:
                return float(os.environ.get(name, default))
            except (TypeError, ValueError):
                return default
        return cls(
            cameras=cameras,
            umap_n_components=_int("UMAP_N_COMPONENTS", 50),
            umap_seed=_int("UMAP_RANDOM_SEED", 42),
            hdbscan_min_cluster_size=_int("HDBSCAN_MIN_CLUSTER_SIZE", 5),
            glosh_threshold=_float("GLOSH_THRESHOLD", 0.7),
        )


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ClusteringError(Exception):
    """Base error for the clustering module."""


class EmbedUnavailableError(ClusteringError):
    """The embedding endpoint could not be reached or returned an error."""
