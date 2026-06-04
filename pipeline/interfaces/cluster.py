"""Shared clustering types — produced by Module 8: Clustering.

The clustering module embeds each window (continuous representation) and groups
windows by density in embedding space (UMAP -> HDBSCAN). These are the only
types that cross its module boundary; like every other interface here they are
plain dataclasses with to_json / from_json and are pinned by a round-trip test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pipeline.interfaces.window import WindowKey


@dataclass
class WindowEmbedding:
    """One window's embedding vector (continuous representation)."""
    window_id: WindowKey
    vector: list[float]
    dim: int

    def to_json(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id.to_json(),
            "vector": list(self.vector),
            "dim": self.dim,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "WindowEmbedding":
        return cls(
            window_id=WindowKey.from_json(d["window_id"]),
            vector=[float(x) for x in d["vector"]],
            dim=int(d["dim"]),
        )


@dataclass
class ClusterAssignment:
    """Which cluster a window landed in, plus its outlier/uncertainty signals."""
    window_id: WindowKey
    cluster_id: int            # -1 = noise / outlier (HDBSCAN convention)
    glosh_score: float         # GLOSH outlier score in [0, 1]; higher = more anomalous
    probability: float         # HDBSCAN soft-membership probability in [0, 1]
    coords_3d: list[float]     # 3D UMAP coords for the Cluster Space viz (len 3)

    @property
    def is_noise(self) -> bool:
        return self.cluster_id == -1

    def to_json(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id.to_json(),
            "cluster_id": self.cluster_id,
            "glosh_score": self.glosh_score,
            "probability": self.probability,
            "coords_3d": list(self.coords_3d),
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "ClusterAssignment":
        return cls(
            window_id=WindowKey.from_json(d["window_id"]),
            cluster_id=int(d["cluster_id"]),
            glosh_score=float(d["glosh_score"]),
            probability=float(d["probability"]),
            coords_3d=[float(x) for x in d.get("coords_3d", [])],
        )


@dataclass
class ClusterReport:
    """Full output of one clustering run over a set of windows."""
    assignments: list[ClusterAssignment]
    n_clusters: int
    n_noise: int
    embedding_dim: int
    config: dict = field(default_factory=dict)
    created_at: str = ""

    @property
    def n_windows(self) -> int:
        return len(self.assignments)

    def to_json(self) -> dict[str, Any]:
        return {
            "assignments": [a.to_json() for a in self.assignments],
            "n_clusters": self.n_clusters,
            "n_noise": self.n_noise,
            "embedding_dim": self.embedding_dim,
            "config": dict(self.config),
            "created_at": self.created_at,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "ClusterReport":
        return cls(
            assignments=[ClusterAssignment.from_json(a) for a in d.get("assignments", [])],
            n_clusters=int(d["n_clusters"]),
            n_noise=int(d["n_noise"]),
            embedding_dim=int(d["embedding_dim"]),
            config=dict(d.get("config", {})),
            created_at=str(d.get("created_at", "")),
        )
