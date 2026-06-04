"""Module 8: Clustering — embed windows, then group by density in embedding space.

Lego boundary: depends only on pipeline.interfaces (WindowKey, WindowStorageBase,
ClusterReport) and an injected EmbedClient. Reads windows over the
WindowStorageBase Protocol — the same surface the Encoder uses — so it shares
Module 1's ingestion with zero coupling to its internals.

Algorithm (ported from the legacy waymo clustering, unchanged math):
  L2-normalize -> UMAP(cosine) cluster pass -> HDBSCAN(GLOSH) -> 3D UMAP viz.
"""

from __future__ import annotations

import datetime
import sys
from typing import Any

from pipeline.interfaces.cluster import (
    ClusterAssignment,
    ClusterReport,
    WindowEmbedding,
)
from pipeline.modules.clustering.config import (
    ClustererConfig,
    EmbedClient,
    EmbedUnavailableError,
)

# Below this many embedded windows, UMAP/HDBSCAN are not meaningful (and UMAP
# requires n_components < n_samples). We still return a valid report — every
# window as its own un-clustered point — rather than crash.
_MIN_WINDOWS_FOR_CLUSTERING = 5


class Clusterer:
    """Module 8 entry point. Embed windows, cluster, emit a ClusterReport."""

    def __init__(
        self,
        embed_client: EmbedClient,
        config: ClustererConfig = ClustererConfig(),
    ) -> None:
        self._embed = embed_client
        self._config = config

    # ------------------------------------------------------------------
    # Stage A — embed each window via the storage Protocol
    # ------------------------------------------------------------------

    def embed_windows(self, windows: list[Any], storage: Any) -> list[WindowEmbedding]:
        """Embed each window (cameras concatenated, L2-normalized).

        `windows` is a list of WindowKey; `storage` satisfies WindowStorageBase.
        Windows whose clip is unavailable are skipped (logged), never fatal.
        """
        out: list[WindowEmbedding] = []
        for w in windows:
            try:
                per_camera: list[float] = []
                for cam in self._config.cameras:
                    url = storage.get_window_video_url(
                        w.segment_id, w.window_idx, camera=cam,
                        ttl_seconds=self._config.ttl_seconds,
                    )
                    per_camera.extend(self._embed.embed(url))
                vec = _l2_normalize(per_camera)
                out.append(WindowEmbedding(window_id=w, vector=vec, dim=len(vec)))
            except EmbedUnavailableError as exc:
                print(f"[Clustering] embed failed for {w.segment_id}/{w.window_idx:04d}: {exc}",
                      file=sys.stderr)
            except Exception as exc:  # noqa: BLE001 — a bad window must not kill the batch
                print(f"[Clustering] skipped {w.segment_id}/{w.window_idx:04d}: "
                      f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return out

    # ------------------------------------------------------------------
    # Stage B — cluster the embeddings
    # ------------------------------------------------------------------

    def cluster(self, embeddings: list[WindowEmbedding]) -> ClusterReport:
        """UMAP -> HDBSCAN over the embeddings. Pure (no I/O)."""
        cfg = self._config
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        config_json = {
            "cameras": list(cfg.cameras),
            "umap_n_components": cfg.umap_n_components,
            "umap_n_neighbors": cfg.umap_n_neighbors,
            "umap_seed": cfg.umap_seed,
            "hdbscan_min_cluster_size": cfg.hdbscan_min_cluster_size,
            "glosh_threshold": cfg.glosh_threshold,
            "embed_model": getattr(self._embed, "model_id", "unknown"),
        }
        n = len(embeddings)
        dim = embeddings[0].dim if embeddings else 0

        if n < _MIN_WINDOWS_FOR_CLUSTERING:
            # Too few to cluster: every window is its own point, no labels.
            assignments = [
                ClusterAssignment(window_id=e.window_id, cluster_id=-1,
                                  glosh_score=0.0, probability=0.0, coords_3d=[0.0, 0.0, 0.0])
                for e in embeddings
            ]
            print(f"[Clustering] only {n} windows (< {_MIN_WINDOWS_FOR_CLUSTERING}); "
                  f"returning unclustered report.", file=sys.stderr)
            return ClusterReport(assignments=assignments, n_clusters=0, n_noise=n,
                                 embedding_dim=dim, config=config_json, created_at=now)

        import numpy as np  # noqa: PLC0415
        import umap  # noqa: PLC0415
        import hdbscan  # noqa: PLC0415

        X = np.asarray([e.vector for e in embeddings], dtype=np.float32)
        # UMAP needs n_neighbors < n_samples and n_components < n_samples.
        n_neighbors = max(2, min(cfg.umap_n_neighbors, n - 1))
        n_components = max(2, min(cfg.umap_n_components, n - 2))

        reduced = umap.UMAP(
            n_components=n_components, n_neighbors=n_neighbors, min_dist=cfg.umap_min_dist,
            metric="cosine", random_state=cfg.umap_seed, low_memory=False,
        ).fit_transform(X)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=cfg.hdbscan_min_cluster_size,
            min_samples=cfg.hdbscan_min_samples,
            cluster_selection_method="eom", metric="euclidean", prediction_data=True,
        )
        labels = clusterer.fit_predict(reduced)
        glosh = getattr(clusterer, "outlier_scores_", None)
        probs = getattr(clusterer, "probabilities_", None)

        viz = umap.UMAP(
            n_components=3, n_neighbors=n_neighbors, min_dist=0.1,
            metric="cosine", random_state=cfg.umap_seed, low_memory=False,
        ).fit_transform(X)

        assignments: list[ClusterAssignment] = []
        for i, e in enumerate(embeddings):
            assignments.append(ClusterAssignment(
                window_id=e.window_id,
                cluster_id=int(labels[i]),
                glosh_score=float(glosh[i]) if glosh is not None else 0.0,
                probability=float(probs[i]) if probs is not None else 0.0,
                coords_3d=[float(viz[i][0]), float(viz[i][1]), float(viz[i][2])],
            ))
        n_clusters = len(set(int(l) for l in labels) - {-1})
        n_noise = int(sum(1 for l in labels if int(l) == -1))
        print(f"[Clustering] {n} windows -> {n_clusters} clusters, {n_noise} noise.",
              file=sys.stderr)
        return ClusterReport(assignments=assignments, n_clusters=n_clusters, n_noise=n_noise,
                             embedding_dim=dim, config=config_json, created_at=now)

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def run(self, windows: list[Any], storage: Any) -> ClusterReport:
        """Embed all windows then cluster. The module's one-call entry point."""
        return self.cluster(self.embed_windows(windows, storage))


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = sum(x * x for x in vec) ** 0.5
    if norm == 0.0:
        return list(vec)
    return [x / norm for x in vec]
