# DEPRECATED (2026-06-04): the UMAP/HDBSCAN/GLOSH math here now lives in
# pipeline/modules/clustering/clusterer.py (Module 8). Kept only until the :8000
# waymo_runner batch is repointed to `pipeline.run cluster`
# (ARCHITECTURE_PROPOSAL.md §8, retirement phase 2), then delete this file.
"""Loads Waymo window embeddings, runs two-pass UMAP + HDBSCAN.

Mirrors ``pipeline/cluster_embeddings.py``. Saves cluster labels + 3D
visualization coordinates to a .npz cache and two JSONL files:
  - clusters.jsonl         : per-window cluster_label + glosh_score
  - flagged_windows.jsonl  : anomaly-ranked rows (AnomalyResultRecord shape)

The flagged_windows.jsonl output is what the runner's agentic-analysis stage
consumes -- the same contract the reference debate pipeline expects.

Usage:
  python -m waymo_pipeline.waymo_cluster_embeddings \
      --input-jsonl outputs/waymo_window_embeddings.jsonl \
      --output-npz outputs/waymo_clusters.npz \
      --output-jsonl outputs/waymo_clusters.jsonl \
      --flagged-jsonl outputs/flagged_windows.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np
from dotenv import load_dotenv
from sklearn.preprocessing import normalize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_embeddings(
    path: str, embedding_dim: int | None = None
) -> tuple[np.ndarray, list[str], list[dict]]:
    """Load embedding matrix, window IDs, and raw rows from a JSONL file."""
    log.info("Loading embeddings from %s ...", path)
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        return np.array([]).reshape(0, 0), [], []

    embeddings = [r["embedding"] for r in rows]
    dim = embedding_dim or min(len(e) for e in embeddings)
    X = np.array([e[:dim] for e in embeddings], dtype=np.float32)
    window_ids = [r.get("window_id", "") for r in rows]
    log.info("Loaded %d embeddings of dimension %d", len(window_ids), X.shape[1])
    return X, window_ids, rows


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize embeddings before cosine-metric UMAP."""
    return normalize(embeddings, norm="l2")


def umap_cluster_pass(
    embeddings: np.ndarray, n_neighbors: int, n_components: int, seed: int
) -> np.ndarray:
    """High-dimensional UMAP pass optimized for clustering."""
    import umap

    log.info(
        "UMAP clustering pass: %dD -> %dD (n_neighbors=%d)",
        embeddings.shape[1], n_components, n_neighbors,
    )
    reducer = umap.UMAP(
        n_components=n_components, n_neighbors=n_neighbors, min_dist=0.0,
        metric="cosine", random_state=seed, low_memory=False,
    )
    return reducer.fit_transform(embeddings)


def umap_viz_pass(
    embeddings: np.ndarray, n_neighbors: int, min_dist: float, seed: int
) -> np.ndarray:
    """3D UMAP pass on original embeddings, purely for visualization."""
    import umap

    log.info("UMAP viz pass: %dD -> 3D (n_neighbors=%d)", embeddings.shape[1], n_neighbors)
    reducer = umap.UMAP(
        n_components=3, n_neighbors=n_neighbors, min_dist=min_dist,
        metric="cosine", random_state=seed, low_memory=False,
    )
    return reducer.fit_transform(embeddings)


def run_hdbscan(
    reduced: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    cluster_selection_method: str = "eom",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cluster in UMAP-reduced space; return labels, GLOSH scores, probabilities."""
    import hdbscan

    log.info(
        "HDBSCAN: min_cluster_size=%d, min_samples=%d, method=%s",
        min_cluster_size, min_samples, cluster_selection_method,
    )
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples,
        cluster_selection_method=cluster_selection_method, metric="euclidean",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(reduced)
    outlier_scores = clusterer.outlier_scores_
    probabilities = clusterer.probabilities_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    log.info(
        "Found %d clusters, %d noise points (%.1f%% noise)",
        n_clusters, n_noise, 100 * n_noise / max(1, len(labels)),
    )
    return labels, outlier_scores, probabilities


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Load embeddings, two-pass UMAP, HDBSCAN, write cluster + flagged JSONL."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Two-pass UMAP + HDBSCAN on Waymo window embeddings."
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument(
        "--flagged-jsonl", default="outputs/flagged_windows.jsonl",
        help="Anomaly-ranked rows consumed by the agentic analysis stage.",
    )
    parser.add_argument(
        "--umap-components", type=int,
        default=int(os.getenv("UMAP_N_COMPONENTS", "50")),
    )
    parser.add_argument(
        "--n-neighbors", type=int, default=int(os.getenv("UMAP_N_NEIGHBORS", "15")),
    )
    parser.add_argument(
        "--viz-min-dist", type=float, default=float(os.getenv("UMAP_VIZ_MIN_DIST", "0.1")),
    )
    parser.add_argument(
        "--min-cluster-size", type=int,
        default=int(os.getenv("HDBSCAN_MIN_CLUSTER_SIZE", "5")),
    )
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument(
        "--cluster-selection-method", type=str, default="eom", choices=["eom", "leaf"],
    )
    parser.add_argument("--seed", type=int, default=int(os.getenv("UMAP_RANDOM_SEED", "42")))
    parser.add_argument("--embedding-dim", type=int, default=None)
    args = parser.parse_args()

    if not os.path.isfile(args.input_jsonl):
        log.error("Input not found: %s", args.input_jsonl)
        return 1

    embeddings, window_ids, rows = load_embeddings(
        args.input_jsonl, embedding_dim=args.embedding_dim
    )
    if embeddings.size == 0:
        log.error("No embeddings in input.")
        return 1

    embeddings = l2_normalize(embeddings)
    # n_neighbors must be < n_samples for UMAP.
    n_neighbors = min(args.n_neighbors, max(2, len(window_ids) - 1))

    reduced_hd = umap_cluster_pass(
        embeddings, n_neighbors, min(args.umap_components, len(window_ids) - 1), args.seed
    )
    labels, outlier_scores, probabilities = run_hdbscan(
        reduced_hd, args.min_cluster_size, args.min_samples, args.cluster_selection_method
    )
    reduced_3d = umap_viz_pass(embeddings, n_neighbors, args.viz_min_dist, args.seed)

    # .npz cache
    os.makedirs(os.path.dirname(args.output_npz) or ".", exist_ok=True)
    np.savez(
        args.output_npz,
        window_ids=np.array(window_ids),
        labels=labels,
        outlier_scores=outlier_scores,
        probabilities=probabilities,
        reduced_hd=reduced_hd,
        reduced_3d=reduced_3d,
    )
    log.info("Saved: %s", args.output_npz)

    # per-window cluster metadata JSONL
    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for i, wid in enumerate(window_ids):
            f.write(json.dumps({
                "window_id": wid,
                "cluster_label": int(labels[i]),
                "cluster_probability": float(probabilities[i]),
                "glosh_score": float(outlier_scores[i]),
                "coord_3d": [float(c) for c in reduced_3d[i]],
            }) + "\n")
    log.info("Saved: %s", args.output_jsonl)

    # anomaly-ranked flagged windows (AnomalyResultRecord contract)
    order = sorted(range(len(window_ids)), key=lambda i: -float(outlier_scores[i]))
    row_by_window = {r.get("window_id", ""): r for r in rows}
    os.makedirs(os.path.dirname(args.flagged_jsonl) or ".", exist_ok=True)
    with open(args.flagged_jsonl, "w", encoding="utf-8") as f:
        for rank, i in enumerate(order, start=1):
            wid = window_ids[i]
            src = row_by_window.get(wid, {})
            f.write(json.dumps({
                "window_id": wid,
                "scene_token_hex": src.get("scene_token_hex", wid),
                "log_id": src.get("log_id", "waymo"),
                "scenario_tags": src.get("scenario_tags", []),
                "window_start_ts": src.get("window_start_ts"),
                "window_end_ts": src.get("window_end_ts"),
                "cluster_label": int(labels[i]),
                "is_noise": bool(labels[i] == -1),
                "cluster_probability": float(probabilities[i]),
                "outlier_score": float(outlier_scores[i]),
                "anomaly_rank": rank,
                "quality": src.get("quality", {}),
                "metadata": {**src.get("metadata", {}), "dataset": "waymo"},
            }) + "\n")
    log.info("Saved: %s", args.flagged_jsonl)

    return 0


if __name__ == "__main__":
    sys.exit(main())
