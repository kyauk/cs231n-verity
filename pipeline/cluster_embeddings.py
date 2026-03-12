"""Loads embeddings from JSONL, runs two-pass UMAP + HDBSCAN.

Saves cluster labels + reduced coordinates to a .npz cache and a .jsonl
metadata file.

Usage:
    python -m pipeline.cluster_embeddings \
        --input-jsonl outputs/all_embeddings.jsonl \
        --output-npz outputs/clusters.npz \
        --output-jsonl outputs/clusters.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import hdbscan
import numpy as np
import umap
from dotenv import load_dotenv
from sklearn.preprocessing import normalize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_embeddings(
    path: str,
    embedding_dim: int | None = None,
) -> tuple[np.ndarray, list[str], list[dict]]:
    '''
    Purpose: Load embedding matrix and window IDs from a JSONL file.
    Parameters:
        path (str): Path to embedding JSONL.
        embedding_dim (int | None): If set, truncate to first N dims.
    Returns:
        tuple: (X array (n, d), window_ids list, raw row dicts list).
    Called by: main().
    Calls: open(), json.loads().
    '''
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


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    '''
    Purpose: L2-normalize embeddings before cosine-metric UMAP.
    Parameters:
        embeddings (np.ndarray): Shape (n, d).
    Returns:
        np.ndarray: L2-normalized, same shape.
    Called by: main().
    Calls: sklearn.preprocessing.normalize().
    '''
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    log.info(
        "Embedding norm stats — min: %.3f, max: %.3f, mean: %.3f",
        norms.min(), norms.max(), norms.mean(),
    )
    return normalize(embeddings, norm="l2")


def umap_cluster_pass(
    embeddings: np.ndarray,
    n_neighbors: int,
    n_components: int,
    seed: int,
) -> np.ndarray:
    '''
    Purpose: High-dimensional UMAP pass optimised for clustering.
    Parameters:
        embeddings (np.ndarray): Shape (n, high_dim).
        n_neighbors (int): UMAP neighborhood size.
        n_components (int): Target intermediate dimensionality.
        seed (int): Random state for reproducibility.
    Returns:
        np.ndarray: Shape (n, n_components).
    Called by: main().
    Calls: umap.UMAP().fit_transform().
    '''
    log.info(
        "UMAP clustering pass: %dD → %dD (n_neighbors=%d)",
        embeddings.shape[1], n_components, n_neighbors,
    )
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.0,
        metric="cosine",
        random_state=seed,
        low_memory=False,
    )
    reduced = reducer.fit_transform(embeddings)
    log.info("UMAP clustering pass complete. Output shape: %s", reduced.shape)
    return reduced


def umap_viz_pass(
    embeddings: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    seed: int,
) -> np.ndarray:
    '''
    Purpose: 3D UMAP pass on original embeddings purely for visualization.
    Parameters:
        embeddings (np.ndarray): Shape (n, high_dim) — original embeddings.
        n_neighbors (int): UMAP neighborhood size.
        min_dist (float): Small nonzero so clusters don't visually collapse.
        seed (int): Random state for reproducibility.
    Returns:
        np.ndarray: Shape (n, 3).
    Called by: main().
    Calls: umap.UMAP().fit_transform().
    '''
    log.info(
        "UMAP viz pass: %dD → 3D (n_neighbors=%d, min_dist=%.2f)",
        embeddings.shape[1], n_neighbors, min_dist,
    )
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
        low_memory=False,
    )
    reduced = reducer.fit_transform(embeddings)
    log.info("UMAP viz pass complete. Output shape: %s", reduced.shape)
    return reduced


def run_hdbscan(
    reduced: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    cluster_selection_method: str = "eom",
) -> tuple[np.ndarray, np.ndarray]:
    '''
    Purpose: Cluster in the UMAP-reduced space and extract GLOSH scores.
    Parameters:
        reduced (np.ndarray): Shape (n, d) — the UMAP-reduced matrix.
        min_cluster_size (int): Minimum points in a cluster.
        min_samples (int): Core-point density parameter.
        cluster_selection_method (str): "eom" or "leaf".
    Returns:
        tuple: (labels (n,), outlier_scores (n,)).
    Called by: main().
    Calls: hdbscan.HDBSCAN().fit_predict().
    '''
    log.info(
        "HDBSCAN: min_cluster_size=%d, min_samples=%d, method=%s",
        min_cluster_size, min_samples, cluster_selection_method,
    )
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
        metric="euclidean",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(reduced)
    outlier_scores = clusterer.outlier_scores_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    log.info(
        "Found %d clusters, %d noise points (%.1f%% noise)",
        n_clusters, n_noise, 100 * n_noise / len(labels),
    )
    for cid in sorted(set(labels)):
        count = int((labels == cid).sum())
        label = "NOISE" if cid == -1 else f"Cluster {cid}"
        log.info("  %s: %d points", label, count)

    return labels, outlier_scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    '''
    Purpose: Load embeddings from JSONL, L2-normalize, two-pass UMAP
        (clustering + viz), HDBSCAN cluster, write .npz and .jsonl.
    Parameters: None (uses CLI args).
    Returns: 0 on success, 1 on error.
    Called by: CLI invocation.
    Calls: load_embeddings(), l2_normalize(), umap_cluster_pass(),
        run_hdbscan(), umap_viz_pass().
    '''
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Two-pass UMAP + HDBSCAN on window embeddings.",
    )
    parser.add_argument("--input-jsonl", required=True,
                        help="Embedding JSONL (one record per window).")
    parser.add_argument("--output-npz", required=True,
                        help="Output .npz cache of all results.")
    parser.add_argument("--output-jsonl", required=True,
                        help="Output .jsonl with per-window cluster metadata.")
    parser.add_argument(
        "--umap-components", type=int,
        default=int(os.getenv("UMAP_N_COMPONENTS", "50")),
        help="UMAP intermediate dimensions for clustering (default 50).",
    )
    parser.add_argument(
        "--n-neighbors", type=int,
        default=int(os.getenv("UMAP_N_NEIGHBORS", "50")),
        help="UMAP n_neighbors (default 50).",
    )
    parser.add_argument(
        "--viz-min-dist", type=float,
        default=float(os.getenv("UMAP_VIZ_MIN_DIST", "0.1")),
        help="UMAP min_dist for 3D viz pass (default 0.1).",
    )
    parser.add_argument(
        "--min-cluster-size", type=int,
        default=int(os.getenv("HDBSCAN_MIN_CLUSTER_SIZE", "20")),
        help="HDBSCAN min_cluster_size (default 20).",
    )
    parser.add_argument(
        "--min-samples", type=int, default=5,
        help="HDBSCAN min_samples (default 5).",
    )
    parser.add_argument(
        "--cluster-selection-method", type=str, default="eom",
        choices=["eom", "leaf"],
        help="HDBSCAN cluster_selection_method (default eom).",
    )
    parser.add_argument(
        "--seed", type=int,
        default=int(os.getenv("UMAP_RANDOM_SEED", "42")),
        help="Random seed for UMAP (default 42).",
    )
    parser.add_argument(
        "--embedding-dim", type=int, default=None,
        help="Truncate each embedding to first N dims (default: full vector).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input_jsonl):
        log.error("Input not found: %s", args.input_jsonl)
        return 1

    # 1. Load
    embeddings, window_ids, _rows = load_embeddings(
        args.input_jsonl, embedding_dim=args.embedding_dim,
    )
    if embeddings.size == 0:
        log.error("No embeddings in input.")
        return 1

    # 2. Normalize
    embeddings = l2_normalize(embeddings)

    # 3. UMAP clustering pass (high-dim)
    reduced_hd = umap_cluster_pass(
        embeddings, args.n_neighbors, args.umap_components, args.seed,
    )

    # 4. HDBSCAN on high-dim reduction
    labels, outlier_scores = run_hdbscan(
        reduced_hd, args.min_cluster_size, args.min_samples,
        args.cluster_selection_method,
    )

    # 5. UMAP viz pass (3D) — run on original embeddings, not the HD reduction
    reduced_3d = umap_viz_pass(
        embeddings, args.n_neighbors, args.viz_min_dist, args.seed,
    )

    # 6. Save .npz cache
    os.makedirs(os.path.dirname(args.output_npz) or ".", exist_ok=True)
    np.savez(
        args.output_npz,
        window_ids=np.array(window_ids),
        labels=labels,
        outlier_scores=outlier_scores,
        reduced_hd=reduced_hd,
        reduced_3d=reduced_3d,
    )
    log.info("Saved: %s", args.output_npz)

    # 7. Save .jsonl metadata
    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for i, wid in enumerate(window_ids):
            row = {
                "window_id": wid,
                "cluster_label": int(labels[i]),
                "glosh_score": float(outlier_scores[i]),
            }
            f.write(json.dumps(row) + "\n")
    log.info("Saved: %s", args.output_jsonl)

    return 0


if __name__ == "__main__":
    sys.exit(main())
