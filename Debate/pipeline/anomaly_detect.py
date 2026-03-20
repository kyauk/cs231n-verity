"""Run UMAP + HDBSCAN anomaly detection and export machine-readable artifacts."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import count
from typing import Any

import numpy as np

from pipeline.models.handoff_contracts import (
    AnomalyResultRecord,
    EmbeddingContractRecord,
)


@dataclass
class ClusteringResult:
    """Container for clustering outputs used by export and sweep steps."""

    labels: np.ndarray
    probabilities: np.ndarray
    outlier_scores: np.ndarray


def parse_args() -> argparse.Namespace:
    """
    Purpose: Parse CLI options for anomaly detection on window embeddings.
    Parameters:
        None
    Returns:
        argparse.Namespace: Parsed command-line options.
    Called by: main()
    Calls: argparse.ArgumentParser.parse_args()
    """

    parser = argparse.ArgumentParser(
        description="Run UMAP + HDBSCAN anomaly detection and export JSONL artifacts.",
    )
    parser.add_argument(
        "--input-jsonl",
        default="outputs/window_embeddings_cosmos.jsonl",
        help="Embedding JSONL input path.",
    )
    parser.add_argument(
        "--output-results-jsonl",
        default="outputs/flagged_windows.jsonl",
        help="Per-window anomaly output JSONL path.",
    )
    parser.add_argument(
        "--output-summary-json",
        default="outputs/anomaly_summary.json",
        help="Aggregate anomaly summary output path.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Optional cap on number of sampled windows for fast iteration.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling and UMAP.",
    )
    parser.add_argument(
        "--umap-n-components",
        type=int,
        default=50,
        help="UMAP output dimensionality used for clustering.",
    )
    parser.add_argument(
        "--umap-n-neighbors",
        type=int,
        default=30,
        help="UMAP n_neighbors parameter.",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.0,
        help="UMAP min_dist parameter.",
    )
    parser.add_argument(
        "--umap-metric",
        default="cosine",
        help="UMAP distance metric.",
    )
    parser.add_argument(
        "--hdbscan-min-cluster-size",
        type=int,
        default=15,
        help="HDBSCAN min_cluster_size parameter.",
    )
    parser.add_argument(
        "--hdbscan-min-samples",
        type=int,
        default=5,
        help="HDBSCAN min_samples parameter.",
    )
    parser.add_argument(
        "--hdbscan-metric",
        default="euclidean",
        help="HDBSCAN distance metric.",
    )
    return parser.parse_args()


def _to_float(value: Any, default: float = 0.0) -> float:
    """
    Purpose: Convert arbitrary values to float with safe fallback.
    Parameters:
        value (Any): Input value to cast.
        default (float): Value returned when cast fails.
    Returns:
        float: Parsed float or default.
    Called by: run_hdbscan()
    Calls: float()
    """

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_embeddings(
    input_jsonl: str,
    max_points: int | None,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """
    Purpose: Load embeddings and source metadata rows from JSONL.
    Parameters:
        input_jsonl (str): Path to JSONL with WindowEmbeddingRecord-like rows.
        max_points (int | None): Optional random sample cap.
        seed (int): Sampling seed.
    Returns:
        tuple[np.ndarray, list[dict[str, Any]]]: Embedding matrix and aligned row metadata.
    Called by: main()
    Calls: open(), json.loads(), random.Random.sample()
    """

    rows: list[dict[str, Any]] = []
    with open(input_jsonl, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw_row = json.loads(line)
            validated = EmbeddingContractRecord.model_validate(raw_row)
            rows.append(validated.model_dump())

    if len(rows) == 0:
        raise ValueError(f"No rows found in input file: {input_jsonl}")

    if max_points is not None and len(rows) > max_points:
        rng = random.Random(seed)
        rows = rng.sample(rows, max_points)

    embeddings = [row.get("embedding", []) for row in rows]
    matrix = np.array(embeddings, dtype=np.float32)

    if matrix.ndim != 2:
        raise ValueError("Embeddings are not a 2D matrix.")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("Embeddings matrix is empty.")
    if not np.isfinite(matrix).all():
        raise ValueError("Embeddings contain NaN or infinite values.")

    return matrix, rows


def reduce_with_umap(
    embeddings: np.ndarray,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    seed: int,
) -> np.ndarray:
    """
    Purpose: Reduce high-dimensional embeddings for density-based clustering.
    Parameters:
        embeddings (np.ndarray): Input matrix with shape (n_rows, n_dims).
        n_components (int): Target UMAP dimensionality.
        n_neighbors (int): UMAP local-neighbor count.
        min_dist (float): UMAP minimum distance in projected space.
        metric (str): UMAP metric name.
        seed (int): Random seed for reproducibility.
    Returns:
        np.ndarray: Reduced matrix with shape (n_rows, n_components).
    Called by: main(), pipeline/anomaly_sweep.py -> run_single_configuration()
    Calls: umap.UMAP().fit_transform()
    """

    import umap

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )
    return reducer.fit_transform(embeddings)


def run_hdbscan(
    reduced_embeddings: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    metric: str,
) -> ClusteringResult:
    """
    Purpose: Cluster reduced embeddings and extract anomaly signals.
    Parameters:
        reduced_embeddings (np.ndarray): Matrix from UMAP with shape (n_rows, n_reduced_dims).
        min_cluster_size (int): HDBSCAN minimum cluster size.
        min_samples (int): HDBSCAN density sensitivity parameter.
        metric (str): HDBSCAN distance metric.
    Returns:
        ClusteringResult: Labels, membership probabilities, and outlier scores.
    Called by: main(), pipeline/anomaly_sweep.py -> run_single_configuration()
    Calls: hdbscan.HDBSCAN().fit()
    """

    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
    )
    clusterer.fit(reduced_embeddings)

    labels = np.array(clusterer.labels_, dtype=np.int32)
    probabilities = np.array(
        [_to_float(value, 0.0) for value in getattr(clusterer, "probabilities_", [])],
        dtype=np.float32,
    )
    outlier_scores = np.array(
        [_to_float(value, 0.0) for value in getattr(clusterer, "outlier_scores_", [])],
        dtype=np.float32,
    )

    if probabilities.shape[0] != labels.shape[0]:
        probabilities = np.zeros(labels.shape[0], dtype=np.float32)
    if outlier_scores.shape[0] != labels.shape[0]:
        outlier_scores = np.zeros(labels.shape[0], dtype=np.float32)

    outlier_scores = np.nan_to_num(outlier_scores, nan=0.0, posinf=1.0, neginf=0.0)
    probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=1.0, neginf=0.0)

    return ClusteringResult(
        labels=labels,
        probabilities=probabilities,
        outlier_scores=outlier_scores,
    )


def rank_anomalies(labels: np.ndarray, outlier_scores: np.ndarray) -> np.ndarray:
    """
    Purpose: Assign anomaly rank where rank 1 is most anomalous row.
    Parameters:
        labels (np.ndarray): Cluster labels where noise is -1.
        outlier_scores (np.ndarray): HDBSCAN outlier scores.
    Returns:
        np.ndarray: Rank array aligned with input rows.
    Called by: build_result_rows()
    Calls: numpy.argsort()
    """

    # Give noise points priority in ranking; within each group sort by score descending.
    # This keeps top anomalies easy to inspect while still ranking clustered outliers.
    sort_keys = np.lexsort((-outlier_scores, labels != -1))
    ranks = np.zeros(labels.shape[0], dtype=np.int32)
    for rank, index in zip(count(1), sort_keys):
        ranks[index] = rank
    return ranks


def build_result_rows(
    source_rows: list[dict[str, Any]],
    clustering: ClusteringResult,
) -> list[dict[str, Any]]:
    """
    Purpose: Build clean per-window anomaly records for downstream consumers.
    Parameters:
        source_rows (list[dict[str, Any]]): Input metadata aligned to embeddings.
        clustering (ClusteringResult): HDBSCAN output arrays.
    Returns:
        list[dict[str, Any]]: Output rows with labels, scores, and metadata pass-through.
    Called by: main()
    Calls: rank_anomalies()
    """

    labels = clustering.labels
    probabilities = clustering.probabilities
    outlier_scores = clustering.outlier_scores
    ranks = rank_anomalies(labels, outlier_scores)

    result_rows: list[dict[str, Any]] = []
    for index, row in enumerate(source_rows):
        output_row = AnomalyResultRecord(
            window_id=row.get("window_id"),
            scene_token_hex=row.get("scene_token_hex"),
            log_id=row.get("log_id"),
            scenario_tags=row.get("scenario_tags", []),
            window_start_ts=row.get("window_start_ts"),
            window_end_ts=row.get("window_end_ts"),
            cluster_label=int(labels[index]),
            is_noise=bool(labels[index] == -1),
            cluster_probability=float(probabilities[index]),
            outlier_score=float(outlier_scores[index]),
            anomaly_rank=int(ranks[index]),
            quality=row.get("quality", {}),
            metadata=row.get("metadata", {}),
        ).model_dump()
        result_rows.append(output_row)
    return result_rows


def build_summary(
    result_rows: list[dict[str, Any]],
    reduced_shape: tuple[int, int],
    embedding_dim: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """
    Purpose: Create aggregate metrics for quick run quality inspection.
    Parameters:
        result_rows (list[dict[str, Any]]): Per-window anomaly rows.
        reduced_shape (tuple[int, int]): Shape of reduced UMAP matrix.
        embedding_dim (int): Original embedding dimensionality.
        args (argparse.Namespace): Runtime parameters.
    Returns:
        dict[str, Any]: Summary metrics and parameter snapshot.
    Called by: main()
    Calls: collections.Counter()
    """

    labels = [int(row["cluster_label"]) for row in result_rows]
    label_counter = Counter(labels)
    noise_count = int(label_counter.get(-1, 0))
    cluster_count = int(len([key for key in label_counter.keys() if key != -1]))
    noise_ratio = float(noise_count / len(result_rows)) if len(result_rows) else 0.0

    summary = {
        "run_timestamp_utc": datetime.now(UTC).isoformat(),
        "input_jsonl": args.input_jsonl,
        "rows_processed": len(result_rows),
        "embedding_dim": embedding_dim,
        "reduced_dim": reduced_shape[1],
        "cluster_count": cluster_count,
        "noise_count": noise_count,
        "noise_ratio": noise_ratio,
        "cluster_size_histogram": {str(label): int(count) for label, count in label_counter.items()},
        "umap": {
            "n_components": args.umap_n_components,
            "n_neighbors": args.umap_n_neighbors,
            "min_dist": args.umap_min_dist,
            "metric": args.umap_metric,
            "seed": args.seed,
        },
        "hdbscan": {
            "min_cluster_size": args.hdbscan_min_cluster_size,
            "min_samples": args.hdbscan_min_samples,
            "metric": args.hdbscan_metric,
        },
        "warnings": [],
    }

    if cluster_count == 0:
        summary["warnings"].append("all_noise")
    if cluster_count == 1:
        summary["warnings"].append("single_cluster")
    if math.isclose(noise_ratio, 1.0):
        summary["warnings"].append("noise_ratio_is_1")

    return summary


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    """
    Purpose: Write newline-delimited JSON records to disk.
    Parameters:
        path (str): Output JSONL path.
        rows (list[dict[str, Any]]): Records to serialize.
    Returns:
        None
    Called by: main()
    Calls: os.makedirs(), json.dumps()
    """

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def write_json(path: str, payload: dict[str, Any]) -> None:
    """
    Purpose: Write one JSON object to disk with indentation.
    Parameters:
        path (str): Output JSON path.
        payload (dict[str, Any]): Object to serialize.
    Returns:
        None
    Called by: main()
    Calls: os.makedirs(), json.dump()
    """

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> int:
    """
    Purpose: Execute anomaly pipeline and export rows + summary artifacts.
    Parameters:
        None
    Returns:
        int: Exit status code (0 success, 1 failure).
    Called by: CLI entrypoint
    Calls: parse_args(), load_embeddings(), reduce_with_umap(), run_hdbscan(),
        build_result_rows(), build_summary(), write_jsonl(), write_json()
    """

    args = parse_args()
    try:
        embeddings, source_rows = load_embeddings(
            input_jsonl=args.input_jsonl,
            max_points=args.max_points,
            seed=args.seed,
        )
        reduced = reduce_with_umap(
            embeddings=embeddings,
            n_components=args.umap_n_components,
            n_neighbors=args.umap_n_neighbors,
            min_dist=args.umap_min_dist,
            metric=args.umap_metric,
            seed=args.seed,
        )
        clustering = run_hdbscan(
            reduced_embeddings=reduced,
            min_cluster_size=args.hdbscan_min_cluster_size,
            min_samples=args.hdbscan_min_samples,
            metric=args.hdbscan_metric,
        )
        result_rows = build_result_rows(source_rows=source_rows, clustering=clustering)
        summary = build_summary(
            result_rows=result_rows,
            reduced_shape=reduced.shape,
            embedding_dim=embeddings.shape[1],
            args=args,
        )

        write_jsonl(args.output_results_jsonl, result_rows)
        write_json(args.output_summary_json, summary)

        print(f"Processed rows: {summary['rows_processed']}")
        print(f"Clusters: {summary['cluster_count']} | Noise: {summary['noise_count']}")
        print(f"Saved results JSONL: {args.output_results_jsonl}")
        print(f"Saved summary JSON: {args.output_summary_json}")
        if summary["warnings"]:
            print(f"Warnings: {', '.join(summary['warnings'])}")
        return 0
    except Exception as error:  # noqa: BLE001
        print(f"Error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
