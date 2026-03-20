"""Grid sweep UMAP + HDBSCAN parameters and export quick tuning metrics."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
from datetime import UTC, datetime
from typing import Any

from pipeline.anomaly_detect import (
    build_result_rows,
    load_embeddings,
    reduce_with_umap,
    run_hdbscan,
)


def parse_args() -> argparse.Namespace:
    """
    Purpose: Parse CLI args for anomaly parameter sweep.
    Parameters:
        None
    Returns:
        argparse.Namespace: Parsed arguments.
    Called by: main()
    Calls: argparse.ArgumentParser.parse_args()
    """

    parser = argparse.ArgumentParser(
        description="Sweep UMAP + HDBSCAN settings and write summary table.",
    )
    parser.add_argument(
        "--input-jsonl",
        default="outputs/window_embeddings_cosmos.jsonl",
        help="Embedding JSONL input path.",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/anomaly_sweep_summary.csv",
        help="CSV file path for all run metrics.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=2000,
        help="Optional sample cap for faster sweep runs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling and UMAP.",
    )
    parser.add_argument(
        "--umap-neighbors-grid",
        default="15,30,50",
        help="Comma-separated UMAP n_neighbors values.",
    )
    parser.add_argument(
        "--umap-min-dist-grid",
        default="0.0,0.1,0.3",
        help="Comma-separated UMAP min_dist values.",
    )
    parser.add_argument(
        "--hdbscan-min-cluster-size-grid",
        default="10,15,25",
        help="Comma-separated HDBSCAN min_cluster_size values.",
    )
    parser.add_argument(
        "--hdbscan-min-samples-grid",
        default="3,5,10",
        help="Comma-separated HDBSCAN min_samples values.",
    )
    parser.add_argument(
        "--umap-n-components",
        type=int,
        default=50,
        help="Reduced dimensionality used before clustering.",
    )
    parser.add_argument(
        "--umap-metric",
        default="cosine",
        help="UMAP metric.",
    )
    parser.add_argument(
        "--hdbscan-metric",
        default="euclidean",
        help="HDBSCAN metric.",
    )
    return parser.parse_args()


def _parse_int_grid(raw: str) -> list[int]:
    """
    Purpose: Parse comma-separated integers into list.
    Parameters:
        raw (str): Comma-separated int string.
    Returns:
        list[int]: Parsed integer values.
    Called by: main()
    Calls: str.split(), int()
    """

    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def _parse_float_grid(raw: str) -> list[float]:
    """
    Purpose: Parse comma-separated floats into list.
    Parameters:
        raw (str): Comma-separated float string.
    Returns:
        list[float]: Parsed float values.
    Called by: main()
    Calls: str.split(), float()
    """

    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def _compute_metrics(result_rows: list[dict[str, Any]]) -> dict[str, float]:
    """
    Purpose: Compute sweep metrics from per-window anomaly rows.
    Parameters:
        result_rows (list[dict[str, Any]]): Output rows from one config.
    Returns:
        dict[str, float]: Cluster/noise/anomaly quality summary.
    Called by: run_single_configuration()
    Calls: None
    """

    total = len(result_rows)
    labels = [int(row["cluster_label"]) for row in result_rows]
    noise_count = sum(1 for label in labels if label == -1)
    cluster_count = len({label for label in labels if label != -1})
    top10_mean_outlier = 0.0

    if total > 0:
        scores = sorted(
            (float(row["outlier_score"]) for row in result_rows),
            reverse=True,
        )
        top = scores[:10]
        if top:
            top10_mean_outlier = sum(top) / len(top)

    return {
        "rows_processed": float(total),
        "cluster_count": float(cluster_count),
        "noise_count": float(noise_count),
        "noise_ratio": (float(noise_count) / total) if total else 0.0,
        "top10_mean_outlier_score": top10_mean_outlier,
    }


def run_single_configuration(
    embeddings,
    rows,
    args: argparse.Namespace,
    umap_neighbors: int,
    umap_min_dist: float,
    hdbscan_min_cluster_size: int,
    hdbscan_min_samples: int,
) -> dict[str, Any]:
    """
    Purpose: Execute one UMAP + HDBSCAN config and return row for sweep CSV.
    Parameters:
        embeddings (np.ndarray): Input embedding matrix.
        rows (list[dict[str, Any]]): Input metadata rows aligned with embeddings.
        args (argparse.Namespace): Shared runtime arguments.
        umap_neighbors (int): UMAP n_neighbors value for this run.
        umap_min_dist (float): UMAP min_dist value for this run.
        hdbscan_min_cluster_size (int): HDBSCAN min_cluster_size value.
        hdbscan_min_samples (int): HDBSCAN min_samples value.
    Returns:
        dict[str, Any]: Flat summary row for CSV.
    Called by: main()
    Calls: reduce_with_umap(), run_hdbscan(), build_result_rows(), _compute_metrics()
    """

    reduced = reduce_with_umap(
        embeddings=embeddings,
        n_components=args.umap_n_components,
        n_neighbors=umap_neighbors,
        min_dist=umap_min_dist,
        metric=args.umap_metric,
        seed=args.seed,
    )
    clustering = run_hdbscan(
        reduced_embeddings=reduced,
        min_cluster_size=hdbscan_min_cluster_size,
        min_samples=hdbscan_min_samples,
        metric=args.hdbscan_metric,
    )
    result_rows = build_result_rows(source_rows=rows, clustering=clustering)
    metrics = _compute_metrics(result_rows)

    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "umap_n_neighbors": umap_neighbors,
        "umap_min_dist": umap_min_dist,
        "umap_n_components": args.umap_n_components,
        "hdbscan_min_cluster_size": hdbscan_min_cluster_size,
        "hdbscan_min_samples": hdbscan_min_samples,
        "umap_metric": args.umap_metric,
        "hdbscan_metric": args.hdbscan_metric,
        **metrics,
    }


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    """
    Purpose: Write sweep result rows to CSV.
    Parameters:
        path (str): Output CSV path.
        rows (list[dict[str, Any]]): Sweep summary rows.
    Returns:
        None
    Called by: main()
    Calls: csv.DictWriter()
    """

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if not rows:
        raise ValueError("No sweep rows to write.")
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    """
    Purpose: Run full parameter sweep and export tuning metrics CSV.
    Parameters:
        None
    Returns:
        int: Exit status code.
    Called by: CLI entrypoint
    Calls: parse_args(), load_embeddings(), run_single_configuration(), write_csv()
    """

    args = parse_args()
    try:
        embeddings, rows = load_embeddings(
            input_jsonl=args.input_jsonl,
            max_points=args.max_points,
            seed=args.seed,
        )

        neighbors_grid = _parse_int_grid(args.umap_neighbors_grid)
        min_dist_grid = _parse_float_grid(args.umap_min_dist_grid)
        min_cluster_grid = _parse_int_grid(args.hdbscan_min_cluster_size_grid)
        min_samples_grid = _parse_int_grid(args.hdbscan_min_samples_grid)

        sweep_rows: list[dict[str, Any]] = []
        for config in itertools.product(
            neighbors_grid,
            min_dist_grid,
            min_cluster_grid,
            min_samples_grid,
        ):
            (
                umap_neighbors,
                umap_min_dist,
                hdbscan_min_cluster_size,
                hdbscan_min_samples,
            ) = config
            sweep_row = run_single_configuration(
                embeddings=embeddings,
                rows=rows,
                args=args,
                umap_neighbors=umap_neighbors,
                umap_min_dist=umap_min_dist,
                hdbscan_min_cluster_size=hdbscan_min_cluster_size,
                hdbscan_min_samples=hdbscan_min_samples,
            )
            sweep_rows.append(sweep_row)
            print(
                "done",
                json.dumps(
                    {
                        "neighbors": umap_neighbors,
                        "min_dist": umap_min_dist,
                        "min_cluster_size": hdbscan_min_cluster_size,
                        "min_samples": hdbscan_min_samples,
                        "clusters": sweep_row["cluster_count"],
                        "noise_ratio": round(sweep_row["noise_ratio"], 4),
                    }
                ),
            )

        write_csv(args.output_csv, sweep_rows)
        print(f"Saved sweep summary: {args.output_csv}")
        print(f"Total configs: {len(sweep_rows)}")
        return 0
    except Exception as error:  # noqa: BLE001
        print(f"Error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
