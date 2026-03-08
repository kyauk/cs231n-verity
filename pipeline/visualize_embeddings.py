"""Reduce window embeddings to 2D (UMAP or PCA) and plot for visual inspection.

Optionally runs HDBSCAN for anomaly/cluster labels and colors the scatter by cluster.
Input: window embedding JSONL (e.g. window_embeddings_cosmos.jsonl).
Output: PNG and optionally interactive Plotly HTML.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np


def parse_args() -> argparse.Namespace:
    '''
    Purpose: Parse CLI options for embedding visualization.
    Parameters: None.
    Returns: argparse.Namespace with input, output, reducer, and optional flags.
    Called by: main().
    Calls: argparse.ArgumentParser().parse_args().
    '''
    p = argparse.ArgumentParser(
        description="Visualize window embeddings via 2D reduction and scatter plot.",
    )
    p.add_argument(
        "--input-jsonl",
        default="outputs/window_embeddings_cosmos.jsonl",
        help="Path to embedding JSONL (default: outputs/window_embeddings_cosmos.jsonl).",
    )
    p.add_argument(
        "--output",
        default="outputs/embedding_umap.png",
        help="Path for static PNG (default: outputs/embedding_umap.png).",
    )
    p.add_argument(
        "--output-html",
        default=None,
        help="If set, also write interactive Plotly HTML with hover.",
    )
    p.add_argument(
        "--reducer",
        default="umap",
        choices=["umap", "pca"],
        help="2D reduction method (default: umap).",
    )
    p.add_argument(
        "--run-hdbscan",
        action="store_true",
        help="Run HDBSCAN on 2D coords and color points by cluster (noise = -1).",
    )
    p.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Cap number of points to plot (default: no cap).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for UMAP and sampling (default: 42).",
    )
    return p.parse_args()


def load_embeddings_from_jsonl(
    path: str,
    max_points: int | None,
    seed: int,
) -> tuple[np.ndarray, list[str], list[dict]]:
    '''
    Purpose: Load embedding matrix and ids from JSONL.
    Parameters:
        path (str): Path to JSONL file.
        max_points (int | None): If set, cap number of rows (random sample).
        seed (int): Random seed for sampling.
    Returns:
        tuple: (X shape (n, dim), list of window_id, list of row dicts for hover).
    Called by: main().
    Calls: open(), json.loads().
    '''
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    if not rows:
        return np.array([]).reshape(0, 0), [], []

    if max_points is not None and len(rows) > max_points:
        rng = random.Random(seed)
        rows = rng.sample(rows, max_points)

    embeddings = [r["embedding"] for r in rows]
    window_ids = [r.get("window_id", "") for r in rows]
    X = np.array(embeddings, dtype=np.float32)
    return X, window_ids, rows


def reduce_2d(X: np.ndarray, reducer: str, seed: int) -> np.ndarray:
    '''
    Purpose: Reduce embedding matrix to 2D with UMAP or PCA.
    Parameters:
        X (np.ndarray): Shape (n, dim).
        reducer (str): "umap" or "pca".
        seed (int): Random state for UMAP.
    Returns:
        np.ndarray: Shape (n, 2).
    Called by: main().
    Calls: umap.UMAP, sklearn.decomposition.PCA.
    '''
    if reducer == "umap":
        import umap
        reducer_obj = umap.UMAP(
            n_components=2,
            metric="cosine",
            random_state=seed,
        )
        return reducer_obj.fit_transform(X)
    # pca
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=seed)
    return pca.fit_transform(X)


def run_hdbscan(coords_2d: np.ndarray, min_cluster_size: int = 5) -> np.ndarray:
    '''
    Purpose: Run HDBSCAN on 2D coords; return labels (noise = -1).
    Parameters:
        coords_2d (np.ndarray): Shape (n, 2).
        min_cluster_size (int): HDBSCAN min_cluster_size.
    Returns:
        np.ndarray: Integer labels, shape (n,).
    Called by: main().
    Calls: hdbscan.HDBSCAN().
    '''
    import hdbscan
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
    )
    clusterer.fit(coords_2d)
    return clusterer.labels_


def plot_matplotlib(
    coords_2d: np.ndarray,
    labels: np.ndarray | None,
    output_path: str,
    title: str,
) -> None:
    '''
    Purpose: Save 2D scatter plot as PNG.
    Parameters:
        coords_2d (np.ndarray): Shape (n, 2).
        labels (np.ndarray | None): If set, color by label (noise = -1).
        output_path (str): Path for PNG.
        title (str): Plot title.
    Returns: None.
    Called by: main().
    Calls: matplotlib.pyplot.
    '''
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    if labels is not None:
        noise = labels == -1
        ax.scatter(
            coords_2d[~noise, 0],
            coords_2d[~noise, 1],
            c=labels[~noise],
            cmap="tab20",
            s=8,
            alpha=0.7,
        )
        ax.scatter(
            coords_2d[noise, 0],
            coords_2d[noise, 1],
            c="gray",
            s=8,
            alpha=0.5,
            label="noise",
        )
        ax.legend()
    else:
        ax.scatter(coords_2d[:, 0], coords_2d[:, 1], s=8, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_plotly(
    coords_2d: np.ndarray,
    labels: np.ndarray | None,
    rows: list[dict],
    output_path: str,
    title: str,
) -> None:
    '''
    Purpose: Save interactive Plotly HTML with hover (window_id, log_id).
    Parameters:
        coords_2d (np.ndarray): Shape (n, 2).
        labels (np.ndarray | None): Optional cluster labels.
        rows (list[dict]): Original row dicts for hover text.
        output_path (str): Path for HTML.
        title (str): Plot title.
    Returns: None.
    Called by: main().
    Calls: plotly.graph_objects.
    '''
    import plotly.express as px
    import plotly.graph_objects as go
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    hover_text = [
        f"window_id: {r.get('window_id', '')}<br>log_id: {r.get('log_id', '')}"
        for r in rows
    ]
    if labels is not None:
        fig = go.Figure()
        noise = labels == -1
        if np.any(~noise):
            fig.add_trace(go.Scatter(
                x=coords_2d[~noise, 0],
                y=coords_2d[~noise, 1],
                mode="markers",
                marker=dict(color=labels[~noise], colorscale="tab20", size=4),
                text=[hover_text[i] for i in np.where(~noise)[0]],
                hoverinfo="text",
                name="clusters",
            ))
        if np.any(noise):
            fig.add_trace(go.Scatter(
                x=coords_2d[noise, 0],
                y=coords_2d[noise, 1],
                mode="markers",
                marker=dict(color="gray", size=4),
                text=[hover_text[i] for i in np.where(noise)[0]],
                hoverinfo="text",
                name="noise",
            ))
    else:
        fig = go.Figure(go.Scatter(
            x=coords_2d[:, 0],
            y=coords_2d[:, 1],
            mode="markers",
            marker=dict(size=4),
            text=hover_text,
            hoverinfo="text",
        ))
    fig.update_layout(title=title, xaxis_title="Component 1", yaxis_title="Component 2")
    fig.write_html(output_path)


def main() -> int:
    '''
    Purpose: Load embeddings, reduce to 2D, optionally run HDBSCAN, plot and save.
    Parameters: None (uses parse_args()).
    Returns: 0 on success, 1 on error.
    Called by: CLI invocation.
    Calls: load_embeddings_from_jsonl(), reduce_2d(), run_hdbscan(), plot_matplotlib(),
        plot_plotly().
    '''
    args = parse_args()
    if not os.path.isfile(args.input_jsonl):
        print(f"Error: input file not found: {args.input_jsonl}", file=sys.stderr)
        return 1

    X, window_ids, rows = load_embeddings_from_jsonl(
        args.input_jsonl,
        args.max_points,
        args.seed,
    )
    if X.size == 0:
        print("Error: no embeddings in input.", file=sys.stderr)
        return 1
    n, dim = X.shape
    print(f"Loaded {n} embeddings (dim={dim})")

    coords_2d = reduce_2d(X, args.reducer, args.seed)
    labels = None
    if args.run_hdbscan:
        try:
            labels = run_hdbscan(coords_2d)
            n_noise = int(np.sum(labels == -1))
            n_clusters = len(set(labels)) - (1 if -1 in set(labels) else 0)
            print(f"HDBSCAN: {n_clusters} clusters, {n_noise} noise points")
        except ImportError:
            print("Warning: hdbscan not installed; skipping --run-hdbscan", file=sys.stderr)

    title = f"Window embeddings ({args.reducer} 2D)" + (" + HDBSCAN" if labels is not None else "")
    plot_matplotlib(coords_2d, labels, args.output, title)
    print(f"Saved: {args.output}")

    if args.output_html:
        try:
            plot_plotly(coords_2d, labels, rows, args.output_html, title)
            print(f"Saved: {args.output_html}")
        except ImportError:
            print("Warning: plotly not installed; skipping --output-html", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
