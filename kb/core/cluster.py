"""K-Means clustering over stored chunk embeddings."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .config import get_db_conn

_FETCH_EMBEDDINGS_SQL = """
    SELECT id, embedding::text
    FROM chunks
    WHERE embedding IS NOT NULL
"""

_UPDATE_CLUSTER_SQL = """
    UPDATE chunks SET cluster_id = %s WHERE id = %s
"""

_FETCH_CLUSTER_SQL = """
    SELECT id, source_id, chunk_text, cluster_id
    FROM chunks
    WHERE cluster_id = %s
    ORDER BY id
"""


def _parse_pgvector(text: str) -> np.ndarray:
    """Convert pgvector's text representation '[0.1,0.2,...]' to a numpy array."""
    return np.fromstring(text.strip("[]"), sep=",", dtype=np.float32)


def cluster_chunks(n_clusters: int = 5) -> dict[str, Any]:
    """Run K-Means over every embedded chunk and write cluster_id back to the DB.

    Returns a summary dict with:
      - n_clusters: number of clusters used
      - silhouette: silhouette score (-1 to 1, higher is better)
      - clusters: list of {cluster_id, count, representative} dicts
    """
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(_FETCH_EMBEDDINGS_SQL)
            rows = cur.fetchall()

        if len(rows) < n_clusters:
            raise ValueError(
                f"Only {len(rows)} embedded chunks in DB, need at least {n_clusters} "
                f"for {n_clusters} clusters."
            )

        ids = [r[0] for r in rows]
        matrix = np.stack([_parse_pgvector(r[1]) for r in rows])

        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = km.fit_predict(matrix)

        sil = float(silhouette_score(matrix, labels)) if n_clusters > 1 else 0.0

        with conn.cursor() as cur:
            cur.executemany(
                _UPDATE_CLUSTER_SQL,
                [(int(label), int(row_id)) for label, row_id in zip(labels, ids)],
            )

        # Build per-cluster summaries: pick the chunk closest to centroid
        summaries: list[dict[str, Any]] = []
        for cid in range(n_clusters):
            mask = labels == cid
            cluster_ids = [ids[i] for i, m in enumerate(mask) if m]
            cluster_vecs = matrix[mask]
            centroid = km.cluster_centers_[cid]
            dists = np.linalg.norm(cluster_vecs - centroid, axis=1)
            closest_idx = int(np.argmin(dists))
            closest_db_id = cluster_ids[closest_idx]

            with conn.cursor() as cur:
                cur.execute(
                    "SELECT chunk_text FROM chunks WHERE id = %s", (closest_db_id,)
                )
                rep_text = cur.fetchone()[0]

            summaries.append(
                {
                    "cluster_id": cid,
                    "count": int(mask.sum()),
                    "representative": rep_text[:200],
                }
            )

        return {
            "n_clusters": n_clusters,
            "silhouette": round(sil, 4),
            "clusters": summaries,
        }
    finally:
        conn.close()


def get_cluster(cluster_id: int) -> list[dict[str, Any]]:
    """Fetch all chunks assigned to *cluster_id*."""
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(_FETCH_CLUSTER_SQL, (cluster_id,))
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()
