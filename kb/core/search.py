"""Cosine-similarity search over the chunks table using pgvector's HNSW index."""

from __future__ import annotations

from typing import Any

from .config import get_db_conn
from .embeddings import embed_query

_SEARCH_SQL = """
    SELECT id, source_id, chunk_text,
           1 - (embedding <=> %s::vector) AS similarity
    FROM chunks
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> %s::vector
    LIMIT %s
"""


def search(query_text: str, *, top_k: int = 5) -> list[dict[str, Any]]:
    """Return the *top_k* most similar chunks to *query_text*.

    Each result is a dict with keys: id, source_id, chunk_text, similarity.
    Similarity is in [0, 1] where 1 = identical direction.
    """
    query_vec = embed_query(query_text)
    vec_literal = query_vec.tolist()

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(_SEARCH_SQL, (vec_literal, vec_literal, top_k))
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()
