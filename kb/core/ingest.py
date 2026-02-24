"""Ingestion pipeline: chunk -> embed -> store in pgvector."""

from __future__ import annotations

from typing import Sequence

from .chunker import chunk_text
from .config import get_db_conn
from .embeddings import embed_texts

_INSERT_SQL = """
    INSERT INTO chunks (source_id, chunk_text, embedding)
    VALUES (%s, %s, %s)
"""


def ingest_text(source_id: str, text: str) -> int:
    """Chunk *text*, embed every chunk, and bulk-insert into the chunks table.

    Returns the number of rows inserted.
    """
    chunks = chunk_text(text)
    if not chunks:
        return 0
    return ingest_chunks(source_id, [c.text for c in chunks])


def ingest_chunks(source_id: str, chunk_texts: Sequence[str]) -> int:
    """Embed pre-split *chunk_texts* and bulk-insert into the chunks table.

    Returns the number of rows inserted.
    """
    if not chunk_texts:
        return 0

    embeddings = embed_texts(chunk_texts)

    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.executemany(
                _INSERT_SQL,
                [
                    (source_id, text, emb.tolist())
                    for text, emb in zip(chunk_texts, embeddings)
                ],
            )
        return len(chunk_texts)
    finally:
        conn.close()
