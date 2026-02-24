"""Thin wrapper around OpenAI text-embedding-3-small with automatic batching."""

from __future__ import annotations
from typing import Sequence
import numpy as np
from .config import get_openai_client

MODEL = "text-embedding-3-small"
DIMENSIONS = 1536
_BATCH_SIZE = 100  # stay well under the 2048-input API limit


def embed_texts(texts: Sequence[str]) -> list[np.ndarray]:
    """Embed a list of texts, batching transparently.

    Returns a list of numpy arrays, one per input text, each of shape (1536,).
    """
    client = get_openai_client()
    all_embeddings: list[np.ndarray] = []

    for start in range(0, len(texts), _BATCH_SIZE):
        batch = list(texts[start : start + _BATCH_SIZE])
        response = client.embeddings.create(model=MODEL, input=batch)
        for item in sorted(response.data, key=lambda d: d.index):
            all_embeddings.append(np.array(item.embedding, dtype=np.float32))

    return all_embeddings


def embed_query(text: str) -> np.ndarray:
    """Embed a single query string. Returns a (1536,) numpy array."""
    return embed_texts([text])[0]
