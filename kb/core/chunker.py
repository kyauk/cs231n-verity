"""Sentence-aware text chunker.

Splits text on paragraph boundaries and sentence endings, then reassembles
into chunks of roughly `max_tokens` words with `overlap` words of overlap
between consecutive chunks.  Uses a word-count heuristic (1 token ~ 0.75 words)
which is close enough for English prose and avoids a tokeniser dependency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    index: int  # position in the source document (0-based)


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")

# Rough conversion: 1 OpenAI token ≈ 0.75 English words
_TOKENS_TO_WORDS = 0.75


def chunk_text(
    text: str,
    *,
    max_tokens: int = 500,
    overlap_tokens: int = 50,
) -> list[Chunk]:
    """Split *text* into overlapping chunks of approximately *max_tokens* tokens."""
    max_words = int(max_tokens * _TOKENS_TO_WORDS)
    overlap_words = int(overlap_tokens * _TOKENS_TO_WORDS)

    sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
    if not sentences:
        return []

    chunks: list[Chunk] = []
    current_words: list[str] = []
    idx = 0

    for sentence in sentences:
        words = sentence.split()
        if current_words and len(current_words) + len(words) > max_words:
            chunks.append(Chunk(text=" ".join(current_words), index=idx))
            idx += 1
            # keep the last `overlap_words` words for context continuity
            current_words = current_words[-overlap_words:] if overlap_words else []
        current_words.extend(words)

    if current_words:
        chunks.append(Chunk(text=" ".join(current_words), index=idx))

    return chunks
