"""Curator configuration and errors.

The curator turns immutable RawDescriptor evidence into a versioned Taxonomy of
CanonicalLabels — deterministically, given (descriptors, base taxonomy, seed).
All knobs live here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CuratorConfig:
    """Knobs for canonicalization + projection. Distances are cosine distance
    in [0, 2] (1 - cosine similarity); thresholds are radii in that space.

    The mint guard is intentionally TWO gates, both required:
      * support_threshold  — a cluster must recur this many times. (Support alone
        mints noise-labels from frequent phrasings.)
      * cohesion_threshold — the cluster must be tight enough to be ONE concept.
        (Cohesion alone mints tight-but-rare single-scene flukes.)
    Requiring both rejects both failure modes.
    """
    match_threshold: float = 0.30      # assign a descriptor to an existing label if within this
    cohesion_threshold: float = 0.30   # a mintable cluster's radius must be <= this (one concept)
    support_threshold: int = 3         # a cluster must have >= this many descriptors to mint
    merge_threshold: float = 0.12      # two labels closer than this are merged (older id wins)
    seed: int = 42                     # processing-order seed; determinism is per-seed

    def with_seed(self, seed: int) -> "CuratorConfig":
        return CuratorConfig(
            match_threshold=self.match_threshold,
            cohesion_threshold=self.cohesion_threshold,
            support_threshold=self.support_threshold,
            merge_threshold=self.merge_threshold,
            seed=seed,
        )


class CuratorError(Exception):
    """Base class for curator failures."""


class EmbeddingDimMismatchError(CuratorError):
    """Descriptors / labels disagree on embedding dimension."""
