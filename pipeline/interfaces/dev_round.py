"""Shared DevRoundManifest type — produced by Module 7: Dev Dashboard.

Records exactly what each discrimination-test round sampled. Persists the
source-pool labels so the export endpoint can reveal them after rating
completes (rater sees blinded; analyst sees revealed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pipeline.interfaces.window import WindowKey


@dataclass
class DevRoundManifest:
    """One discrimination-test round: sample composition + presentation order.

    The rater sees `shuffled_order` (no source labels). The analyst calls the
    export endpoint, which joins ratings against `pools` to reveal which
    pool each rated window came from.
    """
    round_id: str                              # e.g. "round_2026-05-30T22-15-30Z"
    created_at: str                            # ISO-8601 UTC
    dataset_label: str                         # operator-provided
    pool_size: int                             # 30 per pool by default
    seed: int                                  # RNG seed — round is reproducible
    pools: dict[str, list[WindowKey]]          # {"verity"|"random"|"naive_rare": [...]}
    shuffled_order: list[WindowKey]            # presentation order, blinded
    naive_rare_atoms: list[str] = field(default_factory=list)  # top-K rarest atoms

    def to_json(self) -> dict[str, Any]:
        return {
            "round_id": self.round_id,
            "created_at": self.created_at,
            "dataset_label": self.dataset_label,
            "pool_size": self.pool_size,
            "seed": self.seed,
            "pools": {
                arm: [w.to_json() for w in windows]
                for arm, windows in self.pools.items()
            },
            "shuffled_order": [w.to_json() for w in self.shuffled_order],
            "naive_rare_atoms": list(self.naive_rare_atoms),
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "DevRoundManifest":
        return cls(
            round_id=str(d["round_id"]),
            created_at=str(d["created_at"]),
            dataset_label=str(d["dataset_label"]),
            pool_size=int(d["pool_size"]),
            seed=int(d.get("seed", 0)),
            pools={
                arm: [WindowKey.from_json(w) for w in windows]
                for arm, windows in d.get("pools", {}).items()
            },
            shuffled_order=[
                WindowKey.from_json(w) for w in d.get("shuffled_order", [])
            ],
            naive_rare_atoms=list(d.get("naive_rare_atoms", [])),
        )
