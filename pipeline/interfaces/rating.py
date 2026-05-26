"""Shared Rating type — produced by Module 5: Judge UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pipeline.interfaces.window import WindowKey


@dataclass
class Rating:
    """One human rater's evaluation of one proposal."""
    rater_id: str
    proposal_id: str
    arm: str                            # blinded to rater; recorded server-side
    coherence_score: int               # 1–5
    usefulness_score: int              # 1–5
    timestamp: str                     # ISO-8601
    free_text_note: str | None = None
    seen_motivating_scenes: list[WindowKey] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "rater_id": self.rater_id,
            "proposal_id": self.proposal_id,
            "arm": self.arm,
            "coherence_score": self.coherence_score,
            "usefulness_score": self.usefulness_score,
            "timestamp": self.timestamp,
            "free_text_note": self.free_text_note,
            "seen_motivating_scenes": [w.to_json() for w in self.seen_motivating_scenes],
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Rating":
        return cls(
            rater_id=str(d["rater_id"]),
            proposal_id=str(d["proposal_id"]),
            arm=str(d["arm"]),
            coherence_score=int(d["coherence_score"]),
            usefulness_score=int(d["usefulness_score"]),
            timestamp=str(d["timestamp"]),
            free_text_note=d.get("free_text_note"),
            seen_motivating_scenes=[
                WindowKey.from_json(w) for w in d.get("seen_motivating_scenes", [])
            ],
        )
