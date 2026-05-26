"""Shared proposal types — produced by Module 3: Hypothesizer and Module 4: Scorer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pipeline.interfaces.window import WindowKey


@dataclass
class CompositionProposal:
    """A compositionally novel scenario proposed by the Hypothesizer."""
    composition_id: str                 # deterministic hash of constituents
    constituents: list[str]            # condition tags from vocabulary
    marginal_frequencies: dict[str, float]
    pairwise_frequencies: dict[str, float]
    expected_joint: float
    observed_joint: float
    novelty_score: float               # log(expected / max(observed, epsilon))
    motivating_scene_ids: list[WindowKey]
    arm: str                           # "reasoning"

    def to_json(self) -> dict[str, Any]:
        return {
            "composition_id": self.composition_id,
            "constituents": self.constituents,
            "marginal_frequencies": self.marginal_frequencies,
            "pairwise_frequencies": self.pairwise_frequencies,
            "expected_joint": self.expected_joint,
            "observed_joint": self.observed_joint,
            "novelty_score": self.novelty_score,
            "motivating_scene_ids": [w.to_json() for w in self.motivating_scene_ids],
            "arm": self.arm,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "CompositionProposal":
        return cls(
            composition_id=str(d["composition_id"]),
            constituents=list(d["constituents"]),
            marginal_frequencies=dict(d["marginal_frequencies"]),
            pairwise_frequencies=dict(d["pairwise_frequencies"]),
            expected_joint=float(d["expected_joint"]),
            observed_joint=float(d["observed_joint"]),
            novelty_score=float(d["novelty_score"]),
            motivating_scene_ids=[WindowKey.from_json(w) for w in d.get("motivating_scene_ids", [])],
            arm=str(d.get("arm", "reasoning")),
        )


@dataclass
class ScoredProposal:
    """A CompositionProposal after Module 4: Scorer has run."""
    composition_id: str
    constituents: list[str]
    marginal_frequencies: dict[str, float]
    pairwise_frequencies: dict[str, float]
    expected_joint: float
    observed_joint: float
    novelty_score: float
    motivating_scene_ids: list[WindowKey]
    arm: str
    plausibility_score: float
    plausibility_justification: str  # empty string when rejection_reason == "plausibility_check_failed"
    frontier_difficulty_score: float | None
    frontier_difficulty_signals: dict[str, Any]
    final_rank_score: float
    accepted: bool
    rejection_reason: str | None

    def to_json(self) -> dict[str, Any]:
        return {
            "composition_id": self.composition_id,
            "constituents": self.constituents,
            "marginal_frequencies": self.marginal_frequencies,
            "pairwise_frequencies": self.pairwise_frequencies,
            "expected_joint": self.expected_joint,
            "observed_joint": self.observed_joint,
            "novelty_score": self.novelty_score,
            "motivating_scene_ids": [w.to_json() for w in self.motivating_scene_ids],
            "arm": self.arm,
            "plausibility_score": self.plausibility_score,
            "plausibility_justification": self.plausibility_justification,
            "frontier_difficulty_score": self.frontier_difficulty_score,
            "frontier_difficulty_signals": self.frontier_difficulty_signals,
            "final_rank_score": self.final_rank_score,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "ScoredProposal":
        return cls(
            composition_id=str(d["composition_id"]),
            constituents=list(d["constituents"]),
            marginal_frequencies=dict(d["marginal_frequencies"]),
            pairwise_frequencies=dict(d["pairwise_frequencies"]),
            expected_joint=float(d["expected_joint"]),
            observed_joint=float(d["observed_joint"]),
            novelty_score=float(d["novelty_score"]),
            motivating_scene_ids=[WindowKey.from_json(w) for w in d.get("motivating_scene_ids", [])],
            arm=str(d.get("arm", "reasoning")),
            plausibility_score=float(d["plausibility_score"]),
            plausibility_justification=str(d["plausibility_justification"]),
            frontier_difficulty_score=d.get("frontier_difficulty_score"),
            frontier_difficulty_signals=dict(d.get("frontier_difficulty_signals", {})),
            final_rank_score=float(d["final_rank_score"]),
            accepted=bool(d["accepted"]),
            rejection_reason=d.get("rejection_reason"),
        )
