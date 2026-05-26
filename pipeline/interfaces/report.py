"""Shared EvaluationReport and DifferentialExample types — produced by Module 6: Evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# DifferentialExample — one composition where arms diverged meaningfully
# ---------------------------------------------------------------------------

@dataclass
class DifferentialExample:
    """A composition where two encoder arms assigned notably different scores or ranks.

    Used for qualitative differential analysis in the evaluation report.
    """
    proposal_id: str                           # = composition_id
    constituents: list[str]
    arm_scores: dict[str, float]               # arm_name -> final_rank_score
    arm_ranks: dict[str, int]                  # arm_name -> rank within that arm's output
    coherence_ratings: dict[str, float]        # arm_name -> mean coherence (from raters)
    usefulness_ratings: dict[str, float]       # arm_name -> mean usefulness (from raters)

    def to_json(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "constituents": self.constituents,
            "arm_scores": self.arm_scores,
            "arm_ranks": self.arm_ranks,
            "coherence_ratings": self.coherence_ratings,
            "usefulness_ratings": self.usefulness_ratings,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "DifferentialExample":
        return cls(
            proposal_id=str(d["proposal_id"]),
            constituents=list(d["constituents"]),
            arm_scores={k: float(v) for k, v in d["arm_scores"].items()},
            arm_ranks={k: int(v) for k, v in d["arm_ranks"].items()},
            coherence_ratings={k: float(v) for k, v in d.get("coherence_ratings", {}).items()},
            usefulness_ratings={k: float(v) for k, v in d.get("usefulness_ratings", {}).items()},
        )


# ---------------------------------------------------------------------------
# EvaluationReport — full Phase 1 evaluation output
# ---------------------------------------------------------------------------

@dataclass
class EvaluationReport:
    """Full Phase 1 evaluation output."""

    # Seeded recall — nested: {arm: {subset: {k_str: recall}}}
    # subset ∈ {"overall", "familiar", "unfamiliar"}
    # k_str ∈ {"@10", "@30", "@all"}
    seeded_recall: dict[str, dict[str, dict[str, float]]]
    recall_k_primary: int                      # = 30, pre-registered before results seen

    # Per-arm rating statistics
    mean_coherence: dict[str, float]           # arm -> mean
    mean_usefulness: dict[str, float]
    coherence_ci_95: dict[str, tuple[float, float] | None]   # None if n_ratings < 30
    usefulness_ci_95: dict[str, tuple[float, float] | None]
    n_ratings_per_arm: dict[str, int]          # arm -> count of ratings used

    # Cross-rater agreement (None when < 2 overlapping raters)
    inter_rater_agreement_coherence: float | None
    inter_rater_agreement_usefulness: float | None
    n_raters_overlapping: int                  # raters with ≥1 rating on the same proposal

    # Qualitative
    differential_examples: list[DifferentialExample]
    failure_mode_distribution: dict[str, Any]

    # Methodology
    n_proposals_per_arm: dict[str, int]
    n_proposals_filtered: dict[str, int]
    n_raters: int
    seeded_set_size: dict[str, int]            # {"familiar": N, "unfamiliar": N}

    def to_json(self) -> dict[str, Any]:
        import math

        def _safe(v: float) -> "float | None":
            # Python's json.dumps rejects float('nan')/float('inf') — convert to None.
            return None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v

        return {
            "seeded_recall": self.seeded_recall,
            "recall_k_primary": self.recall_k_primary,
            "mean_coherence": {k: _safe(v) for k, v in self.mean_coherence.items()},
            "mean_usefulness": {k: _safe(v) for k, v in self.mean_usefulness.items()},
            "coherence_ci_95": {
                k: list(v) if v is not None else None
                for k, v in self.coherence_ci_95.items()
            },
            "usefulness_ci_95": {
                k: list(v) if v is not None else None
                for k, v in self.usefulness_ci_95.items()
            },
            "n_ratings_per_arm": self.n_ratings_per_arm,
            "inter_rater_agreement_coherence": self.inter_rater_agreement_coherence,
            "inter_rater_agreement_usefulness": self.inter_rater_agreement_usefulness,
            "n_raters_overlapping": self.n_raters_overlapping,
            "differential_examples": [ex.to_json() for ex in self.differential_examples],
            "failure_mode_distribution": self.failure_mode_distribution,
            "n_proposals_per_arm": self.n_proposals_per_arm,
            "n_proposals_filtered": self.n_proposals_filtered,
            "n_raters": self.n_raters,
            "seeded_set_size": self.seeded_set_size,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "EvaluationReport":
        def _ci(v: list | None) -> tuple[float, float] | None:
            return tuple(v) if v is not None else None  # type: ignore[return-value]

        return cls(
            seeded_recall=d["seeded_recall"],
            recall_k_primary=int(d["recall_k_primary"]),
            mean_coherence=dict(d["mean_coherence"]),
            mean_usefulness=dict(d["mean_usefulness"]),
            coherence_ci_95={k: _ci(v) for k, v in d["coherence_ci_95"].items()},
            usefulness_ci_95={k: _ci(v) for k, v in d["usefulness_ci_95"].items()},
            n_ratings_per_arm=dict(d["n_ratings_per_arm"]),
            inter_rater_agreement_coherence=d.get("inter_rater_agreement_coherence"),
            inter_rater_agreement_usefulness=d.get("inter_rater_agreement_usefulness"),
            n_raters_overlapping=int(d.get("n_raters_overlapping", 0)),
            differential_examples=[
                DifferentialExample.from_json(ex) for ex in d.get("differential_examples", [])
            ],
            failure_mode_distribution=dict(d.get("failure_mode_distribution", {})),
            n_proposals_per_arm=dict(d["n_proposals_per_arm"]),
            n_proposals_filtered=dict(d["n_proposals_filtered"]),
            n_raters=int(d["n_raters"]),
            seeded_set_size=dict(d["seeded_set_size"]),
        )
