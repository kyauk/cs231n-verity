"""Module 6: Evaluation — Evaluator class and EvaluationInput.

evaluate() is a pure function: same inputs → same outputs, no side effects.
save() handles all disk I/O and is kept strictly separate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np

from pipeline.interfaces.proposal import ScoredProposal
from pipeline.interfaces.rating import Rating
from pipeline.interfaces.report import DifferentialExample, EvaluationReport
from pipeline.interfaces.schema_record import SchemaRecord
from pipeline.interfaces.window import WindowKey
from pipeline.modules.evaluation.metrics import (
    compute_differential_examples,
    compute_rating_stats,
    compute_seeded_recall,
    krippendorff_alpha,
)

# Pre-registered primary K — matches the Judge UI session size.
# Lock this before running any pipeline output through evaluation.
RECALL_K_PRIMARY = 30


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class MissingSubsetLabelsError(ValueError):
    """Raised when seeded_subset_labels are absent or incomplete."""


class ArmMismatchError(ValueError):
    """Raised when a Rating references an arm not in proposals_by_arm."""


# ---------------------------------------------------------------------------
# EvaluationInput
# ---------------------------------------------------------------------------

@dataclass
class EvaluationInput:
    """All data Module 6 needs to produce an EvaluationReport.

    proposals_by_arm
        Keyed by arm name (e.g., "reasoning", "visual"). Each list should be
        the full ranked output of the Scorer for that arm (accepted and
        rejected proposals both included — filtering is done internally).

    ratings
        All Rating objects from all raters. Rating.arm must match one of the
        keys in proposals_by_arm.

    seeded_window_ids
        Pre-registered seeded evaluation set. Must be non-empty.

    seeded_subset_labels
        Maps every seeded WindowKey to "familiar" or "unfamiliar".
        Must be pre-registered before any pipeline output is generated.
        Missing keys cause MissingSubsetLabelsError at evaluate() time.

    schema_records
        Optional: encoder SchemaRecord list for computing failure_mode_distribution.
        If None, failure_mode_distribution is reported as empty.

    recall_k
        The primary recall threshold. Default is RECALL_K_PRIMARY (30).
        Set this before running — do not tune it after seeing results.
    """
    proposals_by_arm: dict[str, list[ScoredProposal]]
    ratings: list[Rating]
    seeded_window_ids: list[WindowKey]
    seeded_subset_labels: dict[WindowKey, Literal["familiar", "unfamiliar"]]
    schema_records: list[SchemaRecord] | None = None
    recall_k: int = RECALL_K_PRIMARY


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """Computes Phase 1 evaluation metrics and renders reports.

    Usage (pure evaluation, no I/O):
        evaluator = Evaluator()
        report = evaluator.evaluate(input)

    Usage (with persistence):
        report = evaluator.evaluate(input)
        path = evaluator.save(report, Path("reports/"))
    """

    def evaluate(self, input: EvaluationInput) -> EvaluationReport:
        """Compute all Phase 1 metrics. Pure function — no side effects.

        Raises
        ------
        MissingSubsetLabelsError
            If seeded_subset_labels is empty or missing keys from seeded_window_ids.
        ArmMismatchError
            If any Rating references an arm not in proposals_by_arm.
        """
        self._validate(input)

        # --- Seeded recall (per arm) ---
        recall_by_arm: dict[str, dict[str, dict[str, float]]] = {}
        n_proposals_per_arm: dict[str, int] = {}
        n_proposals_filtered: dict[str, int] = {}

        for arm, proposals in input.proposals_by_arm.items():
            accepted = [p for p in proposals if p.accepted]
            recall_by_arm[arm] = compute_seeded_recall(
                proposals=accepted,
                seeded_window_ids=input.seeded_window_ids,
                seeded_subset_labels=input.seeded_subset_labels,
                k=input.recall_k,
            )
            n_proposals_per_arm[arm] = len(accepted)
            n_proposals_filtered[arm] = len(proposals) - len(accepted)

        # --- Rating statistics (per arm) ---
        rating_stats = compute_rating_stats(input.ratings)

        mean_coherence: dict[str, float] = {}
        mean_usefulness: dict[str, float] = {}
        coherence_ci_95: dict[str, tuple[float, float] | None] = {}
        usefulness_ci_95: dict[str, tuple[float, float] | None] = {}
        n_ratings_per_arm: dict[str, int] = {}

        for arm in input.proposals_by_arm:
            stats = rating_stats.get(arm, {})
            mean_coherence[arm] = stats.get("mean_coherence", float("nan"))
            mean_usefulness[arm] = stats.get("mean_usefulness", float("nan"))
            coherence_ci_95[arm] = stats.get("coherence_ci_95")
            usefulness_ci_95[arm] = stats.get("usefulness_ci_95")
            n_ratings_per_arm[arm] = stats.get("n", 0)

        # --- Inter-rater agreement ---
        ira_coherence, ira_usefulness, n_raters_overlapping = self._compute_ira(input.ratings)

        # --- Differential examples ---
        diff_examples: list[DifferentialExample] = compute_differential_examples(
            input.proposals_by_arm, input.ratings
        )

        # --- Failure mode distribution ---
        failure_dist: dict[str, Any] = {}
        if input.schema_records:
            for rec in input.schema_records:
                if rec.failure_mode:
                    failure_dist[rec.failure_mode] = failure_dist.get(rec.failure_mode, 0) + 1

        # --- Seeded set size ---
        familiar_count = sum(
            1 for lbl in input.seeded_subset_labels.values() if lbl == "familiar"
        )
        unfamiliar_count = sum(
            1 for lbl in input.seeded_subset_labels.values() if lbl == "unfamiliar"
        )

        rater_ids = {r.rater_id for r in input.ratings}

        return EvaluationReport(
            seeded_recall=recall_by_arm,
            recall_k_primary=input.recall_k,
            mean_coherence=mean_coherence,
            mean_usefulness=mean_usefulness,
            coherence_ci_95=coherence_ci_95,
            usefulness_ci_95=usefulness_ci_95,
            n_ratings_per_arm=n_ratings_per_arm,
            inter_rater_agreement_coherence=ira_coherence,
            inter_rater_agreement_usefulness=ira_usefulness,
            n_raters_overlapping=n_raters_overlapping,
            differential_examples=diff_examples,
            failure_mode_distribution=failure_dist,
            n_proposals_per_arm=n_proposals_per_arm,
            n_proposals_filtered=n_proposals_filtered,
            n_raters=len(rater_ids),
            seeded_set_size={"familiar": familiar_count, "unfamiliar": unfamiliar_count},
        )

    def save(self, report: EvaluationReport, output_dir: Path) -> Path:
        """Write report.json and report.md to a timestamped subdirectory.

        Parameters
        ----------
        report
            The EvaluationReport returned by evaluate().
        output_dir
            Root directory under which a timestamped subfolder is created.

        Returns
        -------
        Path to the subdirectory that was written.
        """
        from pipeline.modules.evaluation.renderer import render_markdown

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = Path(output_dir) / ts
        run_dir.mkdir(parents=True, exist_ok=True)

        # JSON
        json_path = run_dir / "report.json"
        json_path.write_text(
            json.dumps(report.to_json(), indent=2, default=_json_default),
            encoding="utf-8",
        )

        # Markdown
        md_path = run_dir / "report.md"
        md_path.write_text(render_markdown(report), encoding="utf-8")

        return run_dir

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _validate(self, input: EvaluationInput) -> None:
        if not input.seeded_window_ids:
            raise MissingSubsetLabelsError(
                "seeded_window_ids is empty — cannot compute seeded recall. "
                "Pre-register the seeded set before running evaluation."
            )
        if not input.seeded_subset_labels:
            raise MissingSubsetLabelsError(
                "seeded_subset_labels is empty — subset split was not pre-registered."
            )
        missing = [w for w in input.seeded_window_ids if w not in input.seeded_subset_labels]
        if missing:
            raise MissingSubsetLabelsError(
                f"seeded_subset_labels missing {len(missing)} seeded windows: "
                f"{missing[:3]}{'...' if len(missing) > 3 else ''}"
            )
        known_arms = set(input.proposals_by_arm)
        for r in input.ratings:
            if r.arm not in known_arms:
                raise ArmMismatchError(
                    f"Rating from rater '{r.rater_id}' references arm '{r.arm}' "
                    f"which is not in proposals_by_arm (known: {sorted(known_arms)})."
                )

    def _compute_ira(
        self,
        ratings: list[Rating],
    ) -> tuple[float | None, float | None, int]:
        """Build rater × proposal matrices and compute Krippendorff's alpha.

        Returns (alpha_coherence, alpha_usefulness, n_raters_overlapping).
        Both alpha values are None when fewer than 2 raters overlap.
        """
        if not ratings:
            return None, None, 0

        rater_ids = sorted({r.rater_id for r in ratings})
        proposal_ids = sorted({r.proposal_id for r in ratings})

        if len(rater_ids) < 2:
            return None, None, 0

        rater_idx = {r: i for i, r in enumerate(rater_ids)}
        prop_idx = {p: i for i, p in enumerate(proposal_ids)}

        coh_matrix = np.full((len(proposal_ids), len(rater_ids)), np.nan)
        use_matrix = np.full((len(proposal_ids), len(rater_ids)), np.nan)

        for r in ratings:
            pi = prop_idx[r.proposal_id]
            ri = rater_idx[r.rater_id]
            coh_matrix[pi, ri] = float(r.coherence_score)
            use_matrix[pi, ri] = float(r.usefulness_score)

        # Count items where ≥2 raters provided ratings
        overlap_mask = (~np.isnan(coh_matrix)).sum(axis=1) >= 2
        n_overlapping_raters = int(overlap_mask.any())
        # Count distinct raters who have at least one non-nan entry
        active = int((~np.isnan(coh_matrix)).any(axis=0).sum())

        if active < 2:
            return None, None, 0

        alpha_coh = krippendorff_alpha(coh_matrix)
        alpha_use = krippendorff_alpha(use_matrix)

        # n_raters_overlapping = distinct raters who rated at least one proposal
        # that another rater also rated. Not the same as total rater count.
        n_overlap = int((~np.isnan(coh_matrix)).any(axis=0).sum())
        return alpha_coh, alpha_use, n_overlap


def _json_default(obj: Any) -> Any:
    """JSON serializer for numpy scalars that escape default encoder."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
