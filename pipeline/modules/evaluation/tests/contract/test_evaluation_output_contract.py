"""Contract tests for Module 6: Evaluation output.

Validates that Evaluator.evaluate() returns an EvaluationReport that:
1. Conforms to the interface type's field schema
2. Survives a JSON round-trip without data loss
3. Contains all required keys for every arm in proposals_by_arm
4. Reports N alongside every CI (contract: n_ratings_per_arm must be present)
"""

from __future__ import annotations

import json
import math

import pytest

from pipeline.interfaces.proposal import ScoredProposal
from pipeline.interfaces.rating import Rating
from pipeline.interfaces.report import DifferentialExample, EvaluationReport
from pipeline.interfaces.window import WindowKey
from pipeline.modules.evaluation.evaluator import EvaluationInput, Evaluator


# ---------------------------------------------------------------------------
# Minimal fixture
# ---------------------------------------------------------------------------

def _make_minimal_input() -> EvaluationInput:
    seeded = [WindowKey("seg", i) for i in range(6)]
    labels = {w: ("familiar" if i < 3 else "unfamiliar") for i, w in enumerate(seeded)}
    proposals = [
        ScoredProposal(
            composition_id=f"cid_{i:02d}",
            constituents=["conditions:fog"],
            marginal_frequencies={"conditions:fog": 0.2},
            pairwise_frequencies={},
            expected_joint=0.04,
            observed_joint=0.001,
            novelty_score=3.7,
            motivating_scene_ids=[seeded[i % len(seeded)]],
            arm="reasoning",
            plausibility_score=0.85,
            plausibility_justification="Ok.",
            frontier_difficulty_score=0.6,
            frontier_difficulty_signals={},
            final_rank_score=0.75,
            accepted=True,
            rejection_reason=None,
        )
        for i in range(8)
    ]
    ratings = [
        Rating(
            rater_id=f"r{ri}", proposal_id=f"cid_{pi:02d}", arm="reasoning",
            coherence_score=(ri % 3) + 3, usefulness_score=(pi % 3) + 2,
            timestamp="2026-05-26T00:00:00Z",
        )
        for ri in range(3) for pi in range(8)
    ]
    return EvaluationInput(
        proposals_by_arm={"reasoning": proposals},
        ratings=ratings,
        seeded_window_ids=seeded,
        seeded_subset_labels=labels,
    )


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------

class TestEvaluationOutputContract:

    @pytest.fixture
    def report(self) -> EvaluationReport:
        return Evaluator().evaluate(_make_minimal_input())

    def test_report_is_evaluation_report_instance(self, report):
        assert isinstance(report, EvaluationReport)

    # --- seeded_recall ---

    def test_seeded_recall_has_all_arms(self, report):
        assert "reasoning" in report.seeded_recall

    def test_seeded_recall_has_all_subsets(self, report):
        # Structure: {arm: {subset: {k: recall}}}
        for arm, subset_data in report.seeded_recall.items():
            for subset in ("overall", "familiar", "unfamiliar"):
                assert subset in subset_data, f"Missing subset '{subset}' for arm '{arm}'"

    def test_seeded_recall_has_at_10_at_30_at_all(self, report):
        for arm, subset_data in report.seeded_recall.items():
            for subset, k_data in subset_data.items():
                for k in ("@10", "@30", "@all"):
                    assert k in k_data, f"Missing '{k}' in seeded_recall[{arm}][{subset}]"

    def test_recall_k_primary_is_30(self, report):
        assert report.recall_k_primary == 30

    def test_all_recall_values_in_unit_interval(self, report):
        for arm, subsets in report.seeded_recall.items():
            for subset, ks in subsets.items():
                for k, val in ks.items():
                    assert 0.0 <= val <= 1.0

    # --- rating stats ---

    def test_mean_coherence_has_all_arms(self, report):
        assert "reasoning" in report.mean_coherence

    def test_mean_usefulness_has_all_arms(self, report):
        assert "reasoning" in report.mean_usefulness

    def test_coherence_ci_has_all_arms(self, report):
        assert "reasoning" in report.coherence_ci_95

    def test_usefulness_ci_has_all_arms(self, report):
        assert "reasoning" in report.usefulness_ci_95

    def test_n_ratings_per_arm_present(self, report):
        assert "reasoning" in report.n_ratings_per_arm
        assert report.n_ratings_per_arm["reasoning"] > 0

    def test_ci_type_is_tuple_or_none(self, report):
        for arm, ci in report.coherence_ci_95.items():
            assert ci is None or (isinstance(ci, tuple) and len(ci) == 2)
        for arm, ci in report.usefulness_ci_95.items():
            assert ci is None or (isinstance(ci, tuple) and len(ci) == 2)

    # --- inter-rater agreement ---

    def test_ira_type_is_float_or_none(self, report):
        assert report.inter_rater_agreement_coherence is None or isinstance(
            report.inter_rater_agreement_coherence, float
        )
        assert report.inter_rater_agreement_usefulness is None or isinstance(
            report.inter_rater_agreement_usefulness, float
        )

    def test_n_raters_overlapping_is_non_negative_int(self, report):
        assert isinstance(report.n_raters_overlapping, int)
        assert report.n_raters_overlapping >= 0

    # --- qualitative ---

    def test_differential_examples_is_list(self, report):
        assert isinstance(report.differential_examples, list)

    def test_differential_examples_are_correct_type(self, report):
        for ex in report.differential_examples:
            assert isinstance(ex, DifferentialExample)

    def test_failure_mode_distribution_is_dict(self, report):
        assert isinstance(report.failure_mode_distribution, dict)

    # --- methodology ---

    def test_n_proposals_per_arm_present(self, report):
        assert "reasoning" in report.n_proposals_per_arm

    def test_n_proposals_filtered_present(self, report):
        assert "reasoning" in report.n_proposals_filtered

    def test_n_raters_positive(self, report):
        assert report.n_raters >= 1

    def test_seeded_set_size_has_both_subsets(self, report):
        assert "familiar" in report.seeded_set_size
        assert "unfamiliar" in report.seeded_set_size
        assert report.seeded_set_size["familiar"] + report.seeded_set_size["unfamiliar"] >= 1

    # --- JSON round-trip ---

    def test_json_round_trip_preserves_all_fields(self, report):
        restored = EvaluationReport.from_json(report.to_json())
        assert restored.n_raters == report.n_raters
        assert restored.recall_k_primary == report.recall_k_primary
        assert restored.n_proposals_per_arm == report.n_proposals_per_arm
        assert restored.n_proposals_filtered == report.n_proposals_filtered
        assert restored.seeded_recall == report.seeded_recall
        assert restored.mean_coherence == report.mean_coherence
        assert restored.n_ratings_per_arm == report.n_ratings_per_arm
        assert len(restored.differential_examples) == len(report.differential_examples)

    def test_json_is_valid_string(self, report):
        serialized = json.dumps(report.to_json())
        assert len(serialized) > 50

    def test_json_has_required_top_level_keys(self, report):
        d = report.to_json()
        required = [
            "seeded_recall", "recall_k_primary",
            "mean_coherence", "mean_usefulness",
            "coherence_ci_95", "usefulness_ci_95", "n_ratings_per_arm",
            "inter_rater_agreement_coherence", "inter_rater_agreement_usefulness",
            "n_raters_overlapping", "differential_examples", "failure_mode_distribution",
            "n_proposals_per_arm", "n_proposals_filtered", "n_raters", "seeded_set_size",
        ]
        for key in required:
            assert key in d, f"Missing key '{key}' in EvaluationReport.to_json()"
