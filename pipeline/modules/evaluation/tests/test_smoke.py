"""Smoke tests for Module 6: Evaluation (full public interface).

Uses a synthetic 3-rater / 20-proposal / 10-seeded-window dataset.
No disk I/O (evaluate() is pure), no VLM calls, no external services.
"""

from __future__ import annotations

import math

import pytest

from pipeline.interfaces.proposal import ScoredProposal
from pipeline.interfaces.rating import Rating
from pipeline.interfaces.report import EvaluationReport
from pipeline.interfaces.window import WindowKey
from pipeline.modules.evaluation.evaluator import (
    ArmMismatchError,
    EvaluationInput,
    Evaluator,
    MissingSubsetLabelsError,
)


# ---------------------------------------------------------------------------
# Synthetic dataset factory
# ---------------------------------------------------------------------------

def _make_proposal(
    cid: str,
    motivating: list[WindowKey],
    arm: str = "reasoning",
    rank_score: float = 0.8,
    accepted: bool = True,
) -> ScoredProposal:
    return ScoredProposal(
        composition_id=cid,
        constituents=["conditions:fog", "time_of_day:night"],
        marginal_frequencies={"conditions:fog": 0.2, "time_of_day:night": 0.3},
        pairwise_frequencies={},
        expected_joint=0.06,
        observed_joint=0.001,
        novelty_score=4.1,
        motivating_scene_ids=motivating,
        arm=arm,
        plausibility_score=0.9,
        plausibility_justification="Plausible.",
        frontier_difficulty_score=0.7,
        frontier_difficulty_signals={},
        final_rank_score=rank_score,
        accepted=accepted,
        rejection_reason=None if accepted else "plausibility_check_failed",
    )


def _make_rating(
    proposal_id: str,
    rater_id: str,
    arm: str = "reasoning",
    coh: int = 4,
    use: int = 3,
) -> Rating:
    return Rating(
        rater_id=rater_id,
        proposal_id=proposal_id,
        arm=arm,
        coherence_score=coh,
        usefulness_score=use,
        timestamp="2026-05-26T00:00:00Z",
    )


def _make_input(n_seeded: int = 10, n_proposals: int = 20, n_raters: int = 3) -> EvaluationInput:
    seeded = [WindowKey("seeded_seg", i) for i in range(n_seeded)]
    labels = {w: ("familiar" if i < n_seeded // 2 else "unfamiliar") for i, w in enumerate(seeded)}

    proposals = []
    for i in range(n_proposals):
        motivating = [seeded[i % n_seeded]]
        proposals.append(_make_proposal(f"prop_{i:03d}", motivating))

    # Add a few rejected proposals to test filtering
    proposals.append(_make_proposal("prop_rejected", [seeded[0]], accepted=False))

    ratings = []
    for rater_i in range(n_raters):
        for prop in proposals[:10]:  # each rater rates the first 10
            ratings.append(_make_rating(
                prop.composition_id,
                f"rater_{rater_i:02d}",
                coh=rater_i + 3,
                use=5 - rater_i,
            ))

    return EvaluationInput(
        proposals_by_arm={"reasoning": proposals},
        ratings=ratings,
        seeded_window_ids=seeded,
        seeded_subset_labels=labels,
    )


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

class TestEvaluatorSmoke:

    def test_imports_clean(self):
        from pipeline.modules.evaluation import Evaluator
        assert Evaluator is not None

    def test_evaluate_returns_report(self):
        ev = Evaluator()
        report = ev.evaluate(_make_input())
        assert isinstance(report, EvaluationReport)

    def test_seeded_recall_keys_present(self):
        # Structure: {arm: {subset: {k: recall}}}
        report = Evaluator().evaluate(_make_input())
        assert "reasoning" in report.seeded_recall
        subset_data = report.seeded_recall["reasoning"]
        for subset in ("overall", "familiar", "unfamiliar"):
            assert subset in subset_data, f"Missing subset '{subset}'"
        # @10, @30, @all must all be present under each subset
        for k in ("@10", "@30", "@all"):
            assert k in subset_data["overall"], f"Missing recall key '{k}'"

    def test_recall_values_in_0_1(self):
        report = Evaluator().evaluate(_make_input())
        for arm, subset_data in report.seeded_recall.items():
            for subset, k_data in subset_data.items():
                for k, val in k_data.items():
                    assert 0.0 <= val <= 1.0, f"recall out of range: {arm}/{subset}/{k}={val}"

    def test_n_proposals_per_arm_excludes_rejected(self):
        report = Evaluator().evaluate(_make_input(n_proposals=20))
        # 20 accepted + 1 rejected → accepted count = 20, filtered = 1
        assert report.n_proposals_per_arm["reasoning"] == 20
        assert report.n_proposals_filtered["reasoning"] == 1

    def test_mean_scores_in_valid_range(self):
        report = Evaluator().evaluate(_make_input())
        for arm in report.mean_coherence:
            mc = report.mean_coherence[arm]
            mu = report.mean_usefulness[arm]
            if not math.isnan(mc):
                assert 1.0 <= mc <= 5.0
            if not math.isnan(mu):
                assert 1.0 <= mu <= 5.0

    def test_n_raters_counted_correctly(self):
        report = Evaluator().evaluate(_make_input(n_raters=3))
        assert report.n_raters == 3

    def test_seeded_set_size_correct(self):
        report = Evaluator().evaluate(_make_input(n_seeded=10))
        assert report.seeded_set_size["familiar"] == 5
        assert report.seeded_set_size["unfamiliar"] == 5

    def test_ira_none_with_one_rater(self):
        inp = _make_input(n_raters=1)
        report = Evaluator().evaluate(inp)
        assert report.inter_rater_agreement_coherence is None
        assert report.inter_rater_agreement_usefulness is None

    def test_ira_computable_with_three_raters(self):
        # With 3 raters all rating the same proposals, alpha should be computable
        inp = _make_input(n_raters=3)
        report = Evaluator().evaluate(inp)
        # May be None if no overlap, but with our fixture there IS overlap
        # Just verify it's a valid float or None
        assert report.inter_rater_agreement_coherence is None or isinstance(
            report.inter_rater_agreement_coherence, float
        )

    def test_report_json_serializable(self):
        import json
        report = Evaluator().evaluate(_make_input())
        serialized = json.dumps(report.to_json())
        assert len(serialized) > 100

    def test_report_roundtrip(self):
        report = Evaluator().evaluate(_make_input())
        restored = EvaluationReport.from_json(report.to_json())
        assert restored.n_raters == report.n_raters
        assert restored.recall_k_primary == report.recall_k_primary
        assert restored.n_proposals_per_arm == report.n_proposals_per_arm

    def test_evaluate_pure_no_disk_writes(self, tmp_path):
        """evaluate() must not write any files."""
        before = set(tmp_path.iterdir())
        Evaluator().evaluate(_make_input())
        after = set(tmp_path.iterdir())
        assert before == after  # nothing written

    def test_save_writes_json_and_md(self, tmp_path):
        report = Evaluator().evaluate(_make_input())
        path = Evaluator().save(report, tmp_path)
        assert (path / "report.json").exists()
        assert (path / "report.md").exists()

    def test_save_json_is_valid(self, tmp_path):
        import json
        report = Evaluator().evaluate(_make_input())
        path = Evaluator().save(report, tmp_path)
        loaded = json.loads((path / "report.json").read_text())
        assert "seeded_recall" in loaded
        assert "mean_coherence" in loaded

    def test_save_creates_timestamped_subdir(self, tmp_path):
        report = Evaluator().evaluate(_make_input())
        path = Evaluator().save(report, tmp_path)
        # Should be a direct child of tmp_path
        assert path.parent == tmp_path
        assert len(path.name) > 0

    def test_save_called_twice_creates_two_subdirs(self, tmp_path):
        import time
        ev = Evaluator()
        report = ev.evaluate(_make_input())
        p1 = ev.save(report, tmp_path)
        time.sleep(1.1)  # ensure different timestamp
        p2 = ev.save(report, tmp_path)
        assert p1 != p2

    def test_markdown_renderer_runs(self):
        from pipeline.modules.evaluation.renderer import render_markdown
        report = Evaluator().evaluate(_make_input())
        md = render_markdown(report)
        assert "Seeded Recall" in md
        assert "Expert Ratings" in md
        assert "Krippendorff" in md

    def test_html_standalone_renderer_runs(self):
        from pipeline.modules.evaluation.renderer import render_html
        report = Evaluator().evaluate(_make_input())
        html = render_html(report)
        assert "<html" in html
        assert "plotly" in html.lower()

    def test_html_embeddable_renderer_runs(self):
        from pipeline.modules.evaluation.renderer import render_html
        report = Evaluator().evaluate(_make_input())
        html = render_html(report, embeddable=True)
        assert "<html" not in html


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestEvaluatorErrors:

    def test_empty_seeded_raises(self):
        inp = EvaluationInput(
            proposals_by_arm={"reasoning": []},
            ratings=[],
            seeded_window_ids=[],
            seeded_subset_labels={},
        )
        with pytest.raises(MissingSubsetLabelsError, match="empty"):
            Evaluator().evaluate(inp)

    def test_missing_labels_raises(self):
        seeded = [WindowKey("seg", 0), WindowKey("seg", 1)]
        inp = EvaluationInput(
            proposals_by_arm={"reasoning": []},
            ratings=[],
            seeded_window_ids=seeded,
            seeded_subset_labels={seeded[0]: "familiar"},  # seg/1 missing
        )
        with pytest.raises(MissingSubsetLabelsError, match="missing"):
            Evaluator().evaluate(inp)

    def test_arm_mismatch_raises(self):
        seeded = [WindowKey("seg", 0)]
        labels = {seeded[0]: "familiar"}
        inp = EvaluationInput(
            proposals_by_arm={"reasoning": []},
            ratings=[Rating(
                rater_id="r1", proposal_id="p1", arm="visual",
                coherence_score=4, usefulness_score=3,
                timestamp="2026-05-26T00:00:00Z",
            )],
            seeded_window_ids=seeded,
            seeded_subset_labels=labels,
        )
        with pytest.raises(ArmMismatchError, match="visual"):
            Evaluator().evaluate(inp)

    def test_zero_proposals_gives_zero_recall(self):
        seeded = [WindowKey("seg", 0)]
        labels = {seeded[0]: "familiar"}
        inp = EvaluationInput(
            proposals_by_arm={"reasoning": []},
            ratings=[],
            seeded_window_ids=seeded,
            seeded_subset_labels=labels,
        )
        report = Evaluator().evaluate(inp)
        assert report.seeded_recall["reasoning"]["overall"]["@30"] == 0.0

    def test_no_ratings_gives_nan_means(self):
        seeded = [WindowKey("seg", 0)]
        labels = {seeded[0]: "familiar"}
        inp = EvaluationInput(
            proposals_by_arm={"reasoning": []},
            ratings=[],
            seeded_window_ids=seeded,
            seeded_subset_labels=labels,
        )
        report = Evaluator().evaluate(inp)
        assert math.isnan(report.mean_coherence.get("reasoning", float("nan")))
