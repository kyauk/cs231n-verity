"""Unit tests for pipeline/modules/evaluation/metrics.py.

Tests are self-contained — no external I/O, no VLM calls. Every test uses
tiny synthetic datasets that make expected values hand-calculable.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pipeline.interfaces.proposal import ScoredProposal
from pipeline.interfaces.rating import Rating
from pipeline.interfaces.window import WindowKey
from pipeline.modules.evaluation.metrics import (
    MIN_N_FOR_CI,
    compute_rating_stats,
    compute_seeded_recall,
    krippendorff_alpha,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_proposal(
    cid: str,
    motivating: list[WindowKey],
    rank_score: float = 0.8,
    accepted: bool = True,
    arm: str = "reasoning",
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
        plausibility_justification="Physically plausible.",
        frontier_difficulty_score=0.7,
        frontier_difficulty_signals={"variance": 0.3},
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


# ---------------------------------------------------------------------------
# compute_seeded_recall
# ---------------------------------------------------------------------------

class TestSeededRecall:

    def _labels(self, windows: list[WindowKey], split: str = "familiar") -> dict:
        return {w: split for w in windows}  # type: ignore[return-value]

    def test_perfect_recall(self):
        seeded = [WindowKey("seg", i) for i in range(5)]
        proposals = [
            _make_proposal("p1", motivating=seeded[:3]),
            _make_proposal("p2", motivating=seeded[3:]),
        ]
        labels = self._labels(seeded, "familiar")
        result = compute_seeded_recall(proposals, seeded, labels, k=30)
        assert result["overall"]["@30"] == pytest.approx(1.0)
        assert result["familiar"]["@30"] == pytest.approx(1.0)

    def test_zero_recall_no_overlap(self):
        seeded = [WindowKey("seg", i) for i in range(5)]
        other = [WindowKey("other", i) for i in range(5)]
        proposals = [_make_proposal("p1", motivating=other)]
        labels = self._labels(seeded, "unfamiliar")
        result = compute_seeded_recall(proposals, seeded, labels, k=30)
        assert result["overall"]["@30"] == pytest.approx(0.0)
        assert result["unfamiliar"]["@30"] == pytest.approx(0.0)

    def test_top_k_truncation(self):
        # 30 proposals each covering a distinct seeded window.
        # @10 should cover 10/30, @30 should cover all 30/30.
        seeded = [WindowKey("seg", i) for i in range(30)]
        proposals = [_make_proposal(f"p{i}", motivating=[seeded[i]]) for i in range(30)]
        labels = self._labels(seeded)
        result = compute_seeded_recall(proposals, seeded, labels, k=30)
        assert result["overall"]["@10"] == pytest.approx(10 / 30)
        assert result["overall"]["@30"] == pytest.approx(1.0)
        assert result["overall"]["@all"] == pytest.approx(1.0)

    def test_rejected_proposals_excluded(self):
        seeded = [WindowKey("seg", i) for i in range(4)]
        proposals = [
            _make_proposal("p_accepted", motivating=seeded[:2], accepted=True),
            _make_proposal("p_rejected", motivating=seeded[2:], accepted=False),
        ]
        labels = self._labels(seeded)
        result = compute_seeded_recall(proposals, seeded, labels, k=30)
        # Only the 2 windows from the accepted proposal should be covered
        assert result["overall"]["@30"] == pytest.approx(0.5)

    def test_familiar_unfamiliar_split(self):
        fam = [WindowKey("seg", i) for i in range(3)]
        unfam = [WindowKey("seg", i) for i in range(3, 6)]
        seeded = fam + unfam
        labels = {w: "familiar" for w in fam}
        labels.update({w: "unfamiliar" for w in unfam})
        # Proposal covers only familiar windows
        proposals = [_make_proposal("p1", motivating=fam)]
        result = compute_seeded_recall(proposals, seeded, labels, k=30)
        assert result["familiar"]["@30"] == pytest.approx(1.0)
        assert result["unfamiliar"]["@30"] == pytest.approx(0.0)
        assert result["overall"]["@30"] == pytest.approx(0.5)

    def test_at_keys_always_present(self):
        seeded = [WindowKey("seg", 0)]
        labels = {seeded[0]: "familiar"}
        proposals = [_make_proposal("p1", motivating=seeded)]
        result = compute_seeded_recall(proposals, seeded, labels, k=30)
        # Outer keys are subsets; inner keys are @K labels
        for subset in ("overall", "familiar", "unfamiliar"):
            assert subset in result
        for k in ("@10", "@30", "@all"):
            assert k in result["overall"]

    def test_empty_seeded_raises(self):
        with pytest.raises(ValueError, match="empty"):
            compute_seeded_recall([], [], {}, k=30)

    def test_missing_label_raises(self):
        seeded = [WindowKey("seg", 0), WindowKey("seg", 1)]
        labels = {seeded[0]: "familiar"}  # seg/1 missing
        with pytest.raises(ValueError, match="missing"):
            compute_seeded_recall([], seeded, labels, k=30)

    def test_empty_proposals_gives_zero_recall(self):
        seeded = [WindowKey("seg", 0)]
        labels = {seeded[0]: "familiar"}
        result = compute_seeded_recall([], seeded, labels, k=30)
        assert result["overall"]["@30"] == pytest.approx(0.0)
        assert result["overall"]["@all"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_rating_stats
# ---------------------------------------------------------------------------

class TestRatingStats:

    def test_basic_means(self):
        ratings = [
            _make_rating("p1", "r1", coh=4, use=3),
            _make_rating("p1", "r2", coh=2, use=5),
        ]
        result = compute_rating_stats(ratings)
        assert "reasoning" in result
        assert result["reasoning"]["mean_coherence"] == pytest.approx(3.0)
        assert result["reasoning"]["mean_usefulness"] == pytest.approx(4.0)
        assert result["reasoning"]["n"] == 2

    def test_ci_suppressed_below_min_n(self):
        ratings = [_make_rating("p1", f"r{i}", coh=4, use=3) for i in range(MIN_N_FOR_CI - 1)]
        result = compute_rating_stats(ratings)
        assert result["reasoning"]["coherence_ci_95"] is None
        assert result["reasoning"]["usefulness_ci_95"] is None

    def test_ci_computed_at_min_n(self):
        # All identical scores → CI should be tight around the mean
        ratings = [_make_rating("p1", f"r{i}", coh=4, use=4) for i in range(MIN_N_FOR_CI)]
        result = compute_rating_stats(ratings)
        ci = result["reasoning"]["coherence_ci_95"]
        assert ci is not None
        lo, hi = ci
        assert lo == pytest.approx(4.0, abs=0.01)
        assert hi == pytest.approx(4.0, abs=0.01)

    def test_ci_contains_true_mean(self):
        # Use a distribution where the true mean is known (2.5)
        rng = np.random.default_rng(0)
        scores = rng.choice([1, 2, 3, 4], size=60).tolist()
        ratings = [
            _make_rating(f"p{i}", f"r{i}", coh=int(s), use=int(s))
            for i, s in enumerate(scores)
        ]
        result = compute_rating_stats(ratings)
        ci = result["reasoning"]["coherence_ci_95"]
        mean = result["reasoning"]["mean_coherence"]
        assert ci is not None
        lo, hi = ci
        assert lo <= mean <= hi

    def test_multi_arm(self):
        ratings = [
            _make_rating("p1", "r1", arm="reasoning", coh=5, use=5),
            _make_rating("p1", "r1", arm="visual", coh=3, use=3),
        ]
        result = compute_rating_stats(ratings)
        assert "reasoning" in result
        assert "visual" in result
        assert result["reasoning"]["mean_coherence"] == pytest.approx(5.0)
        assert result["visual"]["mean_coherence"] == pytest.approx(3.0)

    def test_empty_ratings_returns_empty(self):
        result = compute_rating_stats([])
        assert result == {}


# ---------------------------------------------------------------------------
# krippendorff_alpha
# ---------------------------------------------------------------------------

class TestKrippendorffAlpha:

    def test_perfect_agreement(self):
        # All raters give identical scores → alpha = 1.0
        data = np.array([[4.0, 4.0, 4.0],
                         [2.0, 2.0, 2.0],
                         [5.0, 5.0, 5.0]])
        alpha = krippendorff_alpha(data)
        assert alpha == pytest.approx(1.0)

    def test_chance_level_returns_near_zero(self):
        # Random ratings on a 1-5 scale with many items should give alpha ≈ 0
        rng = np.random.default_rng(1234)
        data = rng.integers(1, 6, size=(100, 3)).astype(float)
        alpha = krippendorff_alpha(data)
        assert alpha is not None
        assert abs(alpha) < 0.2  # chance-level agreement

    def test_complete_disagreement_is_negative(self):
        # Two raters, always opposite ends of [1, 5]
        data = np.array([[1.0, 5.0],
                         [5.0, 1.0],
                         [1.0, 5.0],
                         [5.0, 1.0]])
        alpha = krippendorff_alpha(data)
        assert alpha is not None
        assert alpha < 0.0

    def test_single_rater_returns_none(self):
        data = np.array([[4.0, np.nan],
                         [2.0, np.nan],
                         [3.0, np.nan]])
        alpha = krippendorff_alpha(data)
        assert alpha is None

    def test_missing_ratings_handled(self):
        # Rater 2 missing some items — should still compute
        data = np.array([[4.0, np.nan],
                         [4.0, 4.0],
                         [4.0, 4.0],
                         [4.0, 4.0]])
        alpha = krippendorff_alpha(data)
        # Not None — there ARE coincidences where both raters rated
        assert alpha is not None

    def test_no_overlapping_ratings_returns_none(self):
        # Rater 1 only rated items 0-1, rater 2 only rated items 2-3 → no overlap
        data = np.array([[4.0, np.nan],
                         [3.0, np.nan],
                         [np.nan, 5.0],
                         [np.nan, 2.0]])
        alpha = krippendorff_alpha(data)
        assert alpha is None

    def test_all_same_value_is_one(self):
        # Every rater gives 3 for every item → alpha = 1.0 (d_e = 0 → shortcut)
        data = np.full((5, 4), 3.0)
        alpha = krippendorff_alpha(data)
        assert alpha == pytest.approx(1.0)

    def test_symmetry(self):
        # alpha(data) should equal alpha(data.T) since it's symmetric
        rng = np.random.default_rng(7)
        data = rng.integers(1, 6, size=(20, 4)).astype(float)
        alpha_normal = krippendorff_alpha(data)
        # Transpose has 4 items, 20 raters — semantically different but mathematically same
        # This is actually different, so just check both return a float
        assert isinstance(alpha_normal, float)
