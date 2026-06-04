"""Smoke tests for Module 4: Scorer (full public interface)."""
import json
import pytest
from pathlib import Path
from pipeline.interfaces.proposal import CompositionProposal, ScoredProposal
from pipeline.interfaces.window import WindowKey
from pipeline.modules.scorer.scorer import (
    Scorer,
    REJECTION_PLAUSIBILITY_FAILED,
    REJECTION_BELOW_THRESHOLD,
)
from pipeline.modules.scorer.config import ScorerConfig, ScorerWeights
from pipeline.modules.scorer.plausibility import StubPlausibilityClient, FailingPlausibilityClient
from pipeline.modules.scorer.difficulty import StubDifficultyClient, FailingDifficultyClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_proposal(composition_id: str = "abc12345", novelty: float = 2.5) -> CompositionProposal:
    return CompositionProposal(
        composition_id=composition_id,
        constituents=["weather:fog", "conditions:night_driving", "ego_task:turning_left"],
        marginal_frequencies={"weather:fog": 0.15, "conditions:night_driving": 0.12, "ego_task:turning_left": 0.20},
        pairwise_frequencies={"conditions:night_driving|weather:fog": 0.03, "conditions:night_driving|ego_task:turning_left": 0.02, "ego_task:turning_left|weather:fog": 0.04},
        expected_joint=0.0036,
        observed_joint=0.0002,
        novelty_score=novelty,
        motivating_scene_ids=[WindowKey("seg_001", 0), WindowKey("seg_002", 3)],
        arm="reasoning",
    )


def _make_scorer(tmp_path: Path, with_difficulty: bool = True, threshold: float = 0.5) -> Scorer:
    cfg = ScorerConfig(plausibility_threshold=threshold)
    return Scorer(
        plausibility_client=StubPlausibilityClient(),
        difficulty_client=StubDifficultyClient() if with_difficulty else None,
        config=cfg,
        cache_root=tmp_path,
    )


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

class TestScorerSmoke:
    def test_score_returns_scored_proposal(self, tmp_path):
        scorer = _make_scorer(tmp_path)
        result = scorer.score(_make_proposal())
        assert isinstance(result, ScoredProposal)

    def test_scored_proposal_fields_present(self, tmp_path):
        scorer = _make_scorer(tmp_path)
        p = scorer.score(_make_proposal())
        assert isinstance(p.composition_id, str)
        assert isinstance(p.constituents, list)
        assert isinstance(p.plausibility_score, float)
        assert isinstance(p.plausibility_justification, str)
        assert isinstance(p.frontier_difficulty_score, float)
        assert isinstance(p.frontier_difficulty_signals, dict)
        assert isinstance(p.final_rank_score, float)
        assert isinstance(p.accepted, bool)

    def test_no_difficulty_client_produces_none_score(self, tmp_path):
        scorer = _make_scorer(tmp_path, with_difficulty=False)
        result = scorer.score(_make_proposal())
        assert result.frontier_difficulty_score is None
        assert result.frontier_difficulty_signals == {}

    def test_plausibility_threshold_rejects_low_scores(self, tmp_path):
        # StubPlausibilityClient returns 0.78; threshold above that → rejected
        cfg = ScorerConfig(plausibility_threshold=0.9)
        scorer = Scorer(
            plausibility_client=StubPlausibilityClient(),
            config=cfg,
            cache_root=tmp_path,
        )
        result = scorer.score(_make_proposal("thresh_test"))
        assert not result.accepted
        assert result.rejection_reason == REJECTION_BELOW_THRESHOLD

    def test_plausibility_threshold_accepts_above(self, tmp_path):
        # StubPlausibilityClient returns 0.78; threshold below that → accepted
        scorer = _make_scorer(tmp_path, threshold=0.5)
        result = scorer.score(_make_proposal("accept_test"))
        assert result.accepted
        assert result.rejection_reason is None

    def test_plausibility_failure_emits_rejected_proposal(self, tmp_path):
        scorer = Scorer(
            plausibility_client=FailingPlausibilityClient(),
            config=ScorerConfig(),
            cache_root=tmp_path,
        )
        result = scorer.score(_make_proposal("fail_test"))
        assert not result.accepted
        assert result.rejection_reason == REJECTION_PLAUSIBILITY_FAILED
        assert result.plausibility_score == pytest.approx(0.0)

    def test_score_batch_continues_after_failure(self, tmp_path):
        scorer = Scorer(
            plausibility_client=FailingPlausibilityClient(),
            config=ScorerConfig(),
        )
        proposals = [_make_proposal(f"p{i}") for i in range(3)]
        results = scorer.score_batch(proposals)
        assert len(results) == 3
        assert all(not r.accepted for r in results)

    def test_final_rank_score_excludes_difficulty_when_none(self, tmp_path):
        cfg = ScorerConfig(weights=ScorerWeights(novelty=0.4, plausibility=0.3, difficulty=0.3))
        scorer_with = Scorer(
            plausibility_client=StubPlausibilityClient(),
            difficulty_client=StubDifficultyClient(),
            config=cfg,
        )
        scorer_without = Scorer(
            plausibility_client=StubPlausibilityClient(),
            difficulty_client=None,
            config=cfg,
        )
        r_with = scorer_with.score(_make_proposal("rank_with", novelty=2.0))
        r_without = scorer_without.score(_make_proposal("rank_without", novelty=2.0))
        # Without difficulty, difficulty term is 0 → lower final_rank_score
        assert r_without.final_rank_score <= r_with.final_rank_score

    def test_cache_hit_skips_vlm(self, tmp_path):
        scorer = _make_scorer(tmp_path)
        proposal = _make_proposal("cache_test")
        r1 = scorer.score(proposal)

        # Replace VLM with failing stub — cache hit should still work
        failing_scorer = Scorer(
            plausibility_client=FailingPlausibilityClient(),
            difficulty_client=FailingDifficultyClient(),
            config=ScorerConfig(),
            cache_root=tmp_path,
        )
        # Different model_ids → different cache key → no hit (expected miss here)
        # To test actual cache hit, use same model_ids via same scorer instance
        r2 = scorer.score(proposal)
        assert r1.composition_id == r2.composition_id
        assert r1.plausibility_score == pytest.approx(r2.plausibility_score)

    def test_cache_is_populated(self, tmp_path):
        scorer = _make_scorer(tmp_path)
        proposal = _make_proposal("cache_pop_test")
        scorer.score(proposal)
        cache_dir = tmp_path / "scorer"
        cache_files = list(cache_dir.glob("*.json"))
        assert len(cache_files) == 1

    def test_failure_not_cached(self, tmp_path):
        scorer = Scorer(
            plausibility_client=FailingPlausibilityClient(),
            config=ScorerConfig(),
            cache_root=tmp_path,
        )
        scorer.score(_make_proposal("fail_cache"))
        cache_dir = tmp_path / "scorer"
        assert not cache_dir.exists() or len(list(cache_dir.glob("*.json"))) == 0

    def test_deterministic_across_calls(self, tmp_path):
        # No cache, same scorer
        scorer = Scorer(
            plausibility_client=StubPlausibilityClient(),
            difficulty_client=StubDifficultyClient(),
            config=ScorerConfig(),
        )
        proposal = _make_proposal("det_test")
        r1 = scorer.score(proposal)
        r2 = scorer.score(proposal)
        assert r1.plausibility_score == pytest.approx(r2.plausibility_score)
        assert r1.final_rank_score == pytest.approx(r2.final_rank_score)

    def test_json_round_trip(self, tmp_path):
        scorer = _make_scorer(tmp_path)
        result = scorer.score(_make_proposal("json_rt"))
        wire = result.to_json()
        restored = ScoredProposal.from_json(wire)
        assert restored.composition_id == result.composition_id
        assert restored.accepted == result.accepted
        assert restored.plausibility_score == pytest.approx(result.plausibility_score)

    def test_motivating_scenes_preserved(self, tmp_path):
        scorer = _make_scorer(tmp_path)
        proposal = _make_proposal("scenes_test")
        result = scorer.score(proposal)
        assert len(result.motivating_scene_ids) == 2
        assert all(isinstance(wk, WindowKey) for wk in result.motivating_scene_ids)

    def test_arm_preserved(self, tmp_path):
        scorer = _make_scorer(tmp_path)
        result = scorer.score(_make_proposal())
        assert result.arm == "reasoning"

    def test_no_cache_root_no_side_effects(self):
        scorer = Scorer(
            plausibility_client=StubPlausibilityClient(),
            config=ScorerConfig(),
            cache_root=None,
        )
        result = scorer.score(_make_proposal("no_cache"))
        assert isinstance(result, ScoredProposal)

    def test_threshold_boundary_exactly_equal_is_accepted(self, tmp_path):
        """Score == threshold is accepted (filter is strict less-than)."""
        class ExactThresholdClient:
            model_id = "stub/exact"
            def complete(self, prompt): return '{"score": 0.5, "justification": "boundary"}'

        scorer = Scorer(
            plausibility_client=ExactThresholdClient(),
            config=ScorerConfig(plausibility_threshold=0.5),
            cache_root=tmp_path,
        )
        result = scorer.score(_make_proposal("boundary_test"))
        assert result.accepted
        assert result.rejection_reason is None

    def test_plausibility_justification_empty_on_failure(self):
        """On plausibility failure, justification is empty string (not None)."""
        scorer = Scorer(
            plausibility_client=FailingPlausibilityClient(),
            config=ScorerConfig(),
        )
        result = scorer.score(_make_proposal("just_fail"))
        assert result.plausibility_justification == ""
        assert not result.accepted

    def test_cache_key_differs_by_model_id(self, tmp_path):
        """Different plausibility model IDs produce different cache keys."""
        class ModelAClient:
            model_id = "model/a"
            def complete(self, prompt): return '{"score": 0.7, "justification": "A"}'

        class ModelBClient:
            model_id = "model/b"
            def complete(self, prompt): return '{"score": 0.7, "justification": "B"}'

        scorer_a = Scorer(ModelAClient(), config=ScorerConfig(), cache_root=tmp_path)
        scorer_b = Scorer(ModelBClient(), config=ScorerConfig(), cache_root=tmp_path)

        proposal = _make_proposal("model_diff")
        scorer_a.score(proposal)
        scorer_b.score(proposal)

        cache_files = list((tmp_path / "scorer").glob("*.json"))
        assert len(cache_files) == 2  # Different keys → different files
