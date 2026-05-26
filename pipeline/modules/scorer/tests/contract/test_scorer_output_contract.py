"""Module 4: Scorer — output contract tests.

Validates every field declared in the README Output Contract section for
ScoredProposal. One assertion per contract requirement.
"""
import pytest
from pipeline.interfaces.proposal import ScoredProposal
from pipeline.interfaces.window import WindowKey
from pipeline.modules.scorer.scorer import Scorer, REJECTION_PLAUSIBILITY_FAILED, REJECTION_BELOW_THRESHOLD
from pipeline.modules.scorer.config import ScorerConfig
from pipeline.modules.scorer.plausibility import StubPlausibilityClient, FailingPlausibilityClient
from pipeline.modules.scorer.difficulty import StubDifficultyClient
from pipeline.interfaces.proposal import CompositionProposal


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

def _make_proposal(cid: str = "contract_test", novelty: float = 2.5) -> CompositionProposal:
    return CompositionProposal(
        composition_id=cid,
        constituents=["weather:fog", "conditions:night_driving"],
        marginal_frequencies={"weather:fog": 0.15, "conditions:night_driving": 0.12},
        pairwise_frequencies={"conditions:night_driving|weather:fog": 0.02},
        expected_joint=0.018,
        observed_joint=0.001,
        novelty_score=novelty,
        motivating_scene_ids=[WindowKey("seg_001", 0), WindowKey("seg_002", 3)],
        arm="reasoning",
    )


_scorer = Scorer(
    plausibility_client=StubPlausibilityClient(),
    difficulty_client=StubDifficultyClient(),
    config=ScorerConfig(plausibility_threshold=0.5),
)
_result = _scorer.score(_make_proposal())


# ---------------------------------------------------------------------------
# Contract tests (README: Module 4 — Output Contract)
# ---------------------------------------------------------------------------

def test_output_is_scored_proposal_instance():
    assert isinstance(_result, ScoredProposal)


def test_composition_id_preserved():
    assert _result.composition_id == "contract_test"


def test_constituents_preserved():
    assert _result.constituents == ["weather:fog", "conditions:night_driving"]


def test_marginal_frequencies_preserved():
    assert "weather:fog" in _result.marginal_frequencies
    assert isinstance(_result.marginal_frequencies["weather:fog"], float)


def test_pairwise_frequencies_preserved():
    assert isinstance(_result.pairwise_frequencies, dict)


def test_expected_joint_preserved():
    assert isinstance(_result.expected_joint, float)
    assert _result.expected_joint == pytest.approx(0.018)


def test_observed_joint_preserved():
    assert isinstance(_result.observed_joint, float)


def test_novelty_score_preserved():
    assert isinstance(_result.novelty_score, float)
    assert _result.novelty_score == pytest.approx(2.5)


def test_motivating_scene_ids_are_window_keys():
    assert isinstance(_result.motivating_scene_ids, list)
    for wk in _result.motivating_scene_ids:
        assert isinstance(wk, WindowKey)


def test_arm_preserved():
    assert _result.arm == "reasoning"


def test_plausibility_score_is_float_in_unit_interval():
    assert isinstance(_result.plausibility_score, float)
    assert 0.0 <= _result.plausibility_score <= 1.0


def test_plausibility_justification_is_string():
    assert isinstance(_result.plausibility_justification, str)
    assert len(_result.plausibility_justification) > 0


def test_frontier_difficulty_score_is_float_or_none():
    assert _result.frontier_difficulty_score is None or isinstance(_result.frontier_difficulty_score, float)
    if _result.frontier_difficulty_score is not None:
        assert 0.0 <= _result.frontier_difficulty_score <= 1.0


def test_frontier_difficulty_signals_is_dict():
    assert isinstance(_result.frontier_difficulty_signals, dict)


def test_final_rank_score_is_float():
    assert isinstance(_result.final_rank_score, float)


def test_accepted_is_bool():
    assert isinstance(_result.accepted, bool)


def test_rejection_reason_is_none_or_string():
    assert _result.rejection_reason is None or isinstance(_result.rejection_reason, str)


def test_accepted_and_rejection_reason_consistent():
    """accepted=True implies rejection_reason=None; accepted=False implies reason set."""
    if _result.accepted:
        assert _result.rejection_reason is None
    else:
        assert _result.rejection_reason is not None


def test_plausibility_failure_rejection_reason():
    scorer = Scorer(plausibility_client=FailingPlausibilityClient(), config=ScorerConfig())
    result = scorer.score(_make_proposal("fail_contract"))
    assert not result.accepted
    assert result.rejection_reason == REJECTION_PLAUSIBILITY_FAILED
    assert result.plausibility_score == pytest.approx(0.0)


def test_below_threshold_rejection_reason():
    scorer = Scorer(
        plausibility_client=StubPlausibilityClient(),
        config=ScorerConfig(plausibility_threshold=0.99),  # stub returns 0.78 → rejected
    )
    result = scorer.score(_make_proposal("threshold_contract"))
    assert not result.accepted
    assert result.rejection_reason == REJECTION_BELOW_THRESHOLD


def test_no_difficulty_client_sets_none():
    scorer = Scorer(plausibility_client=StubPlausibilityClient(), difficulty_client=None)
    result = scorer.score(_make_proposal("no_diff"))
    assert result.frontier_difficulty_score is None
    assert result.frontier_difficulty_signals == {}


def test_json_round_trip():
    wire = _result.to_json()
    restored = ScoredProposal.from_json(wire)
    assert restored.composition_id == _result.composition_id
    assert restored.plausibility_score == pytest.approx(_result.plausibility_score)
    assert restored.accepted == _result.accepted
    assert restored.arm == _result.arm
    assert len(restored.motivating_scene_ids) == len(_result.motivating_scene_ids)


def test_side_effect_free():
    """score() on the same proposal twice produces identical results."""
    r1 = _scorer.score(_make_proposal("idempotent"))
    r2 = _scorer.score(_make_proposal("idempotent"))
    assert r1.plausibility_score == pytest.approx(r2.plausibility_score)
    assert r1.final_rank_score == pytest.approx(r2.final_rank_score)
