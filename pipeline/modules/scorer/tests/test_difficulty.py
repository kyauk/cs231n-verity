"""Tests for scorer/difficulty.py."""
import pytest
from pipeline.modules.scorer.difficulty import (
    DifficultyArm,
    StubDifficultyClient,
    FailingDifficultyClient,
    compute_difficulty_signals,
    _extract_difficulty_json,
)

COMP_ID = "test_comp_difficulty"
CONSTITUENTS = ["weather:fog", "conditions:night_driving"]
MARGINAL = {"weather:fog": 0.15, "conditions:night_driving": 0.12}
EXPECTED_JOINT = 0.018
OBSERVED_JOINT = 0.001

WEIGHTS = (0.5, 0.3, 0.2)


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

class TestExtractDifficultyJson:
    def test_bare_json(self):
        result = _extract_difficulty_json(
            '{"action": "slow_down", "confidence": 0.6, "reasoning_consistent_with_action": true}'
        )
        assert result["action"] == "slow_down"
        assert result["confidence"] == pytest.approx(0.6)
        assert result["reasoning_consistent_with_action"] is True

    def test_confidence_clamped(self):
        result = _extract_difficulty_json(
            '{"action": "brake", "confidence": 1.5, "reasoning_consistent_with_action": false}'
        )
        assert result["confidence"] == pytest.approx(1.0)

    def test_missing_reasoning_field_defaults_true(self):
        result = _extract_difficulty_json('{"action": "yield", "confidence": 0.4}')
        assert result["reasoning_consistent_with_action"] is True

    def test_missing_required_fields_raises(self):
        with pytest.raises(ValueError):
            _extract_difficulty_json('{"justification": "no action or confidence"}')

    def test_unparseable_raises(self):
        with pytest.raises(ValueError):
            _extract_difficulty_json("I cannot determine the right action.")

    def test_fence_wrapped(self):
        result = _extract_difficulty_json(
            '```json\n{"action": "stop", "confidence": 0.9, "reasoning_consistent_with_action": true}\n```'
        )
        assert result["action"] == "stop"


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------

class TestComputeDifficultySignals:
    def test_all_agree(self):
        results = [
            {"action": "slow_down", "confidence": 0.8, "reasoning_consistent_with_action": True},
            {"action": "slow_down", "confidence": 0.7, "reasoning_consistent_with_action": True},
            {"action": "slow_down", "confidence": 0.9, "reasoning_consistent_with_action": True},
        ]
        score, signals = compute_difficulty_signals(results, WEIGHTS)
        assert signals["action_variance"] == pytest.approx(0.0)
        assert signals["reasoning_action_mismatch"] == pytest.approx(0.0)
        assert signals["mean_confidence"] == pytest.approx(0.8)
        # score = 0*0.5 + 0.2*0.3 + 0*0.2 = 0.06
        assert score == pytest.approx(0.06)

    def test_all_disagree(self):
        results = [
            {"action": "slow_down", "confidence": 0.2, "reasoning_consistent_with_action": False},
            {"action": "brake",     "confidence": 0.3, "reasoning_consistent_with_action": False},
            {"action": "yield",     "confidence": 0.1, "reasoning_consistent_with_action": False},
        ]
        score, signals = compute_difficulty_signals(results, WEIGHTS)
        # action_variance = 2/3 (2 differ from mode)
        assert signals["action_variance"] == pytest.approx(2 / 3)
        assert signals["reasoning_action_mismatch"] == pytest.approx(1.0)
        mean_conf = (0.2 + 0.3 + 0.1) / 3
        assert signals["mean_confidence"] == pytest.approx(mean_conf)

    def test_single_result(self):
        results = [
            {"action": "stop", "confidence": 0.5, "reasoning_consistent_with_action": True}
        ]
        score, signals = compute_difficulty_signals(results, WEIGHTS)
        assert signals["action_variance"] == pytest.approx(0.0)
        assert signals["mean_confidence"] == pytest.approx(0.5)

    def test_score_clipped_to_unit_interval(self):
        # Extreme case: everything maximally hard
        results = [
            {"action": "a", "confidence": 0.0, "reasoning_consistent_with_action": False},
            {"action": "b", "confidence": 0.0, "reasoning_consistent_with_action": False},
            {"action": "c", "confidence": 0.0, "reasoning_consistent_with_action": False},
        ]
        score, _ = compute_difficulty_signals(results, WEIGHTS)
        assert 0.0 <= score <= 1.0

    def test_signals_dict_has_three_keys(self):
        results = [{"action": "slow_down", "confidence": 0.6, "reasoning_consistent_with_action": True}]
        _, signals = compute_difficulty_signals(results, WEIGHTS)
        assert set(signals.keys()) == {"mean_confidence", "action_variance", "reasoning_action_mismatch"}

    def test_signal_weights_respected(self):
        # With only action_variance non-zero, score = action_variance * w[0]
        results = [
            {"action": "slow_down", "confidence": 1.0, "reasoning_consistent_with_action": True},
            {"action": "brake",     "confidence": 1.0, "reasoning_consistent_with_action": True},
        ]
        _, signals = compute_difficulty_signals(results, (0.5, 0.3, 0.2))
        # action_variance = 0.5 (1 of 2 differs from mode)
        # score = 0.5*0.5 + 0*0.3 + 0*0.2 = 0.25
        assert signals["action_variance"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# DifficultyArm
# ---------------------------------------------------------------------------

class TestDifficultyArm:
    def test_stub_returns_score_and_signals(self):
        arm = DifficultyArm(StubDifficultyClient())
        score, signals = arm.score(COMP_ID, CONSTITUENTS, MARGINAL, EXPECTED_JOINT, OBSERVED_JOINT)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert isinstance(signals, dict)
        assert "mean_confidence" in signals

    def test_stub_is_deterministic(self):
        arm = DifficultyArm(StubDifficultyClient())
        r1 = arm.score(COMP_ID, CONSTITUENTS, MARGINAL, EXPECTED_JOINT, OBSERVED_JOINT)
        r2 = arm.score(COMP_ID, CONSTITUENTS, MARGINAL, EXPECTED_JOINT, OBSERVED_JOINT)
        assert r1 == r2

    def test_all_fail_returns_none(self):
        arm = DifficultyArm(FailingDifficultyClient())
        score, signals = arm.score(COMP_ID, CONSTITUENTS, MARGINAL, EXPECTED_JOINT, OBSERVED_JOINT)
        assert score is None
        assert signals == {}

    def test_partial_failure_still_returns_score(self):
        call_count = 0

        class PartialClient:
            model_id = "stub/partial"
            def complete(self, prompt: str) -> str:
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    return "garbage"
                return '{"action": "slow_down", "confidence": 0.6, "reasoning_consistent_with_action": true}'

        arm = DifficultyArm(PartialClient())
        score, signals = arm.score(COMP_ID, CONSTITUENTS, MARGINAL, EXPECTED_JOINT, OBSERVED_JOINT)
        assert score is not None
        assert isinstance(signals, dict)

    def test_custom_signal_weights(self):
        arm = DifficultyArm(StubDifficultyClient(), signal_weights=(0.6, 0.2, 0.2))
        score, signals = arm.score(COMP_ID, CONSTITUENTS, MARGINAL, EXPECTED_JOINT, OBSERVED_JOINT)
        assert score is not None
