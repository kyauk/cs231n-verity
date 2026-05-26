"""Tests for scorer/config.py."""
import pytest
from pipeline.modules.scorer.config import (
    ScorerConfig,
    ScorerWeights,
    TextClient,
    ScorerError,
    PlausibilityCheckFailedError,
)


class TestScorerWeights:
    def test_defaults(self):
        w = ScorerWeights()
        assert w.novelty == 0.4
        assert w.plausibility == 0.3
        assert w.difficulty == 0.3

    def test_custom_weights(self):
        w = ScorerWeights(novelty=0.5, plausibility=0.3, difficulty=0.2)
        assert w.novelty == 0.5

    def test_default_weights_sum_to_one(self):
        w = ScorerWeights()
        assert abs(w.novelty + w.plausibility + w.difficulty - 1.0) < 1e-9


class TestScorerConfig:
    def test_defaults(self):
        cfg = ScorerConfig()
        assert cfg.plausibility_threshold == 0.5
        assert cfg.plausibility_runs == 3
        assert cfg.difficulty_runs == 3
        assert cfg.plausibility_prompt_version == "v1_plausibility"
        assert cfg.difficulty_prompt_version == "v1_difficulty"

    def test_difficulty_signal_weights_defaults(self):
        cfg = ScorerConfig()
        w = cfg.difficulty_signal_weights
        assert len(w) == 3
        assert w[0] == 0.5   # action_variance
        assert w[1] == 0.3   # inverse_confidence
        assert w[2] == 0.2   # reasoning_mismatch
        assert abs(sum(w) - 1.0) < 1e-9

    def test_custom_config(self):
        cfg = ScorerConfig(plausibility_threshold=0.65, plausibility_runs=5)
        assert cfg.plausibility_threshold == 0.65
        assert cfg.plausibility_runs == 5

    def test_weights_instance_per_config(self):
        a = ScorerConfig()
        b = ScorerConfig()
        a.weights.novelty = 0.9
        assert b.weights.novelty == 0.4  # independent instances


class TestTextClientProtocol:
    def test_stub_satisfies_protocol(self):
        class MyClient:
            model_id = "stub/test"
            def complete(self, prompt: str) -> str:
                return "response"

        client = MyClient()
        assert isinstance(client, TextClient)

    def test_missing_model_id_fails_protocol(self):
        class BadClient:
            def complete(self, prompt: str) -> str:
                return "response"

        # Protocol runtime check: TextClient requires model_id attribute
        client = BadClient()
        assert not isinstance(client, TextClient)

    def test_missing_complete_fails_protocol(self):
        class BadClient:
            model_id = "stub"

        client = BadClient()
        assert not isinstance(client, TextClient)


class TestErrors:
    def test_plausibility_check_failed_message(self):
        exc = PlausibilityCheckFailedError("comp_abc123", "all 3 runs failed")
        assert "comp_abc123" in str(exc)
        assert "all 3 runs failed" in str(exc)
        assert exc.composition_id == "comp_abc123"
        assert exc.detail == "all 3 runs failed"

    def test_plausibility_check_failed_is_scorer_error(self):
        exc = PlausibilityCheckFailedError("x", "y")
        assert isinstance(exc, ScorerError)
        assert isinstance(exc, Exception)
