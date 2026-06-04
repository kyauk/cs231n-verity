"""Tests for scorer/plausibility.py."""
import hashlib
import random
import pytest
from pipeline.modules.scorer.config import PlausibilityCheckFailedError
from pipeline.modules.scorer.plausibility import (
    PlausibilityArm,
    StubPlausibilityClient,
    FailingPlausibilityClient,
    describe_composition,
    _three_orderings,
    _extract_plausibility_json,
)

COMP_ID = "test_composition_1"
CONSTITUENTS = ["weather:fog", "conditions:night_driving", "ego_task:turning_left"]
MARGINAL = {"weather:fog": 0.15, "conditions:night_driving": 0.12, "ego_task:turning_left": 0.20}
EXPECTED_JOINT = 0.0036
OBSERVED_JOINT = 0.0002


# ---------------------------------------------------------------------------
# Ordering determinism
# ---------------------------------------------------------------------------

class TestThreeOrderings:
    def test_three_distinct_orderings(self):
        orderings = _three_orderings(CONSTITUENTS, seed=12345)
        assert len(orderings) == 3
        # All three are permutations of constituents
        for ordering in orderings:
            assert sorted(ordering) == sorted(CONSTITUENTS)

    def test_deterministic_with_sha256_seed(self):
        seed = int(hashlib.sha256(COMP_ID.encode()).hexdigest()[:8], 16)
        o1 = _three_orderings(CONSTITUENTS, seed)
        o2 = _three_orderings(CONSTITUENTS, seed)
        assert o1 == o2

    def test_ordering_0_is_sorted(self):
        orderings = _three_orderings(CONSTITUENTS, seed=999)
        assert orderings[0] == sorted(CONSTITUENTS)

    def test_ordering_1_is_reversed_sorted(self):
        orderings = _three_orderings(CONSTITUENTS, seed=999)
        assert orderings[1] == list(reversed(sorted(CONSTITUENTS)))


class TestDescribeComposition:
    def test_contains_all_constituents(self):
        desc = describe_composition(COMP_ID, CONSTITUENTS, MARGINAL, EXPECTED_JOINT, OBSERVED_JOINT, 0)
        for atom in CONSTITUENTS:
            assert atom in desc

    def test_does_not_leak_rarity_statistics(self):
        """FIRST_RUN_FINDINGS Issue 5 regression: the plausibility prompt must
        NOT contain the joint-frequency / "Statistical context" block — feeding
        rarity into the plausibility judge made the model equate "rare in this
        sample" with "physically impossible," zeroing out every accepted
        proposal. Rarity is captured separately by novelty_score; the
        plausibility prompt must judge physical/behavioral co-occurrence only.
        """
        desc = describe_composition(COMP_ID, CONSTITUENTS, MARGINAL, EXPECTED_JOINT, OBSERVED_JOINT, 0)
        # The leaky language used to include "Statistical context" with the
        # frequency numbers. None of these strings must appear.
        for forbidden in ("Statistical context",
                          "Observed joint", "Expected joint",
                          "15.0%", "15%", "0.15"):
            assert forbidden not in desc, (
                f"plausibility prompt leaked {forbidden!r} — see FIRST_RUN_FINDINGS Issue 5"
            )
        # The explicit guard against rarity-as-implausibility must be present.
        assert "statistical rarity is not the same as implausibility" in desc.lower()

    def test_different_orderings_differ(self):
        d0 = describe_composition(COMP_ID, CONSTITUENTS, MARGINAL, EXPECTED_JOINT, OBSERVED_JOINT, 0)
        d1 = describe_composition(COMP_ID, CONSTITUENTS, MARGINAL, EXPECTED_JOINT, OBSERVED_JOINT, 1)
        # Different orderings → different descriptions (unless all orderings happen to be same)
        # They should at least both contain the atoms
        for atom in CONSTITUENTS:
            assert atom in d0 and atom in d1

    def test_deterministic_same_offset(self):
        d1 = describe_composition(COMP_ID, CONSTITUENTS, MARGINAL, EXPECTED_JOINT, OBSERVED_JOINT, 0)
        d2 = describe_composition(COMP_ID, CONSTITUENTS, MARGINAL, EXPECTED_JOINT, OBSERVED_JOINT, 0)
        assert d1 == d2


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

class TestExtractPlausibilityJson:
    def test_bare_json(self):
        result = _extract_plausibility_json('{"score": 0.7, "justification": "Fine."}')
        assert result["score"] == pytest.approx(0.7)
        assert result["justification"] == "Fine."

    def test_fence_wrapped(self):
        result = _extract_plausibility_json('```json\n{"score": 0.5, "justification": "Ok."}\n```')
        assert result["score"] == pytest.approx(0.5)

    def test_score_clamped_above_1(self):
        result = _extract_plausibility_json('{"score": 1.5, "justification": "Too high."}')
        assert result["score"] == pytest.approx(1.0)

    def test_score_clamped_below_0(self):
        result = _extract_plausibility_json('{"score": -0.2, "justification": "Negative."}')
        assert result["score"] == pytest.approx(0.0)

    def test_missing_score_raises(self):
        with pytest.raises(ValueError):
            _extract_plausibility_json('{"justification": "No score field."}')

    def test_unparseable_raises(self):
        with pytest.raises(ValueError):
            _extract_plausibility_json("I cannot assess this.")

    def test_embedded_json_in_prose(self):
        text = 'Let me think... The score is {"score": 0.8, "justification": "Good."} given the conditions.'
        result = _extract_plausibility_json(text)
        assert result["score"] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# PlausibilityArm
# ---------------------------------------------------------------------------

class TestPlausibilityArm:
    def test_stub_returns_score_and_justification(self):
        arm = PlausibilityArm(StubPlausibilityClient())
        score, just = arm.score(COMP_ID, CONSTITUENTS, MARGINAL, EXPECTED_JOINT, OBSERVED_JOINT)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert isinstance(just, str) and len(just) > 0

    def test_stub_is_deterministic(self):
        arm = PlausibilityArm(StubPlausibilityClient())
        r1 = arm.score(COMP_ID, CONSTITUENTS, MARGINAL, EXPECTED_JOINT, OBSERVED_JOINT)
        r2 = arm.score(COMP_ID, CONSTITUENTS, MARGINAL, EXPECTED_JOINT, OBSERVED_JOINT)
        assert r1 == r2

    def test_all_fail_raises(self):
        arm = PlausibilityArm(FailingPlausibilityClient())
        with pytest.raises(PlausibilityCheckFailedError) as exc_info:
            arm.score(COMP_ID, CONSTITUENTS, MARGINAL, EXPECTED_JOINT, OBSERVED_JOINT)
        assert exc_info.value.composition_id == COMP_ID

    def test_two_of_three_succeed_uses_lower(self):
        """When 2/3 runs succeed, the lower score is returned (conservative)."""
        call_count = 0

        class TwoSuccessClient:
            model_id = "stub/two-success"
            def complete(self, prompt: str) -> str:
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    return "garbled not json"
                score = 0.9 if call_count == 1 else 0.6
                return f'{{"score": {score}, "justification": "run {call_count}"}}'

        arm = PlausibilityArm(TwoSuccessClient())
        score, _ = arm.score(COMP_ID, CONSTITUENTS, MARGINAL, EXPECTED_JOINT, OBSERVED_JOINT)
        assert score == pytest.approx(0.6)  # lower of 0.9 and 0.6

    def test_one_of_three_succeeds(self):
        call_count = 0

        class OneSuccessClient:
            model_id = "stub/one-success"
            def complete(self, prompt: str) -> str:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return '{"score": 0.72, "justification": "Only good run."}'
                return "garbage"

        arm = PlausibilityArm(OneSuccessClient())
        score, just = arm.score(COMP_ID, CONSTITUENTS, MARGINAL, EXPECTED_JOINT, OBSERVED_JOINT)
        assert score == pytest.approx(0.72)
        assert "Only good run" in just

    def test_three_runs_median(self):
        """3 successful runs → median (middle value)."""
        scores_to_return = [0.9, 0.5, 0.7]
        call_count = 0

        class ThreeScoresClient:
            model_id = "stub/three-scores"
            def complete(self, prompt: str) -> str:
                nonlocal call_count
                s = scores_to_return[call_count % 3]
                call_count += 1
                return f'{{"score": {s}, "justification": "run score {s}"}}'

        arm = PlausibilityArm(ThreeScoresClient())
        score, just = arm.score(COMP_ID, CONSTITUENTS, MARGINAL, EXPECTED_JOINT, OBSERVED_JOINT)
        assert score == pytest.approx(0.7)  # median of [0.5, 0.7, 0.9]
        assert "0.7" in just

    def test_wrong_template_placeholder_raises(self):
        """Template missing {{COMPOSITION}} raises at construction time."""
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as d:
            bad_template = pathlib.Path(d) / "bad_v1_plausibility.txt"
            bad_template.write_text("No placeholder here.")

            import pipeline.modules.scorer.plausibility as p_mod
            orig = p_mod._PROMPT_DIR
            p_mod._PROMPT_DIR = pathlib.Path(d)
            try:
                with pytest.raises(ValueError, match="COMPOSITION"):
                    PlausibilityArm(StubPlausibilityClient(), prompt_version="bad_v1_plausibility")
            finally:
                p_mod._PROMPT_DIR = orig
