"""Integration test: Module 3 (Hypothesizer) → Module 4 (Scorer).

Runs the real Hypothesizer on a small fixture, passes every CompositionProposal
directly to the real Scorer, and verifies Scorer can consume the output without
modification.
"""
import pytest
from pipeline.interfaces.proposal import CompositionProposal, ScoredProposal
from pipeline.interfaces.schema_record import SchemaRecord
from pipeline.interfaces.window import WindowKey
from pipeline.modules.hypothesizer.hypothesizer import Hypothesizer
from pipeline.modules.hypothesizer.config import HypothesizerConfig
from pipeline.modules.scorer.scorer import Scorer
from pipeline.modules.scorer.config import ScorerConfig
from pipeline.modules.scorer.plausibility import StubPlausibilityClient
from pipeline.modules.scorer.difficulty import StubDifficultyClient


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_record(segment_id: str, window_idx: int, fields: dict) -> SchemaRecord:
    return SchemaRecord(
        window_id=WindowKey(segment_id, window_idx),
        arm="reasoning",
        schema_version="v1",
        prompt_template_id="v1_reasoning",
        fields=fields,
        failure_mode=None,
    )


def _standard_fields(
    weather: str = "weather:clear",
    time_of_day: str = "time_of_day:day",
    fog: bool = False,
    night: bool = False,
) -> dict:
    conditions = []
    if fog:
        conditions.append("conditions:fog")
    if night:
        conditions.append("conditions:night_driving")
    return {
        "agents": ["agents:pedestrian"],
        "environment": {
            "weather": weather,
            "time_of_day": time_of_day,
            "lighting_condition": "lighting_condition:overcast",
        },
        "road": {"geometry": "road_geometry:straight", "lane_count": 2},
        "traffic_control": "traffic_control:stop_sign",
        "ego_task": "ego_task:straight",
        "conditions": conditions,
    }


def _build_fixture_records():
    """30 records: 20 with fog+night driving (compositionally rare), 10 normal."""
    records = []

    # 20 records with both fog and night driving conditions
    for i in range(20):
        records.append(_make_record(
            f"seg_{i:03d}", 0,
            _standard_fields(
                weather="weather:fog",
                time_of_day="time_of_day:night",
                fog=True,
                night=True,
            ),
        ))

    # 10 records with clear weather, day
    for i in range(20, 30):
        records.append(_make_record(
            f"seg_{i:03d}", 0,
            _standard_fields(
                weather="weather:clear",
                time_of_day="time_of_day:day",
            ),
        ))

    return records


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHypothesizerToScorer:

    @pytest.fixture
    def proposals(self):
        cfg = HypothesizerConfig(
            min_marginal_frequency=0.05,
            max_joint_frequency=0.99,   # permissive — let everything through
            min_pairwise_frequency=0.0,
            composition_sizes=[2],
            top_k=10,
        )
        hyp = Hypothesizer(config=cfg)
        records = _build_fixture_records()
        return hyp.propose(records, arm="reasoning")

    @pytest.fixture
    def scorer(self):
        return Scorer(
            plausibility_client=StubPlausibilityClient(),
            difficulty_client=StubDifficultyClient(),
            config=ScorerConfig(plausibility_threshold=0.5),
        )

    def test_hypothesizer_produces_proposals(self, proposals):
        assert len(proposals) > 0
        assert all(isinstance(p, CompositionProposal) for p in proposals)

    def test_scorer_accepts_every_proposal(self, proposals, scorer):
        """Scorer must not raise on any proposal from Hypothesizer."""
        results = scorer.score_batch(proposals)
        assert len(results) == len(proposals)

    def test_all_results_are_scored_proposals(self, proposals, scorer):
        results = scorer.score_batch(proposals)
        assert all(isinstance(r, ScoredProposal) for r in results)

    def test_composition_ids_preserved_across_boundary(self, proposals, scorer):
        """composition_id must be the same before and after scoring."""
        results = scorer.score_batch(proposals)
        proposal_ids = {p.composition_id for p in proposals}
        result_ids = {r.composition_id for r in results}
        assert proposal_ids == result_ids

    def test_motivating_scene_ids_preserved(self, proposals, scorer):
        results = scorer.score_batch(proposals)
        for r in results:
            assert isinstance(r.motivating_scene_ids, list)
            assert all(isinstance(wk, WindowKey) for wk in r.motivating_scene_ids)

    def test_arm_preserved(self, proposals, scorer):
        results = scorer.score_batch(proposals)
        assert all(r.arm == "reasoning" for r in results)

    def test_scored_proposals_have_valid_scores(self, proposals, scorer):
        results = scorer.score_batch(proposals)
        for r in results:
            assert 0.0 <= r.plausibility_score <= 1.0
            if r.frontier_difficulty_score is not None:
                assert 0.0 <= r.frontier_difficulty_score <= 1.0

    def test_acceptance_and_rejection_consistent(self, proposals, scorer):
        results = scorer.score_batch(proposals)
        for r in results:
            if r.accepted:
                assert r.rejection_reason is None
            else:
                assert r.rejection_reason is not None

    def test_json_round_trip_for_all_results(self, proposals, scorer):
        """Every ScoredProposal produced from Hypothesizer output survives JSON round-trip."""
        results = scorer.score_batch(proposals)
        for r in results:
            wire = r.to_json()
            restored = ScoredProposal.from_json(wire)
            assert restored.composition_id == r.composition_id
            assert restored.accepted == r.accepted
            assert len(restored.motivating_scene_ids) == len(r.motivating_scene_ids)
