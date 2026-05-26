"""Smoke tests for Module 3: Hypothesizer (full public interface)."""
import pytest
from pipeline.interfaces.schema_record import SchemaRecord
from pipeline.interfaces.window import WindowKey
from pipeline.interfaces.proposal import CompositionProposal
from pipeline.modules.hypothesizer.hypothesizer import Hypothesizer
from pipeline.modules.hypothesizer.config import HypothesizerConfig, HypothesizerEmptyInputError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    seg: str,
    idx: int,
    fields: dict,
    failure_mode: str | None = None,
) -> SchemaRecord:
    return SchemaRecord(
        window_id=WindowKey(seg, idx),
        arm="reasoning",
        schema_version="1.0",
        prompt_template_id="v1_describe",
        fields=fields,
        failure_mode=failure_mode,
    )


FIELDS_FOG_NIGHT = {
    "agents": ["car"],
    "environment": {"weather": "fog", "time_of_day": "night", "lighting_condition": "unlit"},
    "road": {"geometry": "straight", "lane_count": 2},
    "traffic_control": "none",
    "ego_task": "cruising",
    "conditions": ["night_driving", "fog"],
}

FIELDS_CLEAR_DAY = {
    "agents": ["car", "pedestrian"],
    "environment": {"weather": "clear", "time_of_day": "day", "lighting_condition": "well_lit"},
    "road": {"geometry": "intersection", "lane_count": 4},
    "traffic_control": "traffic_light",
    "ego_task": "turning_left",
    "conditions": [],
}

FIELDS_FOG_DAY = {
    "agents": ["car"],
    "environment": {"weather": "fog", "time_of_day": "day", "lighting_condition": "dim"},
    "road": {"geometry": "curve", "lane_count": 2},
    "traffic_control": "none",
    "ego_task": "cruising",
    "conditions": ["fog"],
}


def _make_dataset(n: int = 50) -> list[SchemaRecord]:
    """50-window dataset: fog-night, clear-day, fog-day variants."""
    records = []
    for i in range(n):
        if i % 10 == 0:
            fields = FIELDS_FOG_NIGHT
        elif i % 5 == 0:
            fields = FIELDS_FOG_DAY
        else:
            fields = FIELDS_CLEAR_DAY
        records.append(_make_record("seg_smoke", i, fields))
    return records


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

class TestHypothesizerSmoke:
    def test_imports_clean(self):
        from pipeline.modules.hypothesizer import Hypothesizer, HypothesizerConfig
        assert Hypothesizer is not None

    def test_propose_returns_list(self):
        hyp = Hypothesizer(HypothesizerConfig(
            min_marginal_frequency=0.0,
            max_joint_frequency=0.5,
            min_pairwise_frequency=0.0,
            composition_sizes=[2],
            top_k=10,
        ))
        records = _make_dataset(50)
        proposals = hyp.propose(records, arm="reasoning")
        assert isinstance(proposals, list)

    def test_proposals_are_composition_proposal_instances(self):
        hyp = Hypothesizer(HypothesizerConfig(
            min_marginal_frequency=0.0,
            max_joint_frequency=0.5,
            min_pairwise_frequency=0.0,
            composition_sizes=[2],
            top_k=10,
        ))
        records = _make_dataset(50)
        proposals = hyp.propose(records, arm="reasoning")
        for p in proposals:
            assert isinstance(p, CompositionProposal)

    def test_output_fields_present(self):
        hyp = Hypothesizer(HypothesizerConfig(
            min_marginal_frequency=0.0,
            max_joint_frequency=0.5,
            min_pairwise_frequency=0.0,
            composition_sizes=[2],
            top_k=5,
        ))
        proposals = hyp.propose(_make_dataset(50), arm="reasoning")
        if proposals:
            p = proposals[0]
            assert isinstance(p.composition_id, str) and len(p.composition_id) == 16
            assert isinstance(p.constituents, list) and len(p.constituents) >= 2
            assert isinstance(p.marginal_frequencies, dict)
            assert isinstance(p.pairwise_frequencies, dict)
            assert isinstance(p.expected_joint, float)
            assert isinstance(p.observed_joint, float)
            assert isinstance(p.novelty_score, float)
            assert isinstance(p.motivating_scene_ids, list)
            assert p.arm == "reasoning"

    def test_failure_records_skipped(self):
        records = _make_dataset(40)
        bad = _make_record("seg_smoke", 99, {}, failure_mode="invalid_json")
        records.append(bad)
        hyp = Hypothesizer(HypothesizerConfig(
            min_marginal_frequency=0.0,
            max_joint_frequency=0.5,
            min_pairwise_frequency=0.0,
            composition_sizes=[2],
            top_k=5,
        ))
        proposals = hyp.propose(records, arm="reasoning")
        # Should not crash; bad record was skipped
        assert isinstance(proposals, list)

    def test_all_failure_raises_empty_input_error(self):
        records = [_make_record("seg", i, {}, failure_mode="vlm_unavailable") for i in range(10)]
        hyp = Hypothesizer()
        with pytest.raises(HypothesizerEmptyInputError):
            hyp.propose(records, arm="reasoning")

    def test_empty_list_raises(self):
        with pytest.raises(HypothesizerEmptyInputError):
            Hypothesizer().propose([], arm="reasoning")

    def test_top_k_limit(self):
        hyp = Hypothesizer(HypothesizerConfig(
            min_marginal_frequency=0.0,
            max_joint_frequency=0.9,
            min_pairwise_frequency=0.0,
            composition_sizes=[2, 3],
            top_k=3,
        ))
        proposals = hyp.propose(_make_dataset(50), arm="reasoning")
        assert len(proposals) <= 3

    def test_sorted_by_novelty_desc(self):
        hyp = Hypothesizer(HypothesizerConfig(
            min_marginal_frequency=0.0,
            max_joint_frequency=0.5,
            min_pairwise_frequency=0.0,
            composition_sizes=[2],
            top_k=20,
        ))
        proposals = hyp.propose(_make_dataset(50), arm="reasoning")
        scores = [p.novelty_score for p in proposals]
        assert scores == sorted(scores, reverse=True)

    def test_deterministic_across_calls(self):
        records = _make_dataset(50)
        hyp = Hypothesizer(HypothesizerConfig(
            min_marginal_frequency=0.0,
            max_joint_frequency=0.5,
            min_pairwise_frequency=0.0,
            composition_sizes=[2],
            top_k=10,
        ))
        p1 = hyp.propose(records, arm="reasoning")
        p2 = hyp.propose(records, arm="reasoning")
        assert [p.composition_id for p in p1] == [p.composition_id for p in p2]

    def test_compose_over_conditions_only(self):
        records = _make_dataset(50)
        hyp = Hypothesizer(HypothesizerConfig(
            compose_over=["conditions"],
            min_marginal_frequency=0.0,
            max_joint_frequency=0.5,
            min_pairwise_frequency=0.0,
            composition_sizes=[2],
            top_k=10,
        ))
        proposals = hyp.propose(records, arm="reasoning")
        for p in proposals:
            for atom in p.constituents:
                assert atom.startswith("conditions:")

    def test_json_serializable(self):
        import json
        hyp = Hypothesizer(HypothesizerConfig(
            min_marginal_frequency=0.0,
            max_joint_frequency=0.5,
            min_pairwise_frequency=0.0,
            composition_sizes=[2],
            top_k=5,
        ))
        proposals = hyp.propose(_make_dataset(50), arm="reasoning")
        for p in proposals:
            serialized = p.to_json()
            restored = CompositionProposal.from_json(serialized)
            assert restored.composition_id == p.composition_id
            assert restored.constituents == p.constituents

    def test_motivating_scenes_are_window_keys(self):
        hyp = Hypothesizer(HypothesizerConfig(
            min_marginal_frequency=0.0,
            max_joint_frequency=0.5,
            min_pairwise_frequency=0.0,
            composition_sizes=[2],
            top_k=5,
        ))
        proposals = hyp.propose(_make_dataset(50), arm="reasoning")
        from pipeline.interfaces.window import WindowKey
        for p in proposals:
            for scene_id in p.motivating_scene_ids:
                assert isinstance(scene_id, WindowKey)
