"""Module 3: Hypothesizer — output contract tests.

Validates every field declared in the README Output Contract section for
CompositionProposal. Each test asserts a specific contract requirement,
not implementation details.
"""
import pytest
from pipeline.interfaces.proposal import CompositionProposal
from pipeline.interfaces.schema_record import SchemaRecord
from pipeline.interfaces.window import WindowKey
from pipeline.modules.hypothesizer.hypothesizer import Hypothesizer
from pipeline.modules.hypothesizer.config import HypothesizerConfig


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

def _make_record(seg: str, idx: int, fields: dict, failure_mode=None) -> SchemaRecord:
    return SchemaRecord(
        window_id=WindowKey(seg, idx),
        arm="reasoning",
        schema_version="1.0",
        prompt_template_id="v1_describe",
        fields=fields,
        failure_mode=failure_mode,
    )


def _make_records(n: int = 30) -> list[SchemaRecord]:
    base_fields = {
        "agents": ["car"],
        "environment": {"weather": "fog", "time_of_day": "night", "lighting_condition": "unlit"},
        "road": {"geometry": "straight", "lane_count": 2},
        "traffic_control": "none",
        "ego_task": "cruising",
        "conditions": ["night_driving"],
    }
    alt_fields = {
        "agents": ["car", "pedestrian"],
        "environment": {"weather": "clear", "time_of_day": "day", "lighting_condition": "well_lit"},
        "road": {"geometry": "intersection", "lane_count": 4},
        "traffic_control": "traffic_light",
        "ego_task": "turning_left",
        "conditions": [],
    }
    records = []
    for i in range(n):
        fields = base_fields if i % 5 != 0 else alt_fields
        records.append(_make_record("seg_contract", i, fields))
    return records


_CFG = HypothesizerConfig(
    min_marginal_frequency=0.0,
    max_joint_frequency=0.8,
    min_pairwise_frequency=0.0,
    composition_sizes=[2],
    top_k=10,
)

_hyp = Hypothesizer(_CFG)
_records = _make_records(30)
_proposals = _hyp.propose(_records, arm="reasoning")


# ---------------------------------------------------------------------------
# Contract tests (README: Module 3 — Output Contract)
# ---------------------------------------------------------------------------

def test_output_is_list_of_composition_proposals():
    assert isinstance(_proposals, list)
    for p in _proposals:
        assert isinstance(p, CompositionProposal)


def test_composition_id_is_16_char_hex():
    for p in _proposals:
        assert isinstance(p.composition_id, str)
        assert len(p.composition_id) == 16
        int(p.composition_id, 16)  # must be valid hex


def test_constituents_is_list_of_qualified_atoms():
    for p in _proposals:
        assert isinstance(p.constituents, list)
        assert len(p.constituents) >= 2
        for atom in p.constituents:
            assert isinstance(atom, str)
            assert ":" in atom, f"Atom {atom!r} must be 'prefix:value' format"


def test_marginal_frequencies_keys_match_constituents():
    for p in _proposals:
        for atom in p.constituents:
            assert atom in p.marginal_frequencies, \
                f"marginal_frequencies missing atom {atom!r}"
        for key in p.marginal_frequencies:
            assert isinstance(p.marginal_frequencies[key], float)
            assert 0.0 <= p.marginal_frequencies[key] <= 1.0


def test_pairwise_frequencies_cover_all_pairs():
    for p in _proposals:
        atoms = sorted(p.constituents)
        for i, a in enumerate(atoms):
            for b in atoms[i + 1:]:
                key = f"{a}|{b}"
                assert key in p.pairwise_frequencies, \
                    f"pairwise_frequencies missing pair {key!r}"
        for val in p.pairwise_frequencies.values():
            assert isinstance(val, float)
            assert 0.0 <= val <= 1.0


def test_expected_joint_is_positive_float():
    for p in _proposals:
        assert isinstance(p.expected_joint, float)
        assert p.expected_joint > 0.0


def test_observed_joint_is_non_negative_float():
    for p in _proposals:
        assert isinstance(p.observed_joint, float)
        assert p.observed_joint >= 0.0


def test_novelty_score_is_float():
    for p in _proposals:
        assert isinstance(p.novelty_score, float)


def test_motivating_scene_ids_are_window_keys():
    for p in _proposals:
        assert isinstance(p.motivating_scene_ids, list)
        for wk in p.motivating_scene_ids:
            assert isinstance(wk, WindowKey)


def test_arm_is_string():
    for p in _proposals:
        assert isinstance(p.arm, str)
        assert p.arm == "reasoning"


def test_composition_id_is_deterministic():
    """Same constituents always produce the same composition_id."""
    from pipeline.modules.hypothesizer.composition import composition_id
    a = composition_id(["b:y", "a:x"])
    b = composition_id(["a:x", "b:y"])
    assert a == b


def test_no_mutual_exclusivity_violations():
    """No proposal may contain two atoms from the same single-categorical field."""
    from pipeline.modules.hypothesizer.config import SINGLE_CATEGORICAL_FIELDS
    for p in _proposals:
        seen: dict[str, str] = {}
        for atom in p.constituents:
            prefix = atom.split(":", 1)[0]
            if prefix in SINGLE_CATEGORICAL_FIELDS:
                assert prefix not in seen, \
                    f"Mutual exclusivity violation: {seen[prefix]} and {atom} in proposal {p.composition_id}"
                seen[prefix] = atom


def test_json_round_trip():
    for p in _proposals:
        serialized = p.to_json()
        restored = CompositionProposal.from_json(serialized)
        assert restored.composition_id == p.composition_id
        assert restored.constituents == p.constituents
        assert restored.arm == p.arm
        assert len(restored.motivating_scene_ids) == len(p.motivating_scene_ids)


def test_sorted_by_novelty_score_desc():
    scores = [p.novelty_score for p in _proposals]
    assert scores == sorted(scores, reverse=True)


def test_side_effect_free():
    """propose() on the same records produces the same result (no mutation)."""
    p1 = _hyp.propose(_records, arm="reasoning")
    p2 = _hyp.propose(_records, arm="reasoning")
    assert [p.composition_id for p in p1] == [p.composition_id for p in p2]
