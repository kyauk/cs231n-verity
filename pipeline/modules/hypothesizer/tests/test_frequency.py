"""Tests for hypothesizer/frequency.py."""
import pytest
from pipeline.modules.hypothesizer.frequency import extract_atoms, compute_frequencies
from pipeline.modules.hypothesizer.config import VocabularyMismatchError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FULL_FIELDS = {
    "agents": ["car", "pedestrian"],
    "environment": {
        "weather": "clear",
        "time_of_day": "day",
        "lighting_condition": "well_lit",
    },
    "road": {
        "geometry": "intersection",
        "lane_count": 4,
    },
    "traffic_control": "traffic_light",
    "ego_task": "cruising",
    "conditions": ["night_driving"],
}


class TestExtractAtoms:
    def test_all_fields_produce_atoms(self):
        atoms = extract_atoms(FULL_FIELDS, compose_over=None, valid_atoms=None, window_id="s/0000")
        assert "agents:car" in atoms
        assert "agents:pedestrian" in atoms
        assert "weather:clear" in atoms
        assert "time_of_day:day" in atoms
        assert "lighting:well_lit" in atoms
        assert "road_geometry:intersection" in atoms
        assert "traffic_control:traffic_light" in atoms
        assert "ego_task:cruising" in atoms
        assert "conditions:night_driving" in atoms

    def test_lane_count_excluded(self):
        atoms = extract_atoms(FULL_FIELDS, compose_over=None, valid_atoms=None, window_id="s/0000")
        assert not any("lane_count" in a for a in atoms)

    def test_compose_over_filters(self):
        atoms = extract_atoms(FULL_FIELDS, compose_over=["conditions"], valid_atoms=None, window_id="s/0000")
        assert atoms == frozenset({"conditions:night_driving"})

    def test_compose_over_multi_prefix(self):
        atoms = extract_atoms(FULL_FIELDS, compose_over=["weather", "ego_task"], valid_atoms=None, window_id="s/0000")
        assert atoms == frozenset({"weather:clear", "ego_task:cruising"})

    def test_null_field_produces_no_atom(self):
        fields = {**FULL_FIELDS, "traffic_control": None}
        atoms = extract_atoms(fields, compose_over=None, valid_atoms=None, window_id="s/0000")
        assert not any("traffic_control" in a for a in atoms)

    def test_missing_nested_field_skipped(self):
        fields = {"agents": ["car"], "road": {"lane_count": 2}}  # no geometry
        atoms = extract_atoms(fields, compose_over=None, valid_atoms=None, window_id="s/0000")
        assert not any("road_geometry" in a for a in atoms)

    def test_empty_conditions_list_produces_no_atoms(self):
        fields = {**FULL_FIELDS, "conditions": []}
        atoms = extract_atoms(fields, compose_over=None, valid_atoms=None, window_id="s/0000")
        assert not any("conditions" in a for a in atoms)

    def test_valid_atoms_accepts_correct(self):
        valid = frozenset({"agents:car", "agents:pedestrian", "weather:clear",
                           "time_of_day:day", "lighting:well_lit", "road_geometry:intersection",
                           "traffic_control:traffic_light", "ego_task:cruising",
                           "conditions:night_driving"})
        atoms = extract_atoms(FULL_FIELDS, compose_over=None, valid_atoms=valid, window_id="s/0000")
        assert "agents:car" in atoms

    def test_valid_atoms_rejects_unknown(self):
        valid = frozenset({"agents:car"})  # missing pedestrian
        with pytest.raises(VocabularyMismatchError) as exc_info:
            extract_atoms(FULL_FIELDS, compose_over=None, valid_atoms=valid, window_id="s/0001")
        assert exc_info.value.window_id == "s/0001"

    def test_returns_frozenset(self):
        atoms = extract_atoms(FULL_FIELDS, compose_over=None, valid_atoms=None, window_id="s/0000")
        assert isinstance(atoms, frozenset)

    def test_non_string_scalar_skipped(self):
        fields = {**FULL_FIELDS, "ego_task": 42}
        atoms = extract_atoms(fields, compose_over=None, valid_atoms=None, window_id="s/0000")
        assert not any("ego_task" in a for a in atoms)


class TestComputeFrequencies:
    def test_single_window(self):
        atom_sets = [frozenset({"weather:fog", "ego_task:turning_left"})]
        marginal, pairwise = compute_frequencies(atom_sets)
        assert marginal["weather:fog"] == pytest.approx(1.0)
        assert marginal["ego_task:turning_left"] == pytest.approx(1.0)
        assert pairwise["ego_task:turning_left|weather:fog"] == pytest.approx(1.0)

    def test_pairwise_key_sorted(self):
        atom_sets = [frozenset({"z:zzz", "a:aaa"})]
        _, pairwise = compute_frequencies(atom_sets)
        assert "a:aaa|z:zzz" in pairwise
        assert "z:zzz|a:aaa" not in pairwise

    def test_marginal_frequency(self):
        atom_sets = [
            frozenset({"weather:fog"}),
            frozenset({"weather:fog"}),
            frozenset({"weather:clear"}),
            frozenset({"weather:clear"}),
        ]
        marginal, _ = compute_frequencies(atom_sets)
        assert marginal["weather:fog"] == pytest.approx(0.5)
        assert marginal["weather:clear"] == pytest.approx(0.5)

    def test_pairwise_frequency(self):
        atom_sets = [
            frozenset({"a:x", "b:y"}),
            frozenset({"a:x", "b:y"}),
            frozenset({"a:x"}),
        ]
        marginal, pairwise = compute_frequencies(atom_sets)
        assert marginal["a:x"] == pytest.approx(1.0)
        assert marginal["b:y"] == pytest.approx(2 / 3)
        assert pairwise["a:x|b:y"] == pytest.approx(2 / 3)

    def test_empty_input(self):
        marginal, pairwise = compute_frequencies([])
        assert marginal == {}
        assert pairwise == {}

    def test_single_atom_no_pairwise(self):
        atom_sets = [frozenset({"weather:fog"})]
        _, pairwise = compute_frequencies(atom_sets)
        assert pairwise == {}

    def test_multi_value_field_pairwise(self):
        # Two agents co-occurring → pairwise key for same-prefix atoms
        atom_sets = [frozenset({"agents:car", "agents:pedestrian"})]
        marginal, pairwise = compute_frequencies(atom_sets)
        assert "agents:car|agents:pedestrian" in pairwise
        assert pairwise["agents:car|agents:pedestrian"] == pytest.approx(1.0)

    def test_frequencies_are_floats(self):
        atom_sets = [frozenset({"weather:clear"})]
        marginal, pairwise = compute_frequencies(atom_sets)
        for v in marginal.values():
            assert isinstance(v, float)
