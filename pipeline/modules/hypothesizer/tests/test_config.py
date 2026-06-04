"""Tests for hypothesizer/config.py."""
import pytest
from pipeline.modules.hypothesizer.config import (
    HypothesizerConfig,
    HypothesizerEmptyInputError,
    VocabularyMismatchError,
    SCHEMA_PATH_TO_ATOM_PREFIX,
    MULTI_VALUE_FIELDS,
    SINGLE_CATEGORICAL_FIELDS,
)


class TestHypothesizerConfig:
    def test_defaults(self):
        cfg = HypothesizerConfig()
        assert cfg.min_marginal_frequency == 0.05
        assert cfg.max_joint_frequency == 0.005
        assert cfg.min_pairwise_frequency == 0.01
        assert cfg.composition_sizes == [2, 3, 4]
        assert cfg.top_k == 30
        assert cfg.compose_over is None
        assert cfg.valid_atoms is None

    def test_composition_sizes_are_independent_per_instance(self):
        a = HypothesizerConfig()
        b = HypothesizerConfig()
        a.composition_sizes.append(5)
        assert b.composition_sizes == [2, 3, 4]

    def test_compose_over_conditions_only(self):
        cfg = HypothesizerConfig(compose_over=["conditions"])
        assert cfg.compose_over == ["conditions"]

    def test_valid_atoms_accepted(self):
        atoms = frozenset({"agents:car", "weather:fog"})
        cfg = HypothesizerConfig(valid_atoms=atoms)
        assert "agents:car" in cfg.valid_atoms

    def test_custom_thresholds(self):
        cfg = HypothesizerConfig(
            min_marginal_frequency=0.1,
            max_joint_frequency=0.01,
            min_pairwise_frequency=0.02,
            top_k=10,
        )
        assert cfg.min_marginal_frequency == 0.1
        assert cfg.max_joint_frequency == 0.01
        assert cfg.min_pairwise_frequency == 0.02
        assert cfg.top_k == 10


class TestSchemaPathMapping:
    def test_all_expected_paths_present(self):
        expected_paths = {
            "agents",
            "environment.weather",
            "environment.time_of_day",
            "environment.lighting_condition",
            "road.geometry",
            "traffic_control",
            "ego_task",
            "conditions",
        }
        assert set(SCHEMA_PATH_TO_ATOM_PREFIX.keys()) == expected_paths

    def test_lane_count_excluded(self):
        assert "road.lane_count" not in SCHEMA_PATH_TO_ATOM_PREFIX

    def test_prefix_values(self):
        assert SCHEMA_PATH_TO_ATOM_PREFIX["environment.weather"] == "weather"
        assert SCHEMA_PATH_TO_ATOM_PREFIX["environment.time_of_day"] == "time_of_day"
        assert SCHEMA_PATH_TO_ATOM_PREFIX["environment.lighting_condition"] == "lighting"
        assert SCHEMA_PATH_TO_ATOM_PREFIX["road.geometry"] == "road_geometry"
        assert SCHEMA_PATH_TO_ATOM_PREFIX["agents"] == "agents"
        assert SCHEMA_PATH_TO_ATOM_PREFIX["conditions"] == "conditions"

    def test_multi_and_single_categorical_are_disjoint(self):
        assert MULTI_VALUE_FIELDS.isdisjoint(SINGLE_CATEGORICAL_FIELDS)

    def test_all_prefixes_classified(self):
        all_prefixes = set(SCHEMA_PATH_TO_ATOM_PREFIX.values())
        classified = MULTI_VALUE_FIELDS | SINGLE_CATEGORICAL_FIELDS
        assert all_prefixes == classified


class TestErrors:
    def test_vocabulary_mismatch_error_message(self):
        exc = VocabularyMismatchError("weather:tornado", "seg_abc/0001")
        assert "weather:tornado" in str(exc)
        assert "seg_abc/0001" in str(exc)
        assert exc.atom == "weather:tornado"
        assert exc.window_id == "seg_abc/0001"

    def test_vocabulary_mismatch_is_hypothesizer_error(self):
        exc = VocabularyMismatchError("x:y", "seg/0000")
        assert isinstance(exc, Exception)

    def test_empty_input_error(self):
        exc = HypothesizerEmptyInputError("no records")
        assert isinstance(exc, Exception)
