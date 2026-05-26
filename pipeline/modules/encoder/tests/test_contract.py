"""Contract tests for Module 2: Encoder (reasoning arm).

Validates that every field in the README's Output Contract section is present,
correctly typed, and that all documented side effects occur.

README Output Contract reference (SchemaRecord v1.0):
  window_id: WindowKey
  arm: Literal["reasoning", "visual"]
  schema_version: str
  prompt_template_id: str | None
  fields:
    agents: list[str]                # from agent vocabulary (12 tags)
    environment: dict
      weather: str                   # from weather vocabulary
      time_of_day: str               # from time_of_day vocabulary
      lighting_condition: str        # from lighting_condition vocabulary
    road: dict
      geometry: str                  # from road_geometry vocabulary
      lane_count: int                # 1–8
    traffic_control: str             # from traffic_control vocabulary
    ego_task: str                    # from ego_task vocabulary
    conditions: list[str]            # from conditions vocabulary (10 tags)
  failure_mode: str | None           # None = success
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from pipeline.interfaces.schema_record import SchemaRecord
from pipeline.interfaces.window import WindowKey
from pipeline.modules.encoder.vocabulary import DEFAULT_VOCABULARY


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_storage(pose_summary: str | None = "Vehicle traveled 15m.") -> Any:
    manifest = MagicMock()
    manifest.pose_summary = pose_summary
    storage = MagicMock()
    storage.get_window_video_url.return_value = "https://example.com/video.mp4"
    storage.get_window_manifest.return_value = manifest
    return storage


def _make_encoder(tmp_path: Path):
    from pipeline.modules.encoder.encoder import Encoder
    from pipeline.modules.encoder.reasoning_arm import StubVLMClient
    return Encoder(vlm=StubVLMClient(), vocabulary=DEFAULT_VOCABULARY, cache_root=tmp_path)


def _make_window(segment_id: str = "contract_seg", window_idx: int = 0, storage: Any = None):
    from pipeline.modules.encoder.schema import WindowInput
    return WindowInput(
        segment_id=segment_id,
        window_idx=window_idx,
        storage=storage or _make_storage(),
    )


# ---------------------------------------------------------------------------
# Step 2.1: Output is a SchemaRecord from interfaces/
# ---------------------------------------------------------------------------

def test_output_is_interfaces_schema_record(tmp_path: Path) -> None:
    enc = _make_encoder(tmp_path)
    record = enc.process(_make_window())
    assert isinstance(record, SchemaRecord), (
        f"Output must be pipeline.interfaces.schema_record.SchemaRecord, "
        f"got {type(record).__module__}.{type(record).__name__}"
    )


# ---------------------------------------------------------------------------
# Step 2.2: window_id is a WindowKey with correct segment_id and window_idx
# ---------------------------------------------------------------------------

def test_window_id_field(tmp_path: Path) -> None:
    enc = _make_encoder(tmp_path)
    record = enc.process(_make_window(segment_id="seg_abc", window_idx=7))
    assert isinstance(record.window_id, WindowKey), (
        f"window_id must be WindowKey, got {type(record.window_id)}"
    )
    assert record.window_id.segment_id == "seg_abc"
    assert record.window_id.window_idx == 7
    assert str(record.window_id) == "seg_abc/0007"


# ---------------------------------------------------------------------------
# Step 2.3: arm field
# ---------------------------------------------------------------------------

def test_arm_field(tmp_path: Path) -> None:
    enc = _make_encoder(tmp_path)
    record = enc.process(_make_window())
    assert record.arm in ("reasoning", "visual"), f"arm must be reasoning|visual, got {record.arm!r}"
    assert record.arm == "reasoning"


# ---------------------------------------------------------------------------
# Step 2.4: schema_version matches CURRENT_SCHEMA_VERSION
# ---------------------------------------------------------------------------

def test_schema_version_field(tmp_path: Path) -> None:
    from pipeline.modules.encoder.schema import CURRENT_SCHEMA_VERSION
    enc = _make_encoder(tmp_path)
    record = enc.process(_make_window())
    assert isinstance(record.schema_version, str)
    assert record.schema_version == CURRENT_SCHEMA_VERSION, (
        f"schema_version mismatch: got {record.schema_version!r}, "
        f"expected {CURRENT_SCHEMA_VERSION!r}"
    )


# ---------------------------------------------------------------------------
# Step 2.5: prompt_template_id is str | None
# ---------------------------------------------------------------------------

def test_prompt_template_id_field(tmp_path: Path) -> None:
    enc = _make_encoder(tmp_path)
    record = enc.process(_make_window())
    assert record.prompt_template_id is None or isinstance(record.prompt_template_id, str)


# ---------------------------------------------------------------------------
# Step 2.6: failure_mode is None on success
# ---------------------------------------------------------------------------

def test_failure_mode_none_on_success(tmp_path: Path) -> None:
    enc = _make_encoder(tmp_path)
    record = enc.process(_make_window())
    assert record.failure_mode is None, (
        f"failure_mode must be None for a successful record, got {record.failure_mode!r}"
    )
    assert record.succeeded is True


# ---------------------------------------------------------------------------
# Step 2.7: fields — all required top-level keys present on success
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = ["agents", "environment", "road", "traffic_control", "ego_task", "conditions"]

def test_fields_all_required_keys_present(tmp_path: Path) -> None:
    enc = _make_encoder(tmp_path)
    record = enc.process(_make_window())
    assert record.failure_mode is None, "Test requires a successful record"
    for key in REQUIRED_FIELDS:
        assert key in record.fields, f"Required field {key!r} missing from record.fields"


# ---------------------------------------------------------------------------
# Step 2.8: agents — list[str] drawn from vocabulary
# ---------------------------------------------------------------------------

def test_agents_field_type_and_vocabulary(tmp_path: Path) -> None:
    enc = _make_encoder(tmp_path)
    record = enc.process(_make_window())
    agents = record.fields.get("agents")
    assert isinstance(agents, list), f"agents must be list, got {type(agents)}"
    for tag in agents:
        assert isinstance(tag, str), f"each agent must be str, got {type(tag)}"
        assert tag in DEFAULT_VOCABULARY.agents, (
            f"agent tag {tag!r} not in vocabulary {sorted(DEFAULT_VOCABULARY.agents)}"
        )


# ---------------------------------------------------------------------------
# Step 2.9: environment — required sub-keys, vocabulary-checked
# ---------------------------------------------------------------------------

def test_environment_field_shape(tmp_path: Path) -> None:
    enc = _make_encoder(tmp_path)
    record = enc.process(_make_window())
    env = record.fields.get("environment")
    assert isinstance(env, dict), f"environment must be dict, got {type(env)}"
    for subkey in ("weather", "time_of_day", "lighting_condition"):
        assert subkey in env, f"environment.{subkey} missing"
    if env.get("weather") is not None:
        assert env["weather"] in DEFAULT_VOCABULARY.weather
    if env.get("time_of_day") is not None:
        assert env["time_of_day"] in DEFAULT_VOCABULARY.time_of_day
    if env.get("lighting_condition") is not None:
        assert env["lighting_condition"] in DEFAULT_VOCABULARY.lighting_condition


# ---------------------------------------------------------------------------
# Step 2.10: road — geometry and lane_count
# ---------------------------------------------------------------------------

def test_road_field_shape(tmp_path: Path) -> None:
    from pipeline.modules.encoder.vocabulary import LANE_COUNT_MAX, LANE_COUNT_MIN
    enc = _make_encoder(tmp_path)
    record = enc.process(_make_window())
    road = record.fields.get("road")
    assert isinstance(road, dict), f"road must be dict, got {type(road)}"
    assert "geometry" in road
    assert "lane_count" in road
    if road.get("geometry") is not None:
        assert road["geometry"] in DEFAULT_VOCABULARY.road_geometry
    if road.get("lane_count") is not None:
        assert isinstance(road["lane_count"], int)
        assert LANE_COUNT_MIN <= road["lane_count"] <= LANE_COUNT_MAX


# ---------------------------------------------------------------------------
# Step 2.11: traffic_control — vocabulary-checked
# ---------------------------------------------------------------------------

def test_traffic_control_field(tmp_path: Path) -> None:
    enc = _make_encoder(tmp_path)
    record = enc.process(_make_window())
    tc = record.fields.get("traffic_control")
    if tc is not None:
        assert tc in DEFAULT_VOCABULARY.traffic_control, (
            f"traffic_control={tc!r} not in vocabulary"
        )


# ---------------------------------------------------------------------------
# Step 2.12: ego_task — vocabulary-checked
# ---------------------------------------------------------------------------

def test_ego_task_field(tmp_path: Path) -> None:
    enc = _make_encoder(tmp_path)
    record = enc.process(_make_window())
    et = record.fields.get("ego_task")
    if et is not None:
        assert et in DEFAULT_VOCABULARY.ego_task, (
            f"ego_task={et!r} not in vocabulary"
        )


# ---------------------------------------------------------------------------
# Step 2.13: conditions — list[str], vocabulary-checked
# ---------------------------------------------------------------------------

def test_conditions_field_type_and_vocabulary(tmp_path: Path) -> None:
    enc = _make_encoder(tmp_path)
    record = enc.process(_make_window())
    conds = record.fields.get("conditions")
    assert isinstance(conds, list), f"conditions must be list, got {type(conds)}"
    for tag in conds:
        assert isinstance(tag, str)
        assert tag in DEFAULT_VOCABULARY.conditions, (
            f"condition {tag!r} not in vocabulary"
        )


# ---------------------------------------------------------------------------
# Step 2.14: Side effects — cache file written
# ---------------------------------------------------------------------------

def test_side_effect_cache_written(tmp_path: Path) -> None:
    enc = _make_encoder(tmp_path)
    window = _make_window(segment_id="cache_seg", window_idx=3)
    record = enc.process(window)
    cache_dir = tmp_path / "encoder" / "reasoning"
    cache_files = list(cache_dir.glob("*.json"))
    assert len(cache_files) == 1, (
        f"Expected exactly 1 cache file after process(), found {len(cache_files)}"
    )
    cached_data = json.loads(cache_files[0].read_text())
    assert "window_id" in cached_data
    assert "fields" in cached_data
    assert "failure_mode" in cached_data


# ---------------------------------------------------------------------------
# Step 2.15: Failure records — fields null, failure_mode set
# ---------------------------------------------------------------------------

def test_failure_record_shape_on_vlm_failure(tmp_path: Path) -> None:
    from pipeline.modules.encoder.reasoning_arm import VLMUnavailableError
    from pipeline.modules.encoder.schema import FAILURE_VLM_UNAVAILABLE

    class DeadVLM:
        model_id = "stub/dead"
        def complete(self, video_url: str, prompt: str) -> str:
            raise VLMUnavailableError("stub/dead", "connection refused")

    enc = _make_encoder.__wrapped__ if hasattr(_make_encoder, "__wrapped__") else None
    from pipeline.modules.encoder.encoder import Encoder
    enc = Encoder(vlm=DeadVLM(), vocabulary=DEFAULT_VOCABULARY, cache_root=tmp_path)
    record = enc.process(_make_window())

    assert isinstance(record, SchemaRecord)
    assert record.failure_mode == FAILURE_VLM_UNAVAILABLE
    assert record.succeeded is False
    assert record.fields.get("agents") is None, "agents must be None on failure"
    assert record.fields.get("ego_task") is None, "ego_task must be None on failure"


# ---------------------------------------------------------------------------
# Step 2.16: SchemaRecord serialisation round-trip via to_json/from_json
# ---------------------------------------------------------------------------

def test_schema_record_round_trip_via_interfaces(tmp_path: Path) -> None:
    enc = _make_encoder(tmp_path)
    original = enc.process(_make_window())
    serialised = original.to_json()
    restored = SchemaRecord.from_json(serialised)

    assert str(restored.window_id) == str(original.window_id)
    assert restored.arm == original.arm
    assert restored.schema_version == original.schema_version
    assert restored.failure_mode == original.failure_mode
    assert restored.fields == original.fields
