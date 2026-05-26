"""Contract tests for Module 2: Encoder.

Verifies that Encoder.process() outputs conform exactly to the SchemaRecord
type declared in pipeline/interfaces/schema_record.py. Every field is asserted.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_storage():
    manifest = MagicMock()
    manifest.pose_summary = "Vehicle traveled 20m forward."
    storage = MagicMock()
    storage.get_window_video_url.return_value = "https://example.com/video.mp4"
    storage.get_window_manifest.return_value = manifest
    return storage


def _make_encoder(tmp_path: Path):
    from pipeline.modules.encoder.encoder import Encoder
    from pipeline.modules.encoder.reasoning_arm import StubVLMClient
    from pipeline.modules.encoder.vocabulary import DEFAULT_VOCABULARY
    return Encoder(vlm=StubVLMClient(), vocabulary=DEFAULT_VOCABULARY, cache_root=tmp_path)


# ---------------------------------------------------------------------------
# Type contract
# ---------------------------------------------------------------------------

def test_process_returns_schema_record_type() -> None:
    from pipeline.interfaces.schema_record import SchemaRecord
    from pipeline.modules.encoder.schema import WindowInput
    with tempfile.TemporaryDirectory() as tmp:
        enc = _make_encoder(Path(tmp))
        result = enc.process(WindowInput(segment_id="seg_001", window_idx=0, storage=_make_storage()))
    assert isinstance(result, SchemaRecord)


def test_schema_record_all_fields_present_on_success() -> None:
    from pipeline.interfaces.window import WindowKey
    from pipeline.modules.encoder.schema import WindowInput
    with tempfile.TemporaryDirectory() as tmp:
        enc = _make_encoder(Path(tmp))
        r = enc.process(WindowInput(segment_id="seg_001", window_idx=0, storage=_make_storage()))

    # window_id
    assert isinstance(r.window_id, WindowKey)
    assert r.window_id.segment_id == "seg_001"
    assert r.window_id.window_idx == 0

    # arm
    assert r.arm == "reasoning"

    # schema_version
    assert isinstance(r.schema_version, str) and r.schema_version

    # prompt_template_id
    assert r.prompt_template_id is None or isinstance(r.prompt_template_id, str)

    # fields — all 6 top-level keys must be present on success
    assert isinstance(r.fields, dict)
    for key in ("agents", "environment", "road", "traffic_control", "ego_task", "conditions"):
        assert key in r.fields, f"Missing field in SchemaRecord.fields: {key!r}"

    # failure_mode
    assert r.failure_mode is None

    # succeeded property
    assert r.succeeded is True

    # cached
    assert isinstance(r.cached, bool)

    # created_at
    assert isinstance(r.created_at, str) and "T" in r.created_at


def test_schema_record_fields_types_on_success() -> None:
    from pipeline.modules.encoder.schema import WindowInput
    with tempfile.TemporaryDirectory() as tmp:
        enc = _make_encoder(Path(tmp))
        r = enc.process(WindowInput(segment_id="seg_001", window_idx=0, storage=_make_storage()))

    assert isinstance(r.fields["agents"], list)
    assert isinstance(r.fields["environment"], dict)
    assert "weather" in r.fields["environment"]
    assert "time_of_day" in r.fields["environment"]
    assert "lighting_condition" in r.fields["environment"]
    assert isinstance(r.fields["road"], dict)
    assert "geometry" in r.fields["road"]
    assert "lane_count" in r.fields["road"]
    assert isinstance(r.fields["ego_task"], str)
    assert isinstance(r.fields["conditions"], list)


def test_schema_record_failure_mode_on_failure() -> None:
    from pipeline.modules.encoder.encoder import Encoder
    from pipeline.modules.encoder.vocabulary import DEFAULT_VOCABULARY
    from pipeline.modules.encoder.schema import WindowInput

    class AlwaysBadVLM:
        model_id = "stub/bad"
        def complete(self, video_url: str, prompt: str) -> str:
            return "not json at all"

    with tempfile.TemporaryDirectory() as tmp:
        enc = Encoder(vlm=AlwaysBadVLM(), vocabulary=DEFAULT_VOCABULARY, cache_root=tmp, max_retries=1)
        r = enc.process(WindowInput(segment_id="seg_fail", window_idx=0, storage=_make_storage()))

    assert r.failure_mode is not None
    assert isinstance(r.failure_mode, str)
    assert not r.succeeded


# ---------------------------------------------------------------------------
# Serialization contract
# ---------------------------------------------------------------------------

def test_schema_record_to_json_has_all_fields() -> None:
    from pipeline.modules.encoder.schema import WindowInput
    with tempfile.TemporaryDirectory() as tmp:
        enc = _make_encoder(Path(tmp))
        r = enc.process(WindowInput(segment_id="seg_001", window_idx=0, storage=_make_storage()))

    d = r.to_json()
    required = ["window_id", "arm", "schema_version", "prompt_template_id",
                "fields", "failure_mode", "cached", "created_at"]
    for f in required:
        assert f in d, f"Missing key in SchemaRecord.to_json(): {f!r}"

    # window_id must be a dict (not a string) — downstream modules parse it as dict
    assert isinstance(d["window_id"], dict)
    assert "segment_id" in d["window_id"]
    assert "window_idx" in d["window_id"]

    # must be JSON-serializable
    json.dumps(d)


def test_schema_record_round_trip_via_interfaces() -> None:
    from pipeline.interfaces.schema_record import SchemaRecord
    from pipeline.modules.encoder.schema import WindowInput
    with tempfile.TemporaryDirectory() as tmp:
        enc = _make_encoder(Path(tmp))
        original = enc.process(WindowInput(segment_id="seg_001", window_idx=0, storage=_make_storage()))

    restored = SchemaRecord.from_json(original.to_json())
    assert str(restored.window_id) == str(original.window_id)
    assert restored.arm == original.arm
    assert restored.schema_version == original.schema_version
    assert restored.fields == original.fields
    assert restored.failure_mode == original.failure_mode


# ---------------------------------------------------------------------------
# Vocabulary contract
# ---------------------------------------------------------------------------

def test_successful_record_passes_vocabulary_validation() -> None:
    from pipeline.modules.encoder.vocabulary import DEFAULT_VOCABULARY
    from pipeline.modules.encoder.schema import WindowInput
    with tempfile.TemporaryDirectory() as tmp:
        enc = _make_encoder(Path(tmp))
        r = enc.process(WindowInput(segment_id="seg_001", window_idx=0, storage=_make_storage()))

    violations = DEFAULT_VOCABULARY.validate_fields(r.fields)
    assert violations == [], f"Vocabulary violations in output: {violations}"
