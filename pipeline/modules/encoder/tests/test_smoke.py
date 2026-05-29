"""Smoke tests for Module 2: Encoder (reasoning arm only).

All tests run without a real VLM or GCS connection.
Cosmos-Reason2 is bypassed by StubVLMClient — same output type, no network.
Storage is bypassed by a lightweight mock.

Run:
    python -m pytest pipeline/modules/encoder/tests/test_smoke.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_storage(video_url: str = "https://example.com/video.mp4", pose_summary: str | None = "Vehicle traveled 20m forward.") -> Any:
    """Return a mock WindowStorage that returns a fixed URL and pose summary."""
    manifest = MagicMock()
    manifest.pose_summary = pose_summary

    storage = MagicMock()
    storage.get_window_video_url.return_value = video_url
    storage.get_window_manifest.return_value = manifest
    return storage


def _make_window(storage: Any = None, segment_id: str = "seg_001", window_idx: int = 0) -> Any:
    from pipeline.modules.encoder.schema import WindowInput
    return WindowInput(
        segment_id=segment_id,
        window_idx=window_idx,
        storage=storage or _make_mock_storage(),
    )


# ---------------------------------------------------------------------------
# Import checks
# ---------------------------------------------------------------------------

def test_schema_imports() -> None:
    from pipeline.modules.encoder.schema import (
        FAILURE_INVALID_JSON,
        FAILURE_VOCABULARY_VIOLATION,
        FAILURE_VLM_UNAVAILABLE,
        NULL_FIELDS_V1,
        SchemaRecord,
        WindowInput,
    )
    assert FAILURE_INVALID_JSON == "invalid_json"
    assert "agents" in NULL_FIELDS_V1


def test_vocabulary_imports() -> None:
    from pipeline.modules.encoder.vocabulary import (
        DEFAULT_VOCABULARY,
        Vocabulary,
    )
    assert "car" in DEFAULT_VOCABULARY.agents
    assert "night_driving" in DEFAULT_VOCABULARY.conditions
    assert len(DEFAULT_VOCABULARY.conditions) == 10
    assert len(DEFAULT_VOCABULARY.agents) == 12


def test_reasoning_arm_imports() -> None:
    from pipeline.modules.encoder.reasoning_arm import (
        CosmosReason2Client,
        ReasoningArm,
        StubVLMClient,
        VLMClient,
    )
    assert StubVLMClient.model_id == "stub/cosmos-reason2"


def test_encoder_imports() -> None:
    from pipeline.modules.encoder.encoder import Encoder
    assert Encoder.ARM == "reasoning"


# ---------------------------------------------------------------------------
# VLMClient protocol
# ---------------------------------------------------------------------------

def test_stub_satisfies_vlm_protocol() -> None:
    from pipeline.modules.encoder.reasoning_arm import StubVLMClient, VLMClient
    stub = StubVLMClient()
    assert isinstance(stub, VLMClient)


def test_cosmos_client_satisfies_vlm_protocol() -> None:
    from pipeline.modules.encoder.reasoning_arm import CosmosReason2Client, VLMClient
    client = CosmosReason2Client(api_key="test-key")
    assert isinstance(client, VLMClient)


# ---------------------------------------------------------------------------
# StubVLMClient output shape
# ---------------------------------------------------------------------------

def test_stub_returns_valid_json_shape() -> None:
    from pipeline.modules.encoder.reasoning_arm import StubVLMClient, extract_json
    stub = StubVLMClient()
    raw = stub.complete(video_url="https://example.com/video.mp4", prompt="describe this")
    fields = extract_json(raw)
    assert isinstance(fields, dict)
    assert "agents" in fields
    assert "environment" in fields
    assert "road" in fields
    assert "traffic_control" in fields
    assert "ego_task" in fields
    assert "conditions" in fields


def test_stub_output_passes_vocabulary_validation() -> None:
    from pipeline.modules.encoder.reasoning_arm import StubVLMClient, extract_json
    from pipeline.modules.encoder.vocabulary import DEFAULT_VOCABULARY
    stub = StubVLMClient()
    raw = stub.complete(video_url="https://example.com/video.mp4", prompt="describe this")
    fields = extract_json(raw)
    violations = DEFAULT_VOCABULARY.validate_fields(fields)
    assert violations == [], f"StubVLMClient output has vocabulary violations: {violations}"


# ---------------------------------------------------------------------------
# extract_json
# ---------------------------------------------------------------------------

def test_extract_json_direct() -> None:
    from pipeline.modules.encoder.reasoning_arm import extract_json
    raw = '{"agents": ["car"], "ego_task": "cruising"}'
    assert extract_json(raw)["agents"] == ["car"]


def test_extract_json_fenced() -> None:
    from pipeline.modules.encoder.reasoning_arm import extract_json
    raw = 'Some reasoning.\n```json\n{"agents": ["car"]}\n```'
    assert extract_json(raw)["agents"] == ["car"]


def test_extract_json_bare_fence() -> None:
    from pipeline.modules.encoder.reasoning_arm import extract_json
    raw = '```\n{"agents": []}\n```'
    assert extract_json(raw)["agents"] == []


def test_extract_json_think_tags() -> None:
    from pipeline.modules.encoder.reasoning_arm import extract_json
    raw = "<think>I see cars.</think>\n```json\n{\"ego_task\": \"stopping\"}\n```"
    assert extract_json(raw)["ego_task"] == "stopping"


def test_extract_json_raises_on_garbage() -> None:
    from pipeline.modules.encoder.reasoning_arm import extract_json
    with pytest.raises(ValueError, match="No valid JSON"):
        extract_json("This is just prose. No JSON here at all.")


# ---------------------------------------------------------------------------
# Vocabulary validation
# ---------------------------------------------------------------------------

def test_vocabulary_valid_fields() -> None:
    from pipeline.modules.encoder.vocabulary import DEFAULT_VOCABULARY
    fields = {
        "agents": ["car", "pedestrian"],
        "environment": {"weather": "rain", "time_of_day": "night", "lighting_condition": "dim"},
        "road": {"geometry": "intersection", "lane_count": 3},
        "traffic_control": "traffic_light",
        "ego_task": "turning_left",
        "conditions": ["night_driving", "rain"],
    }
    assert DEFAULT_VOCABULARY.validate_fields(fields) == []


def test_vocabulary_rejects_unknown_agent() -> None:
    from pipeline.modules.encoder.vocabulary import DEFAULT_VOCABULARY
    fields = {"agents": ["flying_car"]}
    violations = DEFAULT_VOCABULARY.validate_fields(fields)
    assert any("flying_car" in v for v in violations)


def test_vocabulary_rejects_unknown_condition() -> None:
    from pipeline.modules.encoder.vocabulary import DEFAULT_VOCABULARY
    fields = {"conditions": ["nuclear_explosion"]}
    violations = DEFAULT_VOCABULARY.validate_fields(fields)
    assert any("nuclear_explosion" in v for v in violations)


def test_vocabulary_rejects_bad_lane_count() -> None:
    from pipeline.modules.encoder.vocabulary import DEFAULT_VOCABULARY
    fields = {"road": {"lane_count": 99}}
    violations = DEFAULT_VOCABULARY.validate_fields(fields)
    assert any("lane_count" in v for v in violations)


def test_vocabulary_fill_fraction_full() -> None:
    from pipeline.modules.encoder.vocabulary import DEFAULT_VOCABULARY
    fields = {
        "agents": ["car"],
        "environment": {"weather": "clear", "time_of_day": "day", "lighting_condition": "well_lit"},
        "road": {"geometry": "straight", "lane_count": 2},
        "traffic_control": "none",
        "ego_task": "cruising",
        "conditions": [],
    }
    assert DEFAULT_VOCABULARY.fill_fraction(fields) == 1.0


def test_vocabulary_fill_fraction_empty() -> None:
    from pipeline.modules.encoder.vocabulary import DEFAULT_VOCABULARY
    assert DEFAULT_VOCABULARY.fill_fraction({}) == 0.0


def test_vocabulary_prompt_context_contains_all_tags() -> None:
    from pipeline.modules.encoder.vocabulary import DEFAULT_VOCABULARY
    ctx = DEFAULT_VOCABULARY.prompt_context()
    assert "night_driving" in ctx
    assert "pedestrian" in ctx
    assert "traffic_light" in ctx


# ---------------------------------------------------------------------------
# ReasoningArm with StubVLMClient
# ---------------------------------------------------------------------------

def test_reasoning_arm_annotate_returns_valid_fields() -> None:
    from pipeline.modules.encoder.reasoning_arm import ReasoningArm, StubVLMClient
    from pipeline.modules.encoder.vocabulary import DEFAULT_VOCABULARY
    arm = ReasoningArm(vlm=StubVLMClient(), vocabulary=DEFAULT_VOCABULARY)
    fields, raw = arm.annotate(
        video_url="https://example.com/video.mp4",
        pose_summary="Vehicle traveled 10m forward.",
    )
    assert isinstance(fields, dict)
    assert "agents" in fields
    assert DEFAULT_VOCABULARY.validate_fields(fields) == []


def test_reasoning_arm_annotate_from_storage() -> None:
    from pipeline.modules.encoder.reasoning_arm import ReasoningArm, StubVLMClient
    from pipeline.modules.encoder.vocabulary import DEFAULT_VOCABULARY
    arm = ReasoningArm(vlm=StubVLMClient(), vocabulary=DEFAULT_VOCABULARY)
    storage = _make_mock_storage()
    fields, raw = arm.annotate_from_storage(
        segment_id="seg_001", window_idx=0, storage=storage
    )
    assert fields["ego_task"] == "cruising"
    storage.get_window_video_url.assert_called_once()


def test_reasoning_arm_retries_on_bad_json(capsys: pytest.CaptureFixture) -> None:
    """Arm should retry on malformed JSON; on success return fields."""
    from pipeline.modules.encoder.reasoning_arm import ReasoningArm, VLMClient
    from pipeline.modules.encoder.vocabulary import DEFAULT_VOCABULARY

    good_response = '```json\n{"agents": ["car"], "environment": {"weather": "clear", "time_of_day": "day", "lighting_condition": "well_lit"}, "road": {"geometry": "straight", "lane_count": 2}, "traffic_control": "none", "ego_task": "cruising", "conditions": []}\n```'

    call_count = 0
    class FlakeyClient:
        model_id = "stub/flakey"
        def complete(self, video_url: str, prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                return "this is not json at all"
            return good_response

    arm = ReasoningArm(vlm=FlakeyClient(), vocabulary=DEFAULT_VOCABULARY, max_retries=3)
    fields, _ = arm.annotate(
        video_url="https://example.com/video.mp4", pose_summary=None
    )
    assert fields["ego_task"] == "cruising"
    assert call_count == 2


def test_reasoning_arm_raises_after_max_retries() -> None:
    from pipeline.modules.encoder.reasoning_arm import ReasoningArm
    from pipeline.modules.encoder.vocabulary import DEFAULT_VOCABULARY

    class AlwaysBadClient:
        model_id = "stub/always-bad"
        def complete(self, video_url: str, prompt: str) -> str:
            return "not json"

    arm = ReasoningArm(vlm=AlwaysBadClient(), vocabulary=DEFAULT_VOCABULARY, max_retries=2)
    with pytest.raises(ValueError):
        arm.annotate(video_url="https://example.com/v.mp4", pose_summary=None)


# ---------------------------------------------------------------------------
# Encoder end-to-end with StubVLMClient
# ---------------------------------------------------------------------------

def test_encoder_process_success(tmp_path: Path) -> None:
    from pipeline.modules.encoder.encoder import Encoder
    from pipeline.modules.encoder.reasoning_arm import StubVLMClient
    from pipeline.modules.encoder.vocabulary import DEFAULT_VOCABULARY

    lib = Encoder(
        vlm=StubVLMClient(),
        vocabulary=DEFAULT_VOCABULARY,
        cache_root=tmp_path,
    )
    window = _make_window()
    record = lib.process(window)[0]

    assert record.succeeded
    assert record.failure_mode is None
    assert record.arm == "reasoning"
    assert record.schema_version == "1.0"
    assert "agents" in record.fields
    assert not record.cached


def test_encoder_process_caches_result(tmp_path: Path) -> None:
    from pipeline.modules.encoder.encoder import Encoder
    from pipeline.modules.encoder.reasoning_arm import StubVLMClient
    from pipeline.modules.encoder.vocabulary import DEFAULT_VOCABULARY

    call_count = 0
    class CountingStub(StubVLMClient):
        def complete(self, video_url: str, prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return super().complete(video_url, prompt)

    lib = Encoder(vlm=CountingStub(), vocabulary=DEFAULT_VOCABULARY, cache_root=tmp_path)
    window = _make_window()

    lib.process(window)
    assert call_count == 1

    record2 = lib.process(window)[0]
    assert call_count == 1          # VLM not called again
    assert record2.cached is True


def test_encoder_process_batch(tmp_path: Path) -> None:
    from pipeline.modules.encoder.encoder import Encoder
    from pipeline.modules.encoder.reasoning_arm import StubVLMClient
    from pipeline.modules.encoder.vocabulary import DEFAULT_VOCABULARY

    lib = Encoder(vlm=StubVLMClient(), vocabulary=DEFAULT_VOCABULARY, cache_root=tmp_path)
    windows = [_make_window(window_idx=i) for i in range(3)]
    records = lib.process_batch(windows)

    assert len(records) == 3
    assert all(r.succeeded for r in records)


def test_encoder_records_storage_error(tmp_path: Path) -> None:
    from pipeline.modules.encoder.encoder import Encoder
    from pipeline.modules.encoder.reasoning_arm import StubVLMClient
    from pipeline.modules.encoder.vocabulary import DEFAULT_VOCABULARY
    from pipeline.interfaces.errors import WindowStorageError

    bad_storage = MagicMock()
    bad_storage.get_window_video_url.side_effect = WindowStorageError(
        "seg_001/0000", "blob not found"
    )

    lib = Encoder(vlm=StubVLMClient(), vocabulary=DEFAULT_VOCABULARY, cache_root=tmp_path)
    window = _make_window(storage=bad_storage)
    record = lib.process(window)[0]

    assert not record.succeeded
    assert record.failure_mode == "storage_error"


def test_encoder_records_vlm_unavailable(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    from pipeline.modules.encoder.encoder import Encoder
    from pipeline.modules.encoder.reasoning_arm import VLMUnavailableError
    from pipeline.modules.encoder.vocabulary import DEFAULT_VOCABULARY

    class DeadClient:
        model_id = "stub/dead"
        def complete(self, video_url: str, prompt: str) -> str:
            raise VLMUnavailableError("stub/dead", "connection refused")

    lib = Encoder(vlm=DeadClient(), vocabulary=DEFAULT_VOCABULARY, cache_root=tmp_path)
    record = lib.process(_make_window())[0]

    assert record.failure_mode == "vlm_unavailable"
    captured = capsys.readouterr()
    assert "VLM UNAVAILABLE" in captured.err


# ---------------------------------------------------------------------------
# SchemaRecord serialisation round-trip
# ---------------------------------------------------------------------------

def test_schema_record_round_trip() -> None:
    from pipeline.modules.encoder.encoder import Encoder
    from pipeline.modules.encoder.reasoning_arm import StubVLMClient
    from pipeline.modules.encoder.schema import SchemaRecord
    from pipeline.modules.encoder.vocabulary import DEFAULT_VOCABULARY
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        lib = Encoder(
            vlm=StubVLMClient(),
            vocabulary=DEFAULT_VOCABULARY,
            cache_root=Path(tmp),
        )
        original = lib.process(_make_window())[0]
        d = original.to_json()
        restored = SchemaRecord.from_json(d)

    assert str(restored.window_id) == str(original.window_id)
    assert restored.failure_mode == original.failure_mode
    assert restored.fields == original.fields
