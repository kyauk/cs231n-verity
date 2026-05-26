"""Integration test: Module 1 (Storage) → Module 2 (Encoder) boundary.

Verifies that the output of WindowStorage is directly consumable by Encoder
without any transformation. This is the load-bearing boundary between modules.

No real GCS or VLM connections are used — both are mocked at the minimum level.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gcs_backed_storage(manifest_dict: dict):
    """Build a WindowStorage instance with GCS reads mocked."""
    from pipeline.modules.storage.client import WindowStorage
    ws = WindowStorage.__new__(WindowStorage)
    ws._bucket_name = "test-bucket"
    ws._prefix = "verity"
    ws._creds = None
    ws._project = None
    ws._sign_as = None
    ws._client = None
    ws._bucket_obj = None
    ws._read_json_blob = MagicMock(return_value=manifest_dict)

    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_blob.exists.return_value = False  # no pose.parquet
    mock_bucket.blob.return_value = mock_blob
    ws._get_bucket = MagicMock(return_value=mock_bucket)

    # get_window_video_url uses generate_signed_url; mock the whole path
    ws.get_window_video_url = MagicMock(return_value="https://mocked-gcs/video.mp4")
    return ws


def _make_encoder(tmp_path: Path):
    from pipeline.modules.encoder.encoder import Encoder
    from pipeline.modules.encoder.reasoning_arm import StubVLMClient
    from pipeline.modules.encoder.vocabulary import DEFAULT_VOCABULARY
    return Encoder(vlm=StubVLMClient(), vocabulary=DEFAULT_VOCABULARY, cache_root=tmp_path)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

def test_storage_manifest_flows_to_encoder() -> None:
    """WindowManifest from Storage (pose_summary) reaches Encoder prompt."""
    from pipeline.interfaces.schema_record import SchemaRecord
    from pipeline.modules.encoder.schema import WindowInput

    manifest_dict = {
        "segment_id": "seg_integration_001",
        "window_idx": 0,
        "source_format": "waymo_parquet",
        "source_schema_version": "waymo_v2_camera_image_v1",
        "window_start_ts_us": 0,
        "window_end_ts_us": 8_000_000,
        "frame_count": 80,
        "cameras": ["FRONT"],
        "ingested_at": "2026-05-25T00:00:00Z",
        "pose_summary": "Vehicle traveled 25m forward at 10 km/h.",
        "extra": {},
    }

    storage = _make_gcs_backed_storage(manifest_dict)

    with tempfile.TemporaryDirectory() as tmp:
        enc = _make_encoder(Path(tmp))
        record = enc.process(
            WindowInput(
                segment_id="seg_integration_001",
                window_idx=0,
                storage=storage,
            )
        )

    # The record must be the interface type
    assert isinstance(record, SchemaRecord)
    assert record.succeeded
    assert record.window_id.segment_id == "seg_integration_001"
    assert record.window_id.window_idx == 0

    # Storage's get_window_video_url was called (encoder needs video URL)
    storage.get_window_video_url.assert_called_once()


def test_storage_null_pose_summary_does_not_crash_encoder() -> None:
    """Encoder must handle pose_summary=None gracefully (Storage best-effort)."""
    from pipeline.interfaces.schema_record import SchemaRecord
    from pipeline.modules.encoder.schema import WindowInput

    manifest_dict = {
        "segment_id": "seg_integration_002",
        "window_idx": 1,
        "source_format": "waymo_parquet",
        "source_schema_version": "waymo_v2_camera_image_v1",
        "window_start_ts_us": 0,
        "window_end_ts_us": 8_000_000,
        "frame_count": 80,
        "cameras": ["FRONT"],
        "ingested_at": "2026-05-25T00:00:00Z",
        "pose_summary": None,  # no pose was available at ingest time
        "extra": {},
    }

    storage = _make_gcs_backed_storage(manifest_dict)

    with tempfile.TemporaryDirectory() as tmp:
        enc = _make_encoder(Path(tmp))
        record = enc.process(
            WindowInput(segment_id="seg_integration_002", window_idx=1, storage=storage)
        )

    assert isinstance(record, SchemaRecord)
    assert record.succeeded  # null pose is handled, not a failure


def test_encoder_output_is_json_serializable_for_downstream() -> None:
    """The SchemaRecord produced must be serializable for Hypothesizer (Module 3)."""
    from pipeline.modules.encoder.schema import WindowInput

    manifest_dict = {
        "segment_id": "seg_integration_003",
        "window_idx": 0,
        "source_format": "waymo_parquet",
        "source_schema_version": "waymo_v2_camera_image_v1",
        "window_start_ts_us": 0,
        "window_end_ts_us": 8_000_000,
        "frame_count": 80,
        "cameras": ["FRONT"],
        "ingested_at": "2026-05-25T00:00:00Z",
        "pose_summary": "Vehicle traveled 10m.",
        "extra": {},
    }

    storage = _make_gcs_backed_storage(manifest_dict)

    with tempfile.TemporaryDirectory() as tmp:
        enc = _make_encoder(Path(tmp))
        record = enc.process(
            WindowInput(segment_id="seg_integration_003", window_idx=0, storage=storage)
        )

    # Downstream module (Hypothesizer) will receive this via JSON
    d = record.to_json()
    serialized = json.dumps(d)  # must not raise
    assert len(serialized) > 0

    # Downstream can reconstruct the record
    from pipeline.interfaces.schema_record import SchemaRecord
    restored = SchemaRecord.from_json(json.loads(serialized))
    assert restored.window_id.segment_id == "seg_integration_003"
    assert restored.fields == record.fields


def test_batch_of_windows_all_produce_valid_schema_records() -> None:
    """Encoder.process_batch() must return one SchemaRecord per window."""
    from pipeline.interfaces.schema_record import SchemaRecord
    from pipeline.modules.encoder.schema import WindowInput

    manifest_dict = {
        "segment_id": "seg_batch_001",
        "window_idx": 0,
        "source_format": "waymo_parquet",
        "source_schema_version": "v1",
        "window_start_ts_us": 0,
        "window_end_ts_us": 8_000_000,
        "frame_count": 80,
        "cameras": ["FRONT"],
        "ingested_at": "2026-05-25T00:00:00Z",
        "pose_summary": "Test.",
        "extra": {},
    }

    storage = _make_gcs_backed_storage(manifest_dict)

    with tempfile.TemporaryDirectory() as tmp:
        enc = _make_encoder(Path(tmp))
        windows = [
            WindowInput(segment_id="seg_batch_001", window_idx=i, storage=storage)
            for i in range(4)
        ]
        records = enc.process_batch(windows)

    assert len(records) == 4
    for i, r in enumerate(records):
        assert isinstance(r, SchemaRecord), f"Window {i} did not produce SchemaRecord"
        assert r.window_id.window_idx == i
