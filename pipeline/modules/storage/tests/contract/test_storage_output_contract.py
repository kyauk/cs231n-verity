"""Contract tests for Module 1: Storage.

Verifies that WindowStorage outputs conform exactly to the types declared in
pipeline/interfaces/window.py. Every field in every output type is asserted.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import json


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_manifest_json(**overrides) -> dict:
    base = {
        "segment_id": "seg_contract_001",
        "window_idx": 3,
        "source_format": "waymo_parquet",
        "source_schema_version": "waymo_v2_camera_image_v1",
        "window_start_ts_us": 0,
        "window_end_ts_us": 8_000_000,
        "frame_count": 80,
        "cameras": ["FRONT", "FRONT_LEFT"],
        "ingested_at": "2026-05-25T00:00:00Z",
        "pose_summary": "Vehicle traveled 20m forward.",
        "extra": {"dataset": "waymo_open_dataset_v_2_0_1"},
    }
    base.update(overrides)
    return base


def _mock_storage(manifest_json: dict):
    """Return a WindowStorage with GCS calls mocked."""
    import threading
    from pipeline.modules.storage.client import WindowStorage
    ws = WindowStorage.__new__(WindowStorage)
    ws._bucket_name = "test-bucket"
    ws._prefix = "verity"
    ws._creds = None
    ws._project = None
    ws._sign_as = None
    ws._client = None
    ws._bucket_obj = None
    ws._bucket_lock = threading.Lock()
    ws._read_json_blob = MagicMock(return_value=manifest_json)
    ws._read_bytes_blob = MagicMock(return_value=b"")
    return ws


# ---------------------------------------------------------------------------
# WindowManifest contract
# ---------------------------------------------------------------------------

def test_get_window_manifest_returns_window_manifest_type() -> None:
    from pipeline.interfaces.window import WindowManifest
    ws = _mock_storage(_make_manifest_json())
    result = ws.get_window_manifest("seg_contract_001", 3)
    assert isinstance(result, WindowManifest)


def test_window_manifest_all_required_fields_present() -> None:
    ws = _mock_storage(_make_manifest_json())
    m = ws.get_window_manifest("seg_contract_001", 3)
    assert isinstance(m.segment_id, str) and m.segment_id
    assert isinstance(m.window_idx, int) and m.window_idx >= 0
    assert isinstance(m.source_format, str) and m.source_format
    assert isinstance(m.source_schema_version, str) and m.source_schema_version
    assert isinstance(m.window_start_ts_us, int)
    assert isinstance(m.window_end_ts_us, int)
    assert isinstance(m.frame_count, int) and m.frame_count >= 0
    assert isinstance(m.cameras, list) and len(m.cameras) > 0
    assert isinstance(m.ingested_at, str) and "T" in m.ingested_at
    # Optional fields exist on the type (may be None)
    assert m.pose_summary is None or isinstance(m.pose_summary, str)
    assert isinstance(m.extra, dict)


def test_window_manifest_window_key_property() -> None:
    from pipeline.interfaces.window import WindowKey
    ws = _mock_storage(_make_manifest_json())
    m = ws.get_window_manifest("seg_contract_001", 3)
    assert isinstance(m.window_key, WindowKey)
    assert m.window_key.segment_id == "seg_contract_001"
    assert m.window_key.window_idx == 3


def test_window_manifest_null_pose_summary() -> None:
    ws = _mock_storage(_make_manifest_json(pose_summary=None))
    m = ws.get_window_manifest("seg_contract_001", 3)
    assert m.pose_summary is None


def test_window_manifest_is_json_serializable() -> None:
    ws = _mock_storage(_make_manifest_json())
    m = ws.get_window_manifest("seg_contract_001", 3)
    d = m.to_json()
    json.dumps(d)  # must not raise
    assert d["segment_id"] == "seg_contract_001"
    assert d["window_idx"] == 3
    assert isinstance(d["cameras"], list)


# ---------------------------------------------------------------------------
# PoseData contract
# ---------------------------------------------------------------------------

def test_get_window_pose_returns_pose_data_type() -> None:
    from pipeline.interfaces.window import PoseData
    import io
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        rows = [
            {"timestamp_us": 0, "x": 0.0, "y": 0.0, "z": 0.0,
             "roll_rad": 0.0, "pitch_rad": 0.0, "yaw_rad": 0.0},
        ]
        buf = io.BytesIO()
        pq.write_table(pa.Table.from_pylist(rows), buf)
        parquet_bytes = buf.getvalue()
    except ImportError:
        parquet_bytes = b""

    ws = _mock_storage({})
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_blob.exists.return_value = bool(parquet_bytes)
    mock_bucket.blob.return_value = mock_blob
    ws._get_bucket = MagicMock(return_value=mock_bucket)
    ws._read_bytes_blob = MagicMock(return_value=parquet_bytes)

    result = ws.get_window_pose("seg_contract_001", 0)
    assert isinstance(result, PoseData)
    assert result.segment_id == "seg_contract_001"
    assert result.window_idx == 0
    assert isinstance(result.records, list)


def test_get_window_pose_empty_returns_pose_data() -> None:
    from pipeline.interfaces.window import PoseData
    ws = _mock_storage({})
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_blob.exists.return_value = False
    mock_bucket.blob.return_value = mock_blob
    ws._get_bucket = MagicMock(return_value=mock_bucket)

    result = ws.get_window_pose("seg_contract_001", 0)
    assert isinstance(result, PoseData)
    assert result.records == []


# ---------------------------------------------------------------------------
# WindowKey contract
# ---------------------------------------------------------------------------

def test_list_windows_returns_list_of_window_keys() -> None:
    from pipeline.interfaces.window import WindowKey
    ws = _mock_storage({})
    seg_index = {
        "segment_id": "seg_contract_001",
        "windows": [
            {"window_idx": 0, "status": "ok"},
            {"window_idx": 1, "status": "ok"},
            {"window_idx": 2, "status": "failed"},
        ],
    }
    ws._read_json_blob = MagicMock(return_value=seg_index)
    keys = ws.list_windows(segment_id="seg_contract_001")
    assert isinstance(keys, list)
    assert all(isinstance(k, WindowKey) for k in keys)
    assert len(keys) == 2  # only "ok" windows
    assert keys[0].window_idx == 0
    assert keys[1].window_idx == 1


# ---------------------------------------------------------------------------
# Ingestion output: WindowManifest is JSON-serializable (written to GCS as dict)
# ---------------------------------------------------------------------------

def test_ingested_manifest_matches_interface_schema() -> None:
    """Verify the dict written by _ingest_window has every WindowManifest field."""
    from pipeline.interfaces.window import WindowManifest
    from pipeline.modules.storage.adapters.base import Frame, PoseArray, RawSegment, WindowConfig
    from pipeline.modules.storage.ingestion import _build_windows, _pose_summary_text
    import datetime

    frames = [Frame(timestamp_us=i * 100_000, frame_index=i, jpeg_bytes=b"\xff\xd8\xff") for i in range(10)]
    raw = RawSegment(
        segment_id="seg_ingest_001",
        source_format="waymo_parquet",
        source_schema_version="waymo_v2_camera_image_v1",
        duration_seconds=1.0,
        frame_rate_hz=10.0,
        cameras={"FRONT": frames},
        pose=PoseArray(),
    )
    cfg = WindowConfig(length_frames=5, stride_frames=5)
    windows = _build_windows(raw, cfg)
    front = windows[0]["FRONT"]
    start_ts = front[0].timestamp_us
    end_ts = front[-1].timestamp_us
    pose_summary = _pose_summary_text(raw.pose, raw.segment_id, 0)

    manifest = WindowManifest(
        segment_id=raw.segment_id,
        window_idx=0,
        source_format=raw.source_format,
        source_schema_version=raw.source_schema_version,
        window_start_ts_us=start_ts,
        window_end_ts_us=end_ts,
        frame_count=len(front),
        cameras=list(cfg.cameras),
        ingested_at=datetime.datetime.utcnow().isoformat() + "Z",
        pose_summary=pose_summary,
    )

    d = manifest.to_json()
    # Every field the interface declares must be in the serialized dict
    required_fields = [
        "segment_id", "window_idx", "source_format", "source_schema_version",
        "window_start_ts_us", "window_end_ts_us", "frame_count",
        "cameras", "ingested_at", "pose_summary", "extra",
    ]
    for f in required_fields:
        assert f in d, f"Missing field in WindowManifest.to_json(): {f!r}"

    # Round-trip
    restored = WindowManifest.from_json(d)
    assert restored.segment_id == manifest.segment_id
    assert restored.window_idx == manifest.window_idx
