"""Smoke tests for Module 1: Storage.

Tests run without GCS access. They verify:
  - All public classes and error types import cleanly
  - Error types print loudly (to stderr) on construction
  - SourceAdapter Protocol is satisfied by both adapter classes
  - WindowConfig defaults are sane
  - _build_windows slices frames correctly
  - _pose_summary_text produces a human-readable string
  - WaymoParquetSource raises SourceUnreachableError (not silent) on bad bucket
  - WaymoTFRecordSource raises SourceUnreachableError (not silent) on missing file

Run:
    cd /path/to/project
    python -m pytest pipeline/modules/storage/tests/test_smoke.py -v
"""

from __future__ import annotations

import io
import sys
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import checks
# ---------------------------------------------------------------------------

def test_base_imports() -> None:
    from pipeline.modules.storage.adapters.base import (
        DatasetManifest,
        Frame,
        IngestionRequest,
        PoseArray,
        PoseData,
        PoseRecord,
        RawSegment,
        SourceAdapter,
        SourceAdapterError,
        SourceSchemaVersionError,
        SourceUnreachableError,
        StorageError,
        ValidationResult,
        WindowConfig,
        WindowKey,
        WindowManifest,
        WindowStorageError,
    )
    assert True


def test_parquet_adapter_imports() -> None:
    from pipeline.modules.storage.adapters.parquet import WaymoParquetSource
    assert WaymoParquetSource.format_name == "waymo_parquet"
    assert WaymoParquetSource.schema_version.startswith("waymo_v2")


def test_tfrecord_adapter_imports() -> None:
    from pipeline.modules.storage.adapters.tfrecord import WaymoTFRecordSource
    assert WaymoTFRecordSource.format_name == "waymo_tfrecord"


def test_ingestion_imports() -> None:
    from pipeline.modules.storage.ingestion import IngestionPipeline, IngestionResult
    assert IngestionPipeline is not None


def test_client_imports() -> None:
    from pipeline.modules.storage.client import WindowStorage
    assert WindowStorage is not None


# ---------------------------------------------------------------------------
# Error loudness
# ---------------------------------------------------------------------------

def test_source_unreachable_prints_to_stderr(capsys: pytest.CaptureFixture) -> None:
    from pipeline.modules.storage.adapters.base import SourceUnreachableError
    with pytest.raises(SourceUnreachableError):
        raise SourceUnreachableError("gs://test/bucket", "connection refused")
    captured = capsys.readouterr()
    assert "STORAGE ERROR" in captured.err
    assert "SourceUnreachableError" in captured.err


def test_schema_version_error_prints_to_stderr(capsys: pytest.CaptureFixture) -> None:
    from pipeline.modules.storage.adapters.base import SourceSchemaVersionError
    with pytest.raises(SourceSchemaVersionError):
        raise SourceSchemaVersionError("waymo_parquet", "v1", "v2")
    captured = capsys.readouterr()
    assert "STORAGE ERROR" in captured.err
    assert "Schema version mismatch" in captured.err


def test_adapter_error_prints_to_stderr(capsys: pytest.CaptureFixture) -> None:
    from pipeline.modules.storage.adapters.base import SourceAdapterError
    with pytest.raises(SourceAdapterError):
        raise SourceAdapterError("waymo_parquet", "seg_123", ValueError("bad data"))
    captured = capsys.readouterr()
    assert "STORAGE ERROR" in captured.err


# ---------------------------------------------------------------------------
# SourceAdapter Protocol compliance
# ---------------------------------------------------------------------------

def test_parquet_source_satisfies_protocol() -> None:
    from pipeline.modules.storage.adapters.base import SourceAdapter
    from pipeline.modules.storage.adapters.parquet import WaymoParquetSource
    src = WaymoParquetSource(bucket="test-bucket", prefix="test/prefix")
    assert isinstance(src, SourceAdapter), (
        "WaymoParquetSource does not satisfy the SourceAdapter Protocol"
    )


def test_tfrecord_source_satisfies_protocol() -> None:
    from pipeline.modules.storage.adapters.base import SourceAdapter
    from pipeline.modules.storage.adapters.tfrecord import WaymoTFRecordSource
    src = WaymoTFRecordSource(tfrecord_paths=[])
    assert isinstance(src, SourceAdapter), (
        "WaymoTFRecordSource does not satisfy the SourceAdapter Protocol"
    )


# ---------------------------------------------------------------------------
# WindowConfig defaults
# ---------------------------------------------------------------------------

def test_window_config_defaults() -> None:
    from pipeline.modules.storage.adapters.base import WindowConfig
    cfg = WindowConfig()
    assert cfg.length_frames == 80
    assert cfg.stride_frames == 80
    assert cfg.target_fps == 10
    assert "FRONT" in cfg.cameras
    assert len(cfg.cameras) == 5


# ---------------------------------------------------------------------------
# _build_windows slicing
# ---------------------------------------------------------------------------

def test_build_windows_basic_slicing() -> None:
    from pipeline.modules.storage.adapters.base import Frame, PoseArray, RawSegment, WindowConfig
    from pipeline.modules.storage.ingestion import _build_windows

    # 200 FRONT frames, 80-frame windows → 2 full windows + 1 partial (40 frames)
    n = 200
    front_frames = [
        Frame(timestamp_us=i * 100_000, frame_index=i, jpeg_bytes=b"x")
        for i in range(n)
    ]
    raw = RawSegment(
        segment_id="test_seg",
        source_format="waymo_parquet",
        source_schema_version="v1",
        duration_seconds=20.0,
        frame_rate_hz=10.0,
        cameras={"FRONT": front_frames},
        pose=PoseArray(),
    )
    cfg = WindowConfig(length_frames=80, stride_frames=80)
    windows = _build_windows(raw, cfg)
    assert len(windows) == 3, f"Expected 3 windows, got {len(windows)}"
    assert len(windows[0]["FRONT"]) == 80
    assert len(windows[1]["FRONT"]) == 80
    assert len(windows[2]["FRONT"]) == 40


def test_build_windows_empty_segment() -> None:
    from pipeline.modules.storage.adapters.base import PoseArray, RawSegment, WindowConfig
    from pipeline.modules.storage.ingestion import _build_windows

    raw = RawSegment(
        segment_id="empty",
        source_format="waymo_parquet",
        source_schema_version="v1",
        duration_seconds=0.0,
        frame_rate_hz=10.0,
        cameras={},
        pose=PoseArray(),
    )
    windows = _build_windows(raw, WindowConfig())
    assert windows == []


# ---------------------------------------------------------------------------
# Pose helpers
# ---------------------------------------------------------------------------

def test_pose_summary_with_data() -> None:
    from pipeline.modules.storage.adapters.base import PoseArray, PoseRecord
    from pipeline.modules.storage.ingestion import _pose_summary_text

    pose = PoseArray(records=[
        PoseRecord(timestamp_us=0, x=0.0, y=0.0, z=0.0),
        PoseRecord(timestamp_us=8_000_000, x=20.0, y=5.0, z=0.0),
    ])
    summary = _pose_summary_text(pose, "seg_001", 0)
    assert summary is not None
    assert "seg_001" in summary
    assert "meters" in summary


def test_pose_summary_empty() -> None:
    from pipeline.modules.storage.adapters.base import PoseArray
    from pipeline.modules.storage.ingestion import _pose_summary_text

    assert _pose_summary_text(PoseArray(), "seg_001", 0) is None


# ---------------------------------------------------------------------------
# WaymoParquetSource — no GCS needed for unit tests
# ---------------------------------------------------------------------------

def test_parquet_source_unreachable_on_bad_bucket(capsys: pytest.CaptureFixture) -> None:
    """Connecting to a nonexistent GCS bucket raises SourceUnreachableError loudly."""
    from pipeline.modules.storage.adapters.parquet import WaymoParquetSource
    from pipeline.modules.storage.adapters.base import SourceUnreachableError

    src = WaymoParquetSource(
        bucket="this-bucket-does-not-exist-smoke-test-xyz",
        prefix="test/prefix",
        gcs_credentials=None,
    )

    # Mock gcsfs so the test works without real GCS credentials
    mock_fs = MagicMock()
    mock_fs.ls.side_effect = Exception("403 Forbidden: bucket does not exist")

    with patch.dict(
        sys.modules,
        {"gcsfs": MagicMock(GCSFileSystem=MagicMock(return_value=mock_fs))},
    ):
        with pytest.raises(SourceUnreachableError):
            src.list_segments()

    captured = capsys.readouterr()
    assert "STORAGE ERROR" in captured.err


def test_parquet_source_validate_missing_file() -> None:
    from pipeline.modules.storage.adapters.parquet import WaymoParquetSource

    src = WaymoParquetSource(bucket="test-bucket", prefix="test/prefix")

    mock_fs = MagicMock()
    mock_fs.open.side_effect = FileNotFoundError("not found")

    with patch.object(src, "_get_fs", return_value=mock_fs):
        result = src.validate_segment("nonexistent_segment")
    assert not result.valid
    assert any("not found" in e.lower() or "parquet" in e.lower() for e in result.errors)


# ---------------------------------------------------------------------------
# WaymoTFRecordSource — filesystem tests
# ---------------------------------------------------------------------------

def test_tfrecord_source_missing_file() -> None:
    from pipeline.modules.storage.adapters.base import SourceUnreachableError
    from pipeline.modules.storage.adapters.tfrecord import WaymoTFRecordSource

    src = WaymoTFRecordSource(tfrecord_paths=["/nonexistent/path/segment.tfrecord"])
    with pytest.raises(SourceUnreachableError):
        src.list_segments()


def test_tfrecord_source_validate_missing() -> None:
    from pipeline.modules.storage.adapters.tfrecord import WaymoTFRecordSource

    src = WaymoTFRecordSource(tfrecord_paths=["/nonexistent/segment.tfrecord"])
    result = src.validate_segment("segment")
    assert not result.valid


def test_tfrecord_source_no_paths_raises() -> None:
    with pytest.raises(Exception):
        from pipeline.modules.storage.adapters.tfrecord import WaymoTFRecordSource
        WaymoTFRecordSource()


# ---------------------------------------------------------------------------
# PoseArray helpers
# ---------------------------------------------------------------------------

def test_pose_array_slice() -> None:
    from pipeline.modules.storage.adapters.base import PoseArray, PoseRecord

    pose = PoseArray(records=[
        PoseRecord(timestamp_us=0, x=0, y=0, z=0),
        PoseRecord(timestamp_us=1_000_000, x=1, y=0, z=0),
        PoseRecord(timestamp_us=2_000_000, x=2, y=0, z=0),
        PoseRecord(timestamp_us=3_000_000, x=3, y=0, z=0),
    ])
    sliced = pose.slice_window(1_000_000, 2_000_000)
    assert len(sliced.records) == 2
    assert sliced.records[0].x == 1.0
    assert sliced.records[1].x == 2.0


def test_window_key_str() -> None:
    from pipeline.modules.storage.adapters.base import WindowKey
    k = WindowKey(segment_id="abc123", window_idx=7)
    assert str(k) == "abc123/0007"


# ---------------------------------------------------------------------------
# _parse_bucket_uri
# ---------------------------------------------------------------------------

def test_parse_bucket_uri_valid() -> None:
    from pipeline.modules.storage.ingestion import _parse_bucket_uri
    bucket, prefix = _parse_bucket_uri("gs://my-bucket/some/prefix")
    assert bucket == "my-bucket"
    assert prefix == "some/prefix"


def test_parse_bucket_uri_no_prefix() -> None:
    from pipeline.modules.storage.ingestion import _parse_bucket_uri
    bucket, prefix = _parse_bucket_uri("gs://my-bucket")
    assert bucket == "my-bucket"
    assert prefix == ""


def test_parse_bucket_uri_s3_raises() -> None:
    from pipeline.modules.storage.adapters.base import IngestionError
    from pipeline.modules.storage.ingestion import _parse_bucket_uri, IngestionError
    with pytest.raises(IngestionError):
        _parse_bucket_uri("s3://my-bucket/prefix")


def test_parse_bucket_uri_bad_scheme_raises() -> None:
    from pipeline.modules.storage.ingestion import _parse_bucket_uri, IngestionError
    with pytest.raises(IngestionError):
        _parse_bucket_uri("http://my-bucket/prefix")
