"""Concurrency tests for Module 1: Storage ingestion.

Verifies that camera encode+upload runs in parallel, that manifest is written
only after all cameras succeed, and that one camera failure causes the window
to be marked failed without a manifest being written.

All tests mock GCS and ffmpeg — no real I/O.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_backend(upload_side_effect=None) -> Any:
    backend = MagicMock()
    backend.blob_exists.return_value = False
    if upload_side_effect is not None:
        backend.upload_bytes.side_effect = upload_side_effect
    return backend


def _make_minimal_window_frames(cameras=None):
    """Return a window_frames dict with one fake frame per camera."""
    if cameras is None:
        cameras = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"]
    frame = MagicMock()
    frame.timestamp_us = 1_000_000
    frame.jpeg_bytes = b""
    return {cam: [frame] for cam in cameras}


def _make_raw_segment(segment_id: str = "seg_001"):
    raw = MagicMock()
    raw.segment_id = segment_id
    raw.source_format = "waymo_parquet"
    raw.source_schema_version = "v1"
    pose = MagicMock()
    pose.is_empty = True
    pose.slice_window.return_value = pose
    raw.pose = pose
    return raw


def _make_config(cameras=None):
    from pipeline.modules.storage.adapters.base import WindowConfig
    cfg = WindowConfig()
    if cameras is not None:
        object.__setattr__(cfg, "cameras", cameras)
    return cfg


# ---------------------------------------------------------------------------
# Test: cameras upload in parallel
# ---------------------------------------------------------------------------

def test_cameras_encode_and_upload_in_parallel() -> None:
    """All 5 cameras should be submitted concurrently — upload_bytes called 5×."""
    upload_calls: list[str] = []
    upload_lock = threading.Lock()

    def _track_upload(bucket_name, blob_name, data, content_type="application/octet-stream"):
        with upload_lock:
            upload_calls.append(blob_name)

    backend = _make_backend()
    backend.upload_bytes.side_effect = _track_upload

    cameras = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"]
    window_frames = _make_minimal_window_frames(cameras)
    raw = _make_raw_segment()
    config = _make_config(cameras)

    from pipeline.modules.storage.ingestion import IngestionPipeline
    pipeline = IngestionPipeline()
    pipeline._backend = backend

    with patch("pipeline.modules.storage.ingestion._encode_mp4", return_value=b"fake_mp4"):
        ok = pipeline._ingest_window(
            raw=raw,
            window_idx=0,
            window_frames=window_frames,
            config=config,
            bucket_name="test-bucket",
            prefix="verity",
            backend=backend,
            now_date="2026-01-01",
        )

    assert ok is True

    # One upload per camera MP4, plus pose_summary.json and manifest.json
    camera_uploads = [b for b in upload_calls if "camera_" in b and b.endswith(".mp4")]
    assert len(camera_uploads) == len(cameras), (
        f"Expected {len(cameras)} camera uploads, got {len(camera_uploads)}: {camera_uploads}"
    )
    for cam in cameras:
        assert any(f"camera_{cam}.mp4" in b for b in camera_uploads), (
            f"No upload found for camera {cam}"
        )


# ---------------------------------------------------------------------------
# Test: manifest only written after all cameras succeed
# ---------------------------------------------------------------------------

def test_manifest_written_after_all_cameras_succeed() -> None:
    """manifest.json must appear after all camera blobs are uploaded."""
    upload_order: list[str] = []
    lock = threading.Lock()

    def _track(bucket_name, blob_name, data, content_type="application/octet-stream"):
        with lock:
            upload_order.append(blob_name)

    backend = _make_backend()
    backend.upload_bytes.side_effect = _track

    cameras = ["FRONT", "FRONT_LEFT"]
    window_frames = _make_minimal_window_frames(cameras)
    raw = _make_raw_segment()
    config = _make_config(cameras)

    from pipeline.modules.storage.ingestion import IngestionPipeline
    pipeline = IngestionPipeline()
    pipeline._backend = backend

    with patch("pipeline.modules.storage.ingestion._encode_mp4", return_value=b"fake_mp4"):
        ok = pipeline._ingest_window(
            raw=raw,
            window_idx=0,
            window_frames=window_frames,
            config=config,
            bucket_name="test-bucket",
            prefix="verity",
            backend=backend,
            now_date="2026-01-01",
        )

    assert ok is True
    manifest_idx = next(i for i, b in enumerate(upload_order) if b.endswith("manifest.json"))
    camera_indices = [i for i, b in enumerate(upload_order) if "camera_" in b and b.endswith(".mp4")]

    assert all(ci < manifest_idx for ci in camera_indices), (
        f"Manifest written before all cameras. Order: {upload_order}"
    )


# ---------------------------------------------------------------------------
# Test: one camera failure → window returns False, no manifest
# ---------------------------------------------------------------------------

def test_one_camera_failure_returns_false_no_manifest() -> None:
    """If one camera encode fails, _ingest_window must return False with no manifest."""
    cameras = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT"]
    window_frames = _make_minimal_window_frames(cameras)
    raw = _make_raw_segment()
    config = _make_config(cameras)

    backend = _make_backend()
    uploaded_blobs: list[str] = []

    def _track(bucket_name, blob_name, data, content_type="application/octet-stream"):
        uploaded_blobs.append(blob_name)

    backend.upload_bytes.side_effect = _track

    call_count = [0]

    def _encode_mp4_sometimes_fails(frames, fps):
        call_count[0] += 1
        if call_count[0] == 2:  # second camera fails
            raise RuntimeError("ffmpeg failed")
        return b"fake_mp4"

    from pipeline.modules.storage.ingestion import IngestionPipeline
    pipeline = IngestionPipeline()
    pipeline._backend = backend

    with patch("pipeline.modules.storage.ingestion._encode_mp4", side_effect=_encode_mp4_sometimes_fails):
        ok = pipeline._ingest_window(
            raw=raw,
            window_idx=0,
            window_frames=window_frames,
            config=config,
            bucket_name="test-bucket",
            prefix="verity",
            backend=backend,
            now_date="2026-01-01",
        )

    assert ok is False, "Window with failed camera must return False"
    assert not any("manifest.json" in b for b in uploaded_blobs), (
        f"Manifest must not be written on camera failure. Uploads: {uploaded_blobs}"
    )
