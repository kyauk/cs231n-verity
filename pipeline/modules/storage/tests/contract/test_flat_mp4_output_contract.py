"""Contract test for FlatMP4Storage output.

Asserts every field declared in the README's Output Contract section
(`pipeline/README.md` → Module 1 → Two retrieval implementations table) is
present, correctly typed, and carries the documented value.

If this test fails, EITHER the README is stale OR the implementation has
drifted. Reconcile per Step 5 of the hygiene protocol.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pipeline.interfaces.window import (
    WindowKey,
    WindowManifest,
    WindowStorageBase,
)
from pipeline.modules.storage import FlatMP4Storage


# ---------------------------------------------------------------------------
# Fixture: minimal in-memory FlatMP4Storage with one MP4 blob
# ---------------------------------------------------------------------------

@pytest.fixture()
def storage() -> FlatMP4Storage:
    s = FlatMP4Storage(bucket_uri="gs://test-bucket/some/prefix",
                       cameras=["FRONT"])
    blob = MagicMock()
    blob.name = "some/prefix/seg_001.mp4"
    blob.exists.return_value = True
    blob.reload.return_value = None
    blob.size = 100_000
    blob.content_type = "video/mp4"
    blob.generate_signed_url.return_value = "https://signed/seg_001.mp4"
    fake_bucket = MagicMock()
    fake_bucket.list_blobs.return_value = [blob]
    fake_bucket.blob.return_value = blob
    s._bucket_obj = fake_bucket
    return s


# ---------------------------------------------------------------------------
# Contract: WindowStorageBase Protocol satisfaction (lego boundary)
# ---------------------------------------------------------------------------

def test_satisfies_window_storage_base_protocol(storage: FlatMP4Storage) -> None:
    """README claim: 'Both satisfy WindowStorageBase'."""
    assert isinstance(storage, WindowStorageBase)


# ---------------------------------------------------------------------------
# Contract: list_windows
# ---------------------------------------------------------------------------

def test_list_windows_returns_list_of_window_key(storage: FlatMP4Storage) -> None:
    out = storage.list_windows()
    assert isinstance(out, list)
    assert all(isinstance(w, WindowKey) for w in out)


def test_list_windows_window_idx_is_zero(storage: FlatMP4Storage) -> None:
    """README claim: 'one window per MP4'."""
    for w in storage.list_windows():
        assert w.window_idx == 0


# ---------------------------------------------------------------------------
# Contract: get_window_video_url
# ---------------------------------------------------------------------------

def test_get_window_video_url_returns_str(storage: FlatMP4Storage) -> None:
    url = storage.get_window_video_url("seg_001", 0, camera="FRONT")
    assert isinstance(url, str)
    assert url  # non-empty


# ---------------------------------------------------------------------------
# Contract: get_window_manifest — every documented field
# ---------------------------------------------------------------------------

def test_manifest_is_window_manifest(storage: FlatMP4Storage) -> None:
    m = storage.get_window_manifest("seg_001", 0)
    assert isinstance(m, WindowManifest)


def test_manifest_segment_id(storage: FlatMP4Storage) -> None:
    m = storage.get_window_manifest("seg_001", 0)
    assert isinstance(m.segment_id, str)
    assert m.segment_id == "seg_001"


def test_manifest_window_idx_is_zero(storage: FlatMP4Storage) -> None:
    """README: 'window_idx is always 0'."""
    m = storage.get_window_manifest("seg_001", 0)
    assert isinstance(m.window_idx, int)
    assert m.window_idx == 0


def test_manifest_source_format_is_flat_mp4(storage: FlatMP4Storage) -> None:
    """README: 'source_format=\"flat_mp4\"'."""
    m = storage.get_window_manifest("seg_001", 0)
    assert m.source_format == "flat_mp4"


def test_manifest_source_schema_version_is_string(storage: FlatMP4Storage) -> None:
    m = storage.get_window_manifest("seg_001", 0)
    assert isinstance(m.source_schema_version, str)
    assert m.source_schema_version  # non-empty


def test_manifest_frame_count_is_zero(storage: FlatMP4Storage) -> None:
    """README: 'frame_count=0'."""
    m = storage.get_window_manifest("seg_001", 0)
    assert isinstance(m.frame_count, int)
    assert m.frame_count == 0


def test_manifest_cameras_reflects_configured(storage: FlatMP4Storage) -> None:
    """README: 'cameras=<configured list>'."""
    m = storage.get_window_manifest("seg_001", 0)
    assert isinstance(m.cameras, list)
    assert m.cameras == ["FRONT"]


def test_manifest_pose_summary_is_none(storage: FlatMP4Storage) -> None:
    """README: 'pose_summary=None'."""
    m = storage.get_window_manifest("seg_001", 0)
    assert m.pose_summary is None


def test_manifest_ingested_at_is_iso8601(storage: FlatMP4Storage) -> None:
    m = storage.get_window_manifest("seg_001", 0)
    assert isinstance(m.ingested_at, str)
    assert m.ingested_at.endswith("Z")  # ISO-8601 UTC


# ---------------------------------------------------------------------------
# Contract: round-trip through to_json/from_json (boundary stability)
# ---------------------------------------------------------------------------

def test_manifest_roundtrips_through_json(storage: FlatMP4Storage) -> None:
    """Boundary stability: every WindowManifest from this storage must
    survive JSON round-trip without information loss. The Encoder
    serializes/deserializes manifests through this path."""
    original = storage.get_window_manifest("seg_001", 0)
    restored = WindowManifest.from_json(original.to_json())
    assert restored == original


# ---------------------------------------------------------------------------
# Contract: multi-camera config produces multi-camera manifest
# ---------------------------------------------------------------------------

def test_manifest_carries_all_configured_cameras() -> None:
    """README: 'cameras=<configured>'. Verify the full list is preserved."""
    cams = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"]
    s = FlatMP4Storage(bucket_uri="gs://b/p", cameras=cams)
    m = s.get_window_manifest("seg_001", 0)
    assert m.cameras == cams
