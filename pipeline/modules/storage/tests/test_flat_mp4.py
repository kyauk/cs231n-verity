"""Unit tests for FlatMP4Storage.

All tests mock the GCS client so nothing hits the network. We also verify
the class satisfies WindowStorageBase at runtime.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pipeline.interfaces.errors import WindowStorageError
from pipeline.interfaces.window import (
    WindowKey,
    WindowManifest,
    WindowStorageBase,
)
from pipeline.modules.storage import FlatMP4Storage


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _fake_blob(
    name: str,
    exists: bool = True,
    signed_url: str = "https://x",
    size: int = 100_000,
    content_type: str = "video/mp4",
) -> MagicMock:
    blob = MagicMock()
    blob.name = name
    blob.size = size
    blob.content_type = content_type
    if exists:
        blob.reload.return_value = None
    else:
        blob.reload.side_effect = Exception("404 Not Found")
    blob.exists.return_value = exists  # kept for any test paths still on exists()
    blob.generate_signed_url.return_value = signed_url
    return blob


def _fake_bucket(blob_names: list[str], *, signed_url: str = "https://signed") -> MagicMock:
    """Bucket where list_blobs yields the named blobs and bucket.blob(name)
    returns a per-name mock with exists=True and a canned signed URL."""
    blob_map = {name: _fake_blob(name, exists=True, signed_url=signed_url)
                for name in blob_names}
    bucket = MagicMock()
    bucket.list_blobs.side_effect = lambda prefix=None: [
        blob_map[n] for n in blob_names
        if prefix is None or n.startswith(prefix)
    ]
    bucket.blob.side_effect = lambda name: blob_map.get(name, _fake_blob(name, exists=False))
    return bucket


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

def test_constructor_requires_non_empty_cameras() -> None:
    with pytest.raises(ValueError, match="non-empty `cameras`"):
        FlatMP4Storage(bucket_uri="gs://b/p", cameras=[])


def test_constructor_parses_bucket_uri() -> None:
    s = FlatMP4Storage(bucket_uri="gs://my-bucket/some/prefix", cameras=["FRONT"])
    assert s._bucket_name == "my-bucket"
    assert s._prefix == "some/prefix"


def test_satisfies_window_storage_protocol() -> None:
    s = FlatMP4Storage(bucket_uri="gs://b/p", cameras=["FRONT"])
    assert isinstance(s, WindowStorageBase)


# ---------------------------------------------------------------------------
# Filename convention — single vs multi camera
# ---------------------------------------------------------------------------

def test_blob_name_single_camera_is_bare() -> None:
    s = FlatMP4Storage(bucket_uri="gs://b/p", cameras=["FRONT"])
    assert s._blob_name_for("drive_001", "FRONT") == "p/drive_001.mp4"


def test_blob_name_multi_camera_is_suffixed() -> None:
    s = FlatMP4Storage(bucket_uri="gs://b/p",
                       cameras=["FRONT", "FRONT_LEFT", "FRONT_RIGHT"])
    assert s._blob_name_for("drive_001", "FRONT") == "p/drive_001_FRONT.mp4"
    assert s._blob_name_for("drive_001", "FRONT_LEFT") == "p/drive_001_FRONT_LEFT.mp4"


def test_blob_name_no_prefix() -> None:
    s = FlatMP4Storage(bucket_uri="gs://b", cameras=["FRONT"])
    assert s._blob_name_for("drive_001", "FRONT") == "drive_001.mp4"


def test_parse_blob_name_single_camera() -> None:
    s = FlatMP4Storage(bucket_uri="gs://b/p", cameras=["FRONT"])
    assert s._parse_blob_name("p/drive_001.mp4") == ("drive_001", "FRONT")


def test_parse_blob_name_multi_camera() -> None:
    s = FlatMP4Storage(bucket_uri="gs://b/p",
                       cameras=["FRONT", "FRONT_LEFT"])
    assert s._parse_blob_name("p/drive_001_FRONT.mp4") == ("drive_001", "FRONT")
    assert s._parse_blob_name("p/drive_001_FRONT_LEFT.mp4") == ("drive_001", "FRONT_LEFT")


def test_parse_blob_name_multi_camera_skips_non_conforming() -> None:
    """Multi-camera mode skips MP4s whose name doesn't end in _<camera>."""
    s = FlatMP4Storage(bucket_uri="gs://b/p",
                       cameras=["FRONT", "FRONT_LEFT"])
    assert s._parse_blob_name("p/random.mp4") is None
    assert s._parse_blob_name("p/drive_001_BACK.mp4") is None  # BACK not configured


def test_parse_blob_name_ignores_non_mp4() -> None:
    s = FlatMP4Storage(bucket_uri="gs://b/p", cameras=["FRONT"])
    assert s._parse_blob_name("p/notes.txt") is None
    assert s._parse_blob_name("p/preview.jpg") is None


def test_parse_blob_name_segment_id_containing_camera_substring() -> None:
    """Pessimistic-review pin: documents (current, accepted) ambiguity.

    Multi-camera mode splits on the first `_<camera>` suffix match. If a
    customer names a segment such that its ID contains a configured camera
    name as a substring, the parser will strip it. This is an accepted risk
    documented in the module docstring; customers are responsible for
    picking segment IDs that don't collide with camera names.
    """
    s = FlatMP4Storage(bucket_uri="gs://b/p", cameras=["FRONT", "REAR"])
    # File intended as "segment_id=drive_REAR_001, camera=FRONT" parses as
    # "segment_id=drive_REAR_001" — correct here, because _FRONT is the suffix.
    assert s._parse_blob_name("p/drive_REAR_001_FRONT.mp4") == (
        "drive_REAR_001", "FRONT"
    )
    # But if camera order in cameras matters: this is REAR-suffixed.
    assert s._parse_blob_name("p/drive_001_REAR.mp4") == ("drive_001", "REAR")


# ---------------------------------------------------------------------------
# list_windows
# ---------------------------------------------------------------------------

def test_list_windows_single_camera() -> None:
    s = FlatMP4Storage(bucket_uri="gs://b/p", cameras=["FRONT"])
    fake_bucket = _fake_bucket(["p/drive_001.mp4", "p/drive_002.mp4", "p/notes.txt"])
    with patch.object(s, "_get_bucket", return_value=fake_bucket):
        keys = s.list_windows()
    assert keys == [
        WindowKey(segment_id="drive_001", window_idx=0),
        WindowKey(segment_id="drive_002", window_idx=0),
    ]


def test_list_windows_multi_camera_deduplicates_segments() -> None:
    """3 cameras × 2 segments = 6 blobs but only 2 unique WindowKeys."""
    s = FlatMP4Storage(
        bucket_uri="gs://b/p",
        cameras=["FRONT", "FRONT_LEFT", "FRONT_RIGHT"],
    )
    fake_bucket = _fake_bucket([
        "p/drive_001_FRONT.mp4", "p/drive_001_FRONT_LEFT.mp4", "p/drive_001_FRONT_RIGHT.mp4",
        "p/drive_002_FRONT.mp4", "p/drive_002_FRONT_LEFT.mp4", "p/drive_002_FRONT_RIGHT.mp4",
    ])
    with patch.object(s, "_get_bucket", return_value=fake_bucket):
        keys = s.list_windows()
    assert keys == [
        WindowKey(segment_id="drive_001", window_idx=0),
        WindowKey(segment_id="drive_002", window_idx=0),
    ]


def test_list_windows_filter_by_segment_id() -> None:
    s = FlatMP4Storage(bucket_uri="gs://b/p", cameras=["FRONT"])
    fake_bucket = _fake_bucket(["p/drive_001.mp4", "p/drive_002.mp4"])
    with patch.object(s, "_get_bucket", return_value=fake_bucket):
        keys = s.list_windows(segment_id="drive_002")
    assert keys == [WindowKey(segment_id="drive_002", window_idx=0)]


def test_list_windows_empty_bucket() -> None:
    s = FlatMP4Storage(bucket_uri="gs://b/p", cameras=["FRONT"])
    fake_bucket = _fake_bucket([])
    with patch.object(s, "_get_bucket", return_value=fake_bucket):
        assert s.list_windows() == []


# ---------------------------------------------------------------------------
# get_window_video_url
# ---------------------------------------------------------------------------

def test_get_window_video_url_returns_signed_url() -> None:
    s = FlatMP4Storage(bucket_uri="gs://b/p", cameras=["FRONT"])
    fake_bucket = _fake_bucket(["p/drive_001.mp4"], signed_url="https://signed.example/x")
    with patch.object(s, "_get_bucket", return_value=fake_bucket):
        url = s.get_window_video_url("drive_001", 0, camera="FRONT")
    assert url == "https://signed.example/x"


def test_get_window_video_url_rejects_nonzero_window_idx() -> None:
    s = FlatMP4Storage(bucket_uri="gs://b/p", cameras=["FRONT"])
    with pytest.raises(WindowStorageError, match="single-window-per-segment"):
        s.get_window_video_url("drive_001", 1, camera="FRONT")


def test_get_window_video_url_rejects_unknown_camera() -> None:
    s = FlatMP4Storage(bucket_uri="gs://b/p", cameras=["FRONT"])
    with pytest.raises(WindowStorageError, match="not configured"):
        s.get_window_video_url("drive_001", 0, camera="FRONT_LEFT")


def test_get_window_video_url_blob_missing() -> None:
    s = FlatMP4Storage(bucket_uri="gs://b/p", cameras=["FRONT"])
    fake_bucket = _fake_bucket([])  # no blobs at all → bucket.blob().exists() → False
    with patch.object(s, "_get_bucket", return_value=fake_bucket):
        with pytest.raises(WindowStorageError, match="MP4 not found"):
            s.get_window_video_url("missing_drive", 0, camera="FRONT")


def test_get_window_video_url_rejects_zero_byte_mp4() -> None:
    """Pessimistic-review fix: corrupt/truncated MP4s caught before VLM call."""
    s = FlatMP4Storage(bucket_uri="gs://b/p", cameras=["FRONT"])
    bad_blob = _fake_blob("p/drive_001.mp4", exists=True, size=0)
    fake_bucket = MagicMock()
    fake_bucket.blob.return_value = bad_blob
    with patch.object(s, "_get_bucket", return_value=fake_bucket):
        with pytest.raises(WindowStorageError, match="empty or"):
            s.get_window_video_url("drive_001", 0, camera="FRONT")


def test_get_window_video_url_rejects_truncated_mp4() -> None:
    """Anything below 1 KB is treated as corruption."""
    s = FlatMP4Storage(bucket_uri="gs://b/p", cameras=["FRONT"])
    tiny_blob = _fake_blob("p/drive_001.mp4", exists=True, size=512)
    fake_bucket = MagicMock()
    fake_bucket.blob.return_value = tiny_blob
    with patch.object(s, "_get_bucket", return_value=fake_bucket):
        with pytest.raises(WindowStorageError, match="truncated"):
            s.get_window_video_url("drive_001", 0, camera="FRONT")


def test_get_window_video_url_rejects_missing_size_metadata() -> None:
    """blob.size == None (rare GCS edge case) is also caught."""
    s = FlatMP4Storage(bucket_uri="gs://b/p", cameras=["FRONT"])
    weird_blob = _fake_blob("p/drive_001.mp4", exists=True)
    weird_blob.size = None
    fake_bucket = MagicMock()
    fake_bucket.blob.return_value = weird_blob
    with patch.object(s, "_get_bucket", return_value=fake_bucket):
        with pytest.raises(WindowStorageError, match="empty or"):
            s.get_window_video_url("drive_001", 0, camera="FRONT")


def test_get_window_video_url_accepts_healthy_mp4() -> None:
    """Sanity: a normal 100 KB blob with valid metadata passes through."""
    s = FlatMP4Storage(bucket_uri="gs://b/p", cameras=["FRONT"])
    good_blob = _fake_blob("p/drive_001.mp4", exists=True,
                            size=100_000, signed_url="https://signed")
    fake_bucket = MagicMock()
    fake_bucket.blob.return_value = good_blob
    with patch.object(s, "_get_bucket", return_value=fake_bucket):
        url = s.get_window_video_url("drive_001", 0, camera="FRONT")
    assert url == "https://signed"


def test_get_window_video_url_signing_failure_propagates_with_guidance() -> None:
    s = FlatMP4Storage(bucket_uri="gs://b/p", cameras=["FRONT"])
    fake_blob = _fake_blob("p/drive_001.mp4", exists=True)
    fake_blob.generate_signed_url.side_effect = RuntimeError("no private key")
    fake_bucket = MagicMock()
    fake_bucket.blob.return_value = fake_blob
    with patch.object(s, "_get_bucket", return_value=fake_bucket):
        with pytest.raises(WindowStorageError, match="Signed URL generation failed"):
            s.get_window_video_url("drive_001", 0, camera="FRONT")


# ---------------------------------------------------------------------------
# get_window_manifest
# ---------------------------------------------------------------------------

def test_get_window_manifest_synthesizes_correct_cameras() -> None:
    cams = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT"]
    s = FlatMP4Storage(bucket_uri="gs://b/p", cameras=cams)
    m = s.get_window_manifest("drive_001", 0)
    assert isinstance(m, WindowManifest)
    assert m.segment_id == "drive_001"
    assert m.window_idx == 0
    assert m.cameras == cams
    assert m.source_format == "flat_mp4"
    assert m.pose_summary is None


def test_get_window_manifest_rejects_nonzero_window_idx() -> None:
    s = FlatMP4Storage(bucket_uri="gs://b/p", cameras=["FRONT"])
    with pytest.raises(WindowStorageError, match="single-window-per-segment"):
        s.get_window_manifest("drive_001", 5)


def test_get_window_manifest_round_trips_through_json() -> None:
    """Synthesized manifests must round-trip — Encoder reads pose_summary from this."""
    s = FlatMP4Storage(bucket_uri="gs://b/p", cameras=["FRONT"])
    original = s.get_window_manifest("drive_001", 0)
    restored = WindowManifest.from_json(original.to_json())
    assert restored == original
