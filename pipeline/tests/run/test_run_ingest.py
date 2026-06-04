"""Tests for the `ingest` subcommand handler in pipeline.run.

Unit tests cover the small helpers (`_build_source`, `_parse_segments_arg`).
The smoke test wires the full handler with a mocked source + IngestionPipeline
so no GCS / TFRecord I/O happens.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline.modules.storage import IngestionRequest, WindowConfig
from pipeline.run import (
    _build_source,
    _parse_segments_arg,
    _run_ingest,
    main,
)


# ---------------------------------------------------------------------------
# _build_source — format-specific source-root validation
# ---------------------------------------------------------------------------

def test_build_source_parquet_requires_gs_scheme() -> None:
    with pytest.raises(ValueError, match="gs://"):
        _build_source("waymo_parquet", "/local/path")


def test_build_source_parquet_requires_prefix() -> None:
    with pytest.raises(ValueError, match="bucket and prefix"):
        _build_source("waymo_parquet", "gs://bucket-only")


def test_build_source_tfrecord_rejects_gs_scheme() -> None:
    with pytest.raises(ValueError, match="local"):
        _build_source("waymo_tfrecord", "gs://b/p")


def test_build_source_unknown_format_raises() -> None:
    with pytest.raises(ValueError, match="Unknown source format"):
        _build_source("made_up_format", "anything")


# ---------------------------------------------------------------------------
# _parse_segments_arg
# ---------------------------------------------------------------------------

def test_parse_segments_all_calls_list_segments() -> None:
    source = MagicMock()
    source.list_segments.return_value = ["seg_a", "seg_b"]
    assert _parse_segments_arg("all", source) == ["seg_a", "seg_b"]
    source.list_segments.assert_called_once()


def test_parse_segments_all_empty_source_raises() -> None:
    source = MagicMock()
    source.list_segments.return_value = []
    with pytest.raises(ValueError, match="no segments"):
        _parse_segments_arg("all", source)


def test_parse_segments_comma_separated() -> None:
    out = _parse_segments_arg("seg_a, seg_b ,seg_c", source=MagicMock())
    assert out == ["seg_a", "seg_b", "seg_c"]


def test_parse_segments_empty_string_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        _parse_segments_arg("", source=MagicMock())


def test_parse_segments_file(tmp_path: Path) -> None:
    p = tmp_path / "segments.txt"
    p.write_text("seg_one\n\nseg_two\n   \nseg_three\n")
    out = _parse_segments_arg(f"@{p}", source=MagicMock())
    assert out == ["seg_one", "seg_two", "seg_three"]


def test_parse_segments_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _parse_segments_arg(f"@{tmp_path}/does_not_exist.txt", source=MagicMock())


def test_parse_segments_empty_file_raises(tmp_path: Path) -> None:
    p = tmp_path / "empty.txt"
    p.write_text("")
    with pytest.raises(ValueError, match="empty"):
        _parse_segments_arg(f"@{p}", source=MagicMock())


# ---------------------------------------------------------------------------
# _run_ingest — smoke test (full handler with mocked IngestionPipeline)
# ---------------------------------------------------------------------------

def _make_ingest_args(**overrides: object) -> object:
    """Build an argparse.Namespace-like object with sane defaults for ingest."""
    from argparse import Namespace
    defaults = dict(
        source_format="waymo_tfrecord",
        source_root="/local/data",
        bucket="gs://my-bucket/verity",
        segments="all",
        force=False,
        window_length_frames=80,
        target_fps=10,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def test_run_ingest_builds_request_and_calls_pipeline(tmp_path: Path) -> None:
    # Mock the source adapter to bypass tensorflow / pyarrow imports
    fake_source = MagicMock()
    fake_source.list_segments.return_value = ["seg_001", "seg_002"]

    fake_result = MagicMock(
        bucket_uri="gs://my-bucket/verity",
        segments_succeeded=2, segments_failed=0, segments_skipped=0,
        windows_total=20,
    )
    fake_pipeline = MagicMock()
    fake_pipeline.ingest.return_value = fake_result

    with patch("pipeline.run._build_source", return_value=fake_source), \
         patch("pipeline.modules.storage.IngestionPipeline", return_value=fake_pipeline):
        rc = _run_ingest(_make_ingest_args())

    assert rc == 0
    fake_pipeline.ingest.assert_called_once()
    request = fake_pipeline.ingest.call_args.args[0]
    assert isinstance(request, IngestionRequest)
    assert request.segment_ids == ["seg_001", "seg_002"]
    assert request.bucket_uri == "gs://my-bucket/verity"
    assert request.force_reingest is False
    assert isinstance(request.window_config, WindowConfig)
    assert request.window_config.length_frames == 80
    assert request.window_config.stride_frames == 80  # non-overlapping
    assert request.window_config.target_fps == 10
    assert request.source is fake_source


def test_run_ingest_bad_source_root_returns_2(capsys: pytest.CaptureFixture[str]) -> None:
    rc = _run_ingest(_make_ingest_args(
        source_format="waymo_parquet",
        source_root="/not/gs",  # invalid: parquet requires gs://
    ))
    assert rc == 2
    assert "gs://" in capsys.readouterr().err


def test_run_ingest_source_unreachable_returns_2(capsys: pytest.CaptureFixture[str]) -> None:
    from pipeline.modules.storage import SourceUnreachableError

    fake_source = MagicMock()
    fake_source.list_segments.return_value = ["seg_x"]
    fake_pipeline = MagicMock()
    fake_pipeline.ingest.side_effect = SourceUnreachableError("gs://b/p", "no access")

    with patch("pipeline.run._build_source", return_value=fake_source), \
         patch("pipeline.modules.storage.IngestionPipeline", return_value=fake_pipeline):
        rc = _run_ingest(_make_ingest_args())

    assert rc == 2
    assert "fatal" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# End-to-end via main() — argparse + handler
# ---------------------------------------------------------------------------

def test_main_invokes_ingest_handler() -> None:
    with patch("pipeline.run._run_ingest", return_value=0) as h:
        rc = main([
            "ingest",
            "--source-format", "waymo_tfrecord",
            "--source-root", "/data",
            "--bucket", "gs://b/v",
            "--segments", "all",
        ])
    assert rc == 0
    h.assert_called_once()
