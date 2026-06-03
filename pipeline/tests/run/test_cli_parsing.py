"""Unit tests for pipeline.run argparse — no module work, just CLI shape."""

from __future__ import annotations

import pytest

from pipeline.run import _build_parser, main


# ---------------------------------------------------------------------------
# --help works for every subcommand
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("argv", [
    ["--help"],
    ["ingest", "--help"],
    ["analyze", "--help"],
    ["report", "--help"],
])
def test_help_exits_zero(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Verity" in out or "subcommand" in out.lower() or "usage" in out.lower()


# ---------------------------------------------------------------------------
# Required subcommand
# ---------------------------------------------------------------------------

def test_missing_subcommand_errors(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code != 0


# ---------------------------------------------------------------------------
# ingest: required flags, defaults
# ---------------------------------------------------------------------------

def test_ingest_parses_minimal_flags() -> None:
    parser = _build_parser()
    args = parser.parse_args([
        "ingest",
        "--source-format", "waymo_parquet",
        "--source-root", "/data/waymo",
        "--bucket", "gs://b/verity",
        "--segments", "all",
    ])
    assert args.subcommand == "ingest"
    assert args.source_format == "waymo_parquet"
    assert args.source_root == "/data/waymo"
    assert args.bucket == "gs://b/verity"
    assert args.segments == "all"
    assert args.force is False
    assert args.window_length_frames == 80
    assert args.target_fps == 10


def test_ingest_rejects_unknown_source_format() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "ingest", "--source-format", "made_up",
            "--source-root", "/x", "--bucket", "gs://b", "--segments", "all",
        ])


def test_ingest_missing_required_flag_exits() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["ingest", "--bucket", "gs://b", "--segments", "all"])


# ---------------------------------------------------------------------------
# analyze: required flags, defaults
# ---------------------------------------------------------------------------

def test_analyze_parses_minimal_flags() -> None:
    parser = _build_parser()
    args = parser.parse_args([
        "analyze",
        "--bucket", "gs://b/verity",
        "--output", "/tmp/out",
    ])
    assert args.subcommand == "analyze"
    assert args.bucket == "gs://b/verity"
    assert args.output == "/tmp/out"
    assert args.max_workers == 8
    assert args.stub is False
    assert args.cache_root is None
    assert args.sign_as is None


def test_analyze_stub_flag() -> None:
    parser = _build_parser()
    args = parser.parse_args([
        "analyze", "--bucket", "gs://b", "--output", "/tmp/o", "--stub",
    ])
    assert args.stub is True


# ---------------------------------------------------------------------------
# report: required flags + mutually-exclusive ratings sources
# ---------------------------------------------------------------------------

def test_report_filesystem_ratings() -> None:
    parser = _build_parser()
    args = parser.parse_args([
        "report",
        "--scored", "/tmp/scored.json",
        "--seeds", "/tmp/seeds.json",
        "--output", "/tmp/o",
        "--ratings", "/tmp/ratings",
    ])
    assert args.ratings == "/tmp/ratings"
    assert args.ratings_url is None
    assert args.recall_k == 30


def test_report_http_ratings() -> None:
    parser = _build_parser()
    args = parser.parse_args([
        "report",
        "--scored", "/tmp/scored.json",
        "--seeds", "/tmp/seeds.json",
        "--output", "/tmp/o",
        "--ratings-url", "http://localhost:8001",
    ])
    assert args.ratings is None
    assert args.ratings_url == "http://localhost:8001"


def test_report_ratings_sources_mutually_exclusive() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "report",
            "--scored", "/s", "--seeds", "/sd", "--output", "/o",
            "--ratings", "/r", "--ratings-url", "http://x",
        ])


def test_report_requires_a_ratings_source() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "report", "--scored", "/s", "--seeds", "/sd", "--output", "/o",
        ])


# All three subcommands now have full coverage in their dedicated test files:
#   test_run_ingest.py, test_run_analyze.py, test_run_report.py
