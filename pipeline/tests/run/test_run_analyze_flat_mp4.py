"""Tests for the new --storage-mode flat_mp4 path in `analyze`.

Argparse coverage + handler behavior. Smoke ends at the FlatMP4Storage
construction call (mocked) — the cross-module integration of FlatMP4Storage
itself is covered in test_flat_mp4.py.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline.interfaces.window import WindowKey
from pipeline.run import _build_encoder, _build_parser, _run_analyze


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------

def test_analyze_storage_mode_defaults_canonical() -> None:
    parser = _build_parser()
    args = parser.parse_args(["analyze", "--bucket", "gs://b", "--output", "/o"])
    assert args.storage_mode == "canonical"
    assert args.cameras is None


def test_analyze_storage_mode_flat_mp4_parses() -> None:
    parser = _build_parser()
    args = parser.parse_args([
        "analyze", "--bucket", "gs://b", "--output", "/o",
        "--storage-mode", "flat_mp4",
        "--cameras", "FRONT,FRONT_LEFT",
    ])
    assert args.storage_mode == "flat_mp4"
    assert args.cameras == "FRONT,FRONT_LEFT"


def test_analyze_storage_mode_rejects_unknown_value() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "analyze", "--bucket", "gs://b", "--output", "/o",
            "--storage-mode", "made_up",
        ])


# ---------------------------------------------------------------------------
# _run_analyze validation: --cameras required iff flat_mp4
# ---------------------------------------------------------------------------

def _flat_args(output: Path, **overrides: object) -> Namespace:
    defaults = dict(
        bucket="gs://b/v",
        output=str(output),
        max_workers=2,
        stub=True,
        no_visual=True,
        cache_root=None,
        sign_as=None,
        storage_mode="flat_mp4",
        cameras="FRONT",
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def test_run_analyze_flat_mp4_without_cameras_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = _run_analyze(_flat_args(tmp_path / "out", cameras=None))
    assert rc == 2
    err = capsys.readouterr().err
    assert "--cameras" in err
    assert "flat_mp4" in err


def test_run_analyze_flat_mp4_empty_cameras_string_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Empty string after --cameras should not silently default to anything."""
    rc = _run_analyze(_flat_args(tmp_path / "out", cameras=""))
    assert rc == 2
    assert "--cameras" in capsys.readouterr().err


def test_run_analyze_flat_mp4_constructs_flatmp4storage(tmp_path: Path) -> None:
    """When --storage-mode flat_mp4 + --cameras is set, FlatMP4Storage is built."""
    fake_storage = MagicMock()
    fake_storage.list_windows.return_value = []  # short-circuit before encoder

    with patch("pipeline.modules.storage.FlatMP4Storage",
               return_value=fake_storage) as fake_ctor:
        rc = _run_analyze(_flat_args(
            tmp_path / "out", cameras="FRONT,FRONT_LEFT,FRONT_RIGHT",
        ))

    fake_ctor.assert_called_once()
    kwargs = fake_ctor.call_args.kwargs
    assert kwargs["bucket_uri"] == "gs://b/v"
    assert kwargs["cameras"] == ["FRONT", "FRONT_LEFT", "FRONT_RIGHT"]
    # The handler short-circuits with rc=2 on empty windows — that's expected here.
    assert rc == 2


def test_run_analyze_canonical_mode_still_uses_window_storage(tmp_path: Path) -> None:
    """When --storage-mode is not flat_mp4, the canonical path is unchanged."""
    fake = MagicMock()
    fake.list_windows.return_value = [WindowKey(segment_id="seg_a", window_idx=0)]
    fake.get_window_video_url.return_value = "https://fake/clip.mp4"
    fake_manifest = MagicMock()
    fake_manifest.pose_summary = None
    fake.get_window_manifest.return_value = fake_manifest

    args = Namespace(
        bucket="gs://b/v", output=str(tmp_path / "out"),
        max_workers=2, stub=True, no_visual=True, cache_root=None, sign_as=None,
        storage_mode="canonical", cameras=None,
    )
    with patch("pipeline.modules.storage.WindowStorage", return_value=fake):
        rc = _run_analyze(args)
    assert rc == 0


# ---------------------------------------------------------------------------
# _build_encoder: cameras kwarg threads through to VisualArm
# ---------------------------------------------------------------------------

def test_build_encoder_stub_passes_cameras_to_visual_arm(tmp_path: Path) -> None:
    """When cameras=[...], VisualArm must be constructed with that camera list."""
    with patch("pipeline.modules.encoder.VisualArm") as fake_visual:
        _build_encoder(
            stub=True, no_visual=False, cache_root=str(tmp_path),
            cameras=["FRONT"],
        )
    fake_visual.assert_called_once()
    assert fake_visual.call_args.kwargs.get("cameras") == ["FRONT"]


def test_build_encoder_no_cameras_uses_visual_arm_default(tmp_path: Path) -> None:
    """When cameras is None, VisualArm is constructed without an explicit cameras kwarg."""
    with patch("pipeline.modules.encoder.VisualArm") as fake_visual:
        _build_encoder(
            stub=True, no_visual=False, cache_root=str(tmp_path), cameras=None,
        )
    fake_visual.assert_called_once()
    assert "cameras" not in fake_visual.call_args.kwargs


def test_build_encoder_no_visual_ignores_cameras(tmp_path: Path) -> None:
    """no_visual=True overrides cameras — no VisualArm constructed at all."""
    with patch("pipeline.modules.encoder.VisualArm") as fake_visual:
        _build_encoder(
            stub=True, no_visual=True, cache_root=str(tmp_path),
            cameras=["FRONT"],
        )
    fake_visual.assert_not_called()
