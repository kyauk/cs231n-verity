"""Smoke / cross-module integration tests for `analyze` subcommand.

Runs the full Storage→Encoder→Hypothesizer→Scorer chain end-to-end with stub
clients on a mocked WindowStorage. No network, no GCS, no real VLM.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline.interfaces.schema_record import SchemaRecord
from pipeline.interfaces.proposal import CompositionProposal, ScoredProposal
from pipeline.interfaces.window import WindowKey
from pipeline.run import _build_encoder, _build_scorer, _run_analyze


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _fake_storage(window_keys: list[WindowKey]) -> MagicMock:
    storage = MagicMock()
    storage.list_windows.return_value = window_keys
    storage.get_window_video_url.return_value = "https://fake-signed-url/clip.mp4"
    fake_manifest = MagicMock()
    fake_manifest.pose_summary = None  # critical: must be None or str, not MagicMock
    storage.get_window_manifest.return_value = fake_manifest
    return storage


def _analyze_args(output: Path, **overrides: object) -> Namespace:
    defaults = dict(
        bucket="gs://b/v",
        output=str(output),
        max_workers=2,
        stub=True,
        cache_root=None,
        sign_as=None,
        storage_mode="canonical",
        cameras=None,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


# ---------------------------------------------------------------------------
# _build_encoder / _build_scorer — production path errors
# ---------------------------------------------------------------------------

def test_build_encoder_without_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
        _build_encoder(stub=False, cache_root=None)


def test_build_scorer_without_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
        _build_scorer(stub=False, cache_root=None)


def test_build_encoder_stub_path(tmp_path: Path) -> None:
    enc = _build_encoder(stub=True, cache_root=str(tmp_path))
    assert enc is not None  # constructs cleanly


def test_build_scorer_stub_path(tmp_path: Path) -> None:
    sc = _build_scorer(stub=True, cache_root=str(tmp_path))
    assert sc is not None


# ---------------------------------------------------------------------------
# _run_analyze — end-to-end smoke with mocked storage
# ---------------------------------------------------------------------------

def test_run_analyze_writes_three_files_and_returns_zero(tmp_path: Path) -> None:
    keys = [WindowKey(segment_id=f"seg_{i:03d}", window_idx=0) for i in range(3)]
    fake = _fake_storage(keys)

    with patch("pipeline.modules.storage.WindowStorage", return_value=fake):
        rc = _run_analyze(_analyze_args(tmp_path / "out"))

    assert rc == 0
    out = tmp_path / "out"
    assert (out / "schema_records.json").exists()
    assert (out / "proposals.json").exists()
    assert (out / "scored.json").exists()


def test_run_analyze_records_round_trip(tmp_path: Path) -> None:
    """Every SchemaRecord written must round-trip through from_json."""
    keys = [WindowKey(segment_id="seg_a", window_idx=i) for i in range(3)]
    fake = _fake_storage(keys)

    with patch("pipeline.modules.storage.WindowStorage", return_value=fake):
        rc = _run_analyze(_analyze_args(tmp_path / "out"))

    assert rc == 0
    raw = json.loads((tmp_path / "out" / "schema_records.json").read_text())
    assert len(raw) == 3
    records = [SchemaRecord.from_json(d) for d in raw]
    for r in records:
        assert r.arm == "reasoning"
        assert r.window_id in keys


def test_run_analyze_no_windows_returns_2(tmp_path: Path,
                                            capsys: pytest.CaptureFixture[str]) -> None:
    fake = _fake_storage([])  # storage has no windows
    with patch("pipeline.modules.storage.WindowStorage", return_value=fake):
        rc = _run_analyze(_analyze_args(tmp_path / "out"))
    assert rc == 2
    assert "no windows found" in capsys.readouterr().err


def test_run_analyze_window_storage_error_returns_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Pessimistic-review fix: WindowStorageError from list_windows must NOT
    surface as an uncaught traceback. Customer sees a clean error with
    actionable diagnostic text instead."""
    from pipeline.interfaces.errors import WindowStorageError

    fake = MagicMock()
    fake.list_windows.side_effect = WindowStorageError("gs://b/v", "ADC expired")
    with patch("pipeline.modules.storage.WindowStorage", return_value=fake):
        rc = _run_analyze(_analyze_args(tmp_path / "out"))
    assert rc == 2
    err = capsys.readouterr().err
    assert "cannot list windows" in err
    assert "gcloud auth application-default login" in err


def test_run_analyze_proposals_round_trip_when_present(tmp_path: Path) -> None:
    """When the Hypothesizer emits proposals, they must round-trip."""
    # Build a wide spread of records by patching the Encoder to return
    # synthetic records with varied fields so the Hypothesizer has material.
    fake_storage = _fake_storage(
        [WindowKey(segment_id="seg_a", window_idx=i) for i in range(20)]
    )

    # Inject varied records: 10 day/clear + 10 night/rain → at least one
    # cross-axis composition should land below the joint-frequency threshold.
    def _varied_records(inputs: list) -> list[SchemaRecord]:
        out: list[SchemaRecord] = []
        for i, inp in enumerate(inputs):
            tod = "day" if i % 2 == 0 else "night"
            weather = "clear" if i < 10 else "rain"
            out.append(SchemaRecord(
                window_id=WindowKey(segment_id=inp.segment_id, window_idx=inp.window_idx),
                arm="reasoning",
                schema_version="1.0",
                prompt_template_id="v1_describe",
                fields={
                    "agents": ["car"],
                    "environment": {"weather": weather, "time_of_day": tod,
                                    "lighting_condition": "well_lit"},
                    "road": {"geometry": "straight", "lane_count": 2},
                    "traffic_control": "none",
                    "ego_task": "cruising",
                    "conditions": [],
                },
                failure_mode=None,
            ))
        return out

    fake_encoder = MagicMock()
    fake_encoder.process_batch.side_effect = _varied_records

    with patch("pipeline.modules.storage.WindowStorage", return_value=fake_storage), \
         patch("pipeline.run._build_encoder", return_value=fake_encoder):
        rc = _run_analyze(_analyze_args(tmp_path / "out"))

    assert rc == 0
    proposals_raw = json.loads((tmp_path / "out" / "proposals.json").read_text())
    for d in proposals_raw:
        proposal = CompositionProposal.from_json(d)
        assert proposal.composition_id
        assert len(proposal.constituents) >= 2

    scored_raw = json.loads((tmp_path / "out" / "scored.json").read_text())
    for d in scored_raw:
        sp = ScoredProposal.from_json(d)
        assert 0.0 <= sp.plausibility_score <= 1.0
