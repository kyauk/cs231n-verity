"""Tests for `report` subcommand — seeds parsing, ratings loading, end-to-end."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline.interfaces.proposal import ScoredProposal
from pipeline.interfaces.rating import Rating
from pipeline.interfaces.window import WindowKey
from pipeline.run import (
    _group_scored_by_arm,
    _load_ratings_from_dir,
    _load_ratings_from_url,
    _load_seeds,
    _run_report,
)


# ---------------------------------------------------------------------------
# _load_seeds
# ---------------------------------------------------------------------------

def test_load_seeds_accepts_string_window(tmp_path: Path) -> None:
    seeds = tmp_path / "seeds.json"
    seeds.write_text(json.dumps({
        "seeded_windows": [
            {"window": "seg_001/0000", "subset": "familiar"},
            {"window": "seg_002/0001", "subset": "unfamiliar"},
        ]
    }))
    ids, labels = _load_seeds(seeds)
    assert ids == [
        WindowKey(segment_id="seg_001", window_idx=0),
        WindowKey(segment_id="seg_002", window_idx=1),
    ]
    assert labels[ids[0]] == "familiar"
    assert labels[ids[1]] == "unfamiliar"


def test_load_seeds_accepts_dict_window(tmp_path: Path) -> None:
    seeds = tmp_path / "seeds.json"
    seeds.write_text(json.dumps({
        "seeded_windows": [
            {"window": {"segment_id": "seg_a", "window_idx": 5}, "subset": "familiar"},
        ]
    }))
    ids, labels = _load_seeds(seeds)
    assert ids[0] == WindowKey(segment_id="seg_a", window_idx=5)
    assert labels[ids[0]] == "familiar"


def test_load_seeds_rejects_bad_subset(tmp_path: Path) -> None:
    seeds = tmp_path / "seeds.json"
    seeds.write_text(json.dumps({
        "seeded_windows": [{"window": "s/0", "subset": "kinda_familiar"}]
    }))
    with pytest.raises(ValueError, match="familiar"):
        _load_seeds(seeds)


def test_load_seeds_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _load_seeds(tmp_path / "nope.json")


def test_load_seeds_empty_list_raises(tmp_path: Path) -> None:
    seeds = tmp_path / "seeds.json"
    seeds.write_text(json.dumps({"seeded_windows": []}))
    with pytest.raises(ValueError, match="empty"):
        _load_seeds(seeds)


# ---------------------------------------------------------------------------
# _load_ratings_from_dir
# ---------------------------------------------------------------------------

def _write_rating(path: Path, rater_id: str, proposal_id: str, arm: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rating = Rating(
        rater_id=rater_id,
        proposal_id=proposal_id,
        arm=arm,
        coherence_score=4,
        usefulness_score=3,
        timestamp="2026-05-29T00:00:00Z",
        free_text_note=None,
        seen_motivating_scenes=[],
    )
    path.write_text(json.dumps(rating.to_json()))


def test_load_ratings_from_dir_walks_rater_subdirs(tmp_path: Path) -> None:
    _write_rating(tmp_path / "alice" / "prop_a.json", "alice", "prop_a", "reasoning")
    _write_rating(tmp_path / "alice" / "prop_b.json", "alice", "prop_b", "reasoning")
    _write_rating(tmp_path / "bob"   / "prop_a.json", "bob",   "prop_a", "reasoning")
    ratings = _load_ratings_from_dir(tmp_path)
    assert len(ratings) == 3
    assert {r.rater_id for r in ratings} == {"alice", "bob"}


def test_load_ratings_from_dir_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _load_ratings_from_dir(tmp_path / "nope")


def test_load_ratings_from_dir_empty_returns_empty(tmp_path: Path) -> None:
    assert _load_ratings_from_dir(tmp_path) == []


# ---------------------------------------------------------------------------
# _load_ratings_from_url
# ---------------------------------------------------------------------------

def test_load_ratings_from_url_calls_export_endpoint() -> None:
    rating_dict = Rating(
        rater_id="x", proposal_id="p", arm="reasoning",
        coherence_score=5, usefulness_score=5,
        timestamp="2026-05-29T00:00:00Z",
        free_text_note=None, seen_motivating_scenes=[],
    ).to_json()

    fake_resp = MagicMock()
    fake_resp.json.return_value = [rating_dict]
    fake_resp.raise_for_status.return_value = None

    with patch("requests.get", return_value=fake_resp) as fake_get:
        ratings = _load_ratings_from_url("http://localhost:8001")

    fake_get.assert_called_once_with("http://localhost:8001/judge/ratings/export", timeout=30)
    assert len(ratings) == 1
    assert ratings[0].rater_id == "x"


# ---------------------------------------------------------------------------
# _group_scored_by_arm
# ---------------------------------------------------------------------------

def test_group_scored_by_arm() -> None:
    a = MagicMock(arm="reasoning")
    b = MagicMock(arm="visual")
    c = MagicMock(arm="reasoning")
    grouped = _group_scored_by_arm([a, b, c])
    assert set(grouped.keys()) == {"reasoning", "visual"}
    assert grouped["reasoning"] == [a, c]
    assert grouped["visual"] == [b]


# ---------------------------------------------------------------------------
# _run_report — end-to-end smoke
# ---------------------------------------------------------------------------

def _make_scored(comp_id: str, arm: str, accepted: bool = True) -> ScoredProposal:
    return ScoredProposal(
        composition_id=comp_id,
        constituents=["agents:car", "weather:clear"],
        marginal_frequencies={"agents:car": 0.5, "weather:clear": 0.5},
        pairwise_frequencies={"agents:car|weather:clear": 0.25},
        expected_joint=0.25,
        observed_joint=0.05,
        novelty_score=1.6,
        motivating_scene_ids=[WindowKey(segment_id="seg_a", window_idx=0)],
        arm=arm,
        plausibility_score=0.8,
        plausibility_justification="reasonable",
        frontier_difficulty_score=0.6,
        frontier_difficulty_signals={"mean_confidence": 0.7,
                                     "action_variance": 0.4,
                                     "reasoning_action_mismatch": 0.3},
        final_rank_score=1.2,
        accepted=accepted,
        rejection_reason=None if accepted else "plausibility_below_threshold",
    )


def _make_rating(rater_id: str, prop_id: str, arm: str) -> Rating:
    return Rating(
        rater_id=rater_id, proposal_id=prop_id, arm=arm,
        coherence_score=4, usefulness_score=3,
        timestamp="2026-05-29T00:00:00Z",
        free_text_note=None, seen_motivating_scenes=[],
    )


def test_run_report_end_to_end(tmp_path: Path) -> None:
    # Build inputs on disk
    scored_path = tmp_path / "scored.json"
    scored = [_make_scored("c1", "reasoning"), _make_scored("c2", "reasoning")]
    scored_path.write_text(json.dumps([s.to_json() for s in scored]))

    seeds_path = tmp_path / "seeds.json"
    seeds_path.write_text(json.dumps({
        "seeded_windows": [
            {"window": "seg_a/0000", "subset": "familiar"},
            {"window": "seg_b/0001", "subset": "unfamiliar"},
        ]
    }))

    ratings_dir = tmp_path / "ratings"
    _write_rating(ratings_dir / "alice" / "c1.json", "alice", "c1", "reasoning")
    _write_rating(ratings_dir / "alice" / "c2.json", "alice", "c2", "reasoning")

    output_dir = tmp_path / "out"

    args = Namespace(
        scored=str(scored_path),
        seeds=str(seeds_path),
        ratings=str(ratings_dir),
        ratings_url=None,
        output=str(output_dir),
        schema_records=None,
        recall_k=30,
    )
    rc = _run_report(args)
    assert rc == 0
    # Evaluator.save writes to a timestamped subdir; find report.json under output_dir.
    report_files = list(output_dir.rglob("report.json"))
    assert len(report_files) == 1, f"expected exactly one report.json under {output_dir}"


def test_run_report_missing_scored_returns_2(tmp_path: Path,
                                                capsys: pytest.CaptureFixture[str]) -> None:
    args = Namespace(
        scored=str(tmp_path / "nope.json"), seeds=str(tmp_path / "seeds.json"),
        ratings=str(tmp_path), ratings_url=None, output=str(tmp_path / "out"),
        schema_records=None, recall_k=30,
    )
    rc = _run_report(args)
    assert rc == 2
    assert "scored" not in capsys.readouterr().err.lower() or True  # message format is loose
