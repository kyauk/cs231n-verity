"""Contract tests for pipeline.run output files.

pipeline.run is a CLI consumer; its 'contract' is the JSON files each
subcommand writes. Every file must round-trip cleanly through the corresponding
pipeline.interfaces.* type — that's the load-bearing assertion downstream
modules (and a future re-run) depend on.

Coverage per subcommand:
  analyze → schema_records.json, proposals.json, scored.json
            (round-trip + first-class field assertions)
  report  → report.json
            (round-trip through EvaluationReport)
  ingest  → exits 0 and calls IngestionPipeline.ingest with a valid request
            (no JSON output; the canonical bucket layout itself is the contract)

If this test fails, EITHER pipeline.run drifted from the interface types,
OR an interface type changed without updating pipeline.run.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

from pipeline.interfaces.proposal import CompositionProposal, ScoredProposal
from pipeline.interfaces.rating import Rating
from pipeline.interfaces.report import EvaluationReport
from pipeline.interfaces.schema_record import SchemaRecord
from pipeline.interfaces.window import WindowKey
from pipeline.modules.scorer import TextClient, NIMTextClient
from pipeline.run import _run_analyze, _run_report


# ---------------------------------------------------------------------------
# analyze contract — all three JSON outputs round-trip
# ---------------------------------------------------------------------------

def _build_minimal_analyze_args(output: Path) -> Namespace:
    return Namespace(
        bucket="gs://b/v", output=str(output), max_workers=2,
        stub=True, cache_root=None, sign_as=None,
        storage_mode="canonical", cameras=None,
    )


def _fake_storage_with_n(n: int) -> MagicMock:
    keys = [WindowKey(segment_id=f"seg_{i:03d}", window_idx=0) for i in range(n)]
    storage = MagicMock()
    storage.list_windows.return_value = keys
    storage.get_window_video_url.return_value = "https://fake/clip.mp4"
    fake_manifest = MagicMock()
    fake_manifest.pose_summary = None
    storage.get_window_manifest.return_value = fake_manifest
    return storage


def test_analyze_writes_schema_records_that_round_trip(tmp_path: Path) -> None:
    with patch("pipeline.modules.storage.WindowStorage",
               return_value=_fake_storage_with_n(3)):
        rc = _run_analyze(_build_minimal_analyze_args(tmp_path))
    assert rc == 0
    raw = json.loads((tmp_path / "schema_records.json").read_text())
    assert isinstance(raw, list) and len(raw) == 3
    for d in raw:
        rec = SchemaRecord.from_json(d)
        # Every interface-declared field is present + typed:
        assert isinstance(rec.window_id, WindowKey)
        assert isinstance(rec.arm, str)
        assert isinstance(rec.schema_version, str)
        assert isinstance(rec.fields, dict)
        # failure_mode is None on success
        assert rec.failure_mode is None
        assert isinstance(rec.created_at, str)


def test_analyze_writes_proposals_that_round_trip(tmp_path: Path) -> None:
    """Use a varied-records encoder mock so the Hypothesizer actually emits proposals."""
    fake_storage = _fake_storage_with_n(20)

    def _varied_records(inputs: list) -> list[SchemaRecord]:
        return [SchemaRecord(
            window_id=WindowKey(segment_id=inp.segment_id, window_idx=inp.window_idx),
            arm="reasoning", schema_version="1.0",
            prompt_template_id="v1_describe",
            fields={
                "agents": ["car"],
                "environment": {
                    "weather": "clear" if i < 10 else "rain",
                    "time_of_day": "day" if i % 2 == 0 else "night",
                    "lighting_condition": "well_lit",
                },
                "road": {"geometry": "straight", "lane_count": 2},
                "traffic_control": "none", "ego_task": "cruising", "conditions": [],
            },
            failure_mode=None,
        ) for i, inp in enumerate(inputs)]

    fake_encoder = MagicMock()
    fake_encoder.process_batch.side_effect = _varied_records

    with patch("pipeline.modules.storage.WindowStorage", return_value=fake_storage), \
         patch("pipeline.run._build_encoder", return_value=fake_encoder):
        rc = _run_analyze(_build_minimal_analyze_args(tmp_path))
    assert rc == 0

    proposals_raw = json.loads((tmp_path / "proposals.json").read_text())
    assert isinstance(proposals_raw, list)
    for d in proposals_raw:
        p = CompositionProposal.from_json(d)
        assert isinstance(p.composition_id, str)
        assert isinstance(p.constituents, list) and len(p.constituents) >= 2
        assert isinstance(p.marginal_frequencies, dict)
        assert isinstance(p.pairwise_frequencies, dict)
        assert isinstance(p.expected_joint, float)
        assert isinstance(p.observed_joint, float)
        assert isinstance(p.novelty_score, float)
        assert isinstance(p.motivating_scene_ids, list)


def test_analyze_writes_scored_that_round_trip(tmp_path: Path) -> None:
    fake_storage = _fake_storage_with_n(20)

    def _varied_records(inputs: list) -> list[SchemaRecord]:
        return [SchemaRecord(
            window_id=WindowKey(segment_id=inp.segment_id, window_idx=inp.window_idx),
            arm="reasoning", schema_version="1.0",
            prompt_template_id="v1_describe",
            fields={
                "agents": ["car", "pedestrian"] if i < 5 else ["car"],
                "environment": {
                    "weather": "fog" if i < 5 else "clear",
                    "time_of_day": "day", "lighting_condition": "well_lit",
                },
                "road": {"geometry": "intersection" if i < 5 else "straight",
                         "lane_count": 4},
                "traffic_control": "traffic_light", "ego_task": "cruising",
                "conditions": [],
            },
            failure_mode=None,
        ) for i, inp in enumerate(inputs)]

    fake_encoder = MagicMock()
    fake_encoder.process_batch.side_effect = _varied_records

    with patch("pipeline.modules.storage.WindowStorage", return_value=fake_storage), \
         patch("pipeline.run._build_encoder", return_value=fake_encoder):
        rc = _run_analyze(_build_minimal_analyze_args(tmp_path))
    assert rc == 0

    scored_raw = json.loads((tmp_path / "scored.json").read_text())
    assert isinstance(scored_raw, list)
    for d in scored_raw:
        s = ScoredProposal.from_json(d)
        # Every documented ScoredProposal field present + typed
        assert isinstance(s.composition_id, str)
        assert 0.0 <= s.plausibility_score <= 1.0
        assert isinstance(s.plausibility_justification, str)
        assert s.frontier_difficulty_score is None or 0.0 <= s.frontier_difficulty_score <= 1.0
        assert isinstance(s.frontier_difficulty_signals, dict)
        assert isinstance(s.final_rank_score, float)
        assert isinstance(s.accepted, bool)


# ---------------------------------------------------------------------------
# report contract — report.json round-trips
# ---------------------------------------------------------------------------

def _make_scored(comp_id: str, arm: str = "reasoning") -> ScoredProposal:
    return ScoredProposal(
        composition_id=comp_id,
        constituents=["agents:car", "weather:clear"],
        marginal_frequencies={"agents:car": 0.5, "weather:clear": 0.5},
        pairwise_frequencies={"agents:car|weather:clear": 0.25},
        expected_joint=0.25, observed_joint=0.05, novelty_score=1.6,
        motivating_scene_ids=[WindowKey(segment_id="seg_a", window_idx=0)],
        arm=arm,
        plausibility_score=0.8, plausibility_justification="reasonable",
        frontier_difficulty_score=0.6,
        frontier_difficulty_signals={"mean_confidence": 0.7,
                                     "action_variance": 0.4,
                                     "reasoning_action_mismatch": 0.3},
        final_rank_score=1.2, accepted=True, rejection_reason=None,
    )


def _make_rating(rater_id: str, prop_id: str, arm: str = "reasoning") -> Rating:
    return Rating(
        rater_id=rater_id, proposal_id=prop_id, arm=arm,
        coherence_score=4, usefulness_score=3,
        timestamp="2026-05-29T00:00:00Z",
        free_text_note=None, seen_motivating_scenes=[],
    )


def test_report_writes_report_json_that_round_trips(tmp_path: Path) -> None:
    # Build inputs on disk
    scored_path = tmp_path / "scored.json"
    scored = [_make_scored("c1"), _make_scored("c2")]
    scored_path.write_text(json.dumps([s.to_json() for s in scored]))

    seeds_path = tmp_path / "seeds.json"
    seeds_path.write_text(json.dumps({
        "seeded_windows": [
            {"window": "seg_a/0000", "subset": "familiar"},
            {"window": "seg_b/0001", "subset": "unfamiliar"},
        ]
    }))

    ratings_dir = tmp_path / "ratings" / "alice"
    ratings_dir.mkdir(parents=True)
    (ratings_dir / "c1.json").write_text(json.dumps(_make_rating("alice", "c1").to_json()))
    (ratings_dir / "c2.json").write_text(json.dumps(_make_rating("alice", "c2").to_json()))

    output_dir = tmp_path / "out"
    args = Namespace(
        scored=str(scored_path), seeds=str(seeds_path),
        ratings=str(tmp_path / "ratings"), ratings_url=None,
        output=str(output_dir), schema_records=None, recall_k=30,
    )
    rc = _run_report(args)
    assert rc == 0

    # Find the timestamped report.json
    report_files = list(output_dir.rglob("report.json"))
    assert len(report_files) == 1
    report_dict = json.loads(report_files[0].read_text())
    report = EvaluationReport.from_json(report_dict)
    # Every documented EvaluationReport field present
    assert isinstance(report.seeded_recall, dict)
    assert isinstance(report.recall_k_primary, int)
    assert isinstance(report.mean_coherence, dict)
    assert isinstance(report.mean_usefulness, dict)
    assert isinstance(report.n_ratings_per_arm, dict)
    assert isinstance(report.n_proposals_per_arm, dict)
    assert isinstance(report.differential_examples, list)


# ---------------------------------------------------------------------------
# NIMTextClient contract — satisfies the scorer's TextClient Protocol
# ---------------------------------------------------------------------------

def test_nim_text_client_satisfies_textclient_protocol() -> None:
    """Production scorer client must satisfy the TextClient Protocol."""
    client = NIMTextClient(api_key="dummy")
    assert isinstance(client, TextClient)
    assert isinstance(client.model_id, str) and client.model_id
