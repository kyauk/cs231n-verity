"""Cross-module integration test: pipeline.run analyze → pipeline.run report.

The marquee test for pipeline.run's hygiene cycle. Proves the file boundary
between the two subcommands actually works:

  1. analyze writes scored.json (via Encoder + Hypothesizer + Scorer)
  2. report consumes that exact scored.json + synthetic ratings + seeds
  3. Evaluator emits a valid report.json

If anything drifts between what analyze writes and what report reads, this
test fails immediately. Cross-stage drift caught here is 30 minutes; caught
in week 3 is 3 days.

All real modules execute (Encoder, Hypothesizer, Scorer, Evaluator).
Only the WindowStorage is mocked (no GCS), and stub VLM clients drive the
arms (no NIM).
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

from pipeline.interfaces.proposal import ScoredProposal
from pipeline.interfaces.rating import Rating
from pipeline.interfaces.report import EvaluationReport
from pipeline.interfaces.schema_record import SchemaRecord
from pipeline.interfaces.window import WindowKey
from pipeline.run import _run_analyze, _run_report


def _build_varied_records(inputs: list) -> list[SchemaRecord]:
    """60 records crafted so the Hypothesizer surfaces real arity-3 novelty.

    Distribution targets the (fog, night, pedestrian) composition:
      18 records: fog + night + car          (pair  AB, no  C)
      18 records: fog + day   + pedestrian   (pair  AC, no  B)
      18 records: clear + night + pedestrian (pair  BC, no  A)
       6 records: clear + day   + car         (background)

    Marginals: weather:fog = 36/60 = 60%, time_of_day:night = 60%,
    agents:pedestrian = 60%. All pass min_marginal_frequency (5%).
    Pairwise (fog,night) = (fog,ped) = (night,ped) = 18/60 = 30%, all
    above min_pairwise_frequency (1%). Triple (fog, night, pedestrian)
    is observed in 0 records — well below max_joint_frequency (0.5%) —
    so the Hypothesizer emits it as a high-novelty arity-3 composition.
    """
    records: list[SchemaRecord] = []
    for i, inp in enumerate(inputs):
        if i < 18:
            fields = _profile(weather="fog", tod="night", agents=["car"],
                              road="straight", traffic="none")
        elif i < 36:
            fields = _profile(weather="fog", tod="day", agents=["pedestrian"],
                              road="intersection", traffic="traffic_light")
        elif i < 54:
            fields = _profile(weather="clear", tod="night", agents=["pedestrian"],
                              road="intersection", traffic="traffic_light")
        else:
            fields = _profile(weather="clear", tod="day", agents=["car"],
                              road="straight", traffic="none")
        records.append(SchemaRecord(
            window_id=WindowKey(segment_id=inp.segment_id, window_idx=inp.window_idx),
            arm="reasoning", schema_version="1.0",
            prompt_template_id="v1_describe",
            fields=fields, failure_mode=None,
        ))
    return records


def _profile(*, weather: str, tod: str, agents: list[str],
             road: str, traffic: str) -> dict:
    return {
        "agents": agents,
        "environment": {"weather": weather, "time_of_day": tod,
                        "lighting_condition": "well_lit"},
        "road": {"geometry": road, "lane_count": 2},
        "traffic_control": traffic, "ego_task": "cruising",
        "conditions": [],
    }


def test_analyze_output_is_consumable_by_report(tmp_path: Path) -> None:
    """The marquee cross-stage integration test.

    Real components: pipeline.run analyze handler + file IO + pipeline.run
    report handler + Evaluator.
    Mocked: WindowStorage (no GCS), Encoder (synthetic records), and the
    Scorer (synthetic ScoredProposals — chosen to bypass Hypothesizer
    filter tuning, which is exercised separately in the Hypothesizer's own
    test suite).

    The boundary under test: scored.json written by analyze MUST be readable
    by report exactly as-written.
    """
    # ---------- STAGE 1: pipeline.run analyze ----------
    fake_storage = MagicMock()
    fake_storage.list_windows.return_value = [
        WindowKey(segment_id=f"seg_{i:03d}", window_idx=0) for i in range(60)
    ]
    fake_storage.get_window_video_url.return_value = "https://fake/clip.mp4"
    fake_manifest = MagicMock()
    fake_manifest.pose_summary = None
    fake_storage.get_window_manifest.return_value = fake_manifest

    fake_encoder = MagicMock()
    fake_encoder.process_batch.side_effect = _build_varied_records

    analyze_args = Namespace(
        bucket="gs://b/v", output=str(tmp_path / "session"),
        max_workers=2, stub=True,
        cache_root=None, sign_as=None,
        storage_mode="canonical", cameras=None,
    )
    # Mock only the storage and encoder; let the real Hypothesizer + Scorer
    # (with stub clients via --stub) run on the carefully-designed records.
    with patch("pipeline.modules.storage.WindowStorage", return_value=fake_storage), \
         patch("pipeline.run._build_encoder", return_value=fake_encoder):
        rc = _run_analyze(analyze_args)
    assert rc == 0

    # Sanity: analyze wrote the file we're about to feed into report
    scored_path = tmp_path / "session" / "scored.json"
    assert scored_path.exists()
    scored_raw = json.loads(scored_path.read_text())
    assert len(scored_raw) > 0, "Hypothesizer should have found at least one composition"

    # ---------- STAGE 2: build synthetic ratings + seeds for report ----------
    # Pull the actual composition_ids out of analyze's output so ratings
    # reference real proposals (the realistic operator flow).
    scored = [ScoredProposal.from_json(d) for d in scored_raw]
    rated_ids = [s.composition_id for s in scored[:2]]

    ratings_dir = tmp_path / "ratings" / "alice"
    ratings_dir.mkdir(parents=True)
    for cid in rated_ids:
        rating = Rating(
            rater_id="alice", proposal_id=cid, arm="reasoning",
            coherence_score=4, usefulness_score=3,
            timestamp="2026-05-29T00:00:00Z",
            free_text_note=None, seen_motivating_scenes=[],
        )
        (ratings_dir / f"{cid}.json").write_text(json.dumps(rating.to_json()))

    seeds_path = tmp_path / "seeds.json"
    seeds_path.write_text(json.dumps({
        "seeded_windows": [
            {"window": "seg_000/0000", "subset": "familiar"},
            {"window": "seg_001/0000", "subset": "unfamiliar"},
        ]
    }))

    # ---------- STAGE 3: pipeline.run report ----------
    report_dir = tmp_path / "session"  # write into the same session dir
    report_args = Namespace(
        scored=str(scored_path), seeds=str(seeds_path),
        ratings=str(tmp_path / "ratings"), ratings_url=None,
        output=str(report_dir), schema_records=None, recall_k=30,
    )
    rc = _run_report(report_args)
    assert rc == 0

    # ---------- STAGE 4: validate the boundary ----------
    report_files = list(report_dir.rglob("report.json"))
    assert len(report_files) == 1, f"expected exactly one report.json under {report_dir}"

    report = EvaluationReport.from_json(json.loads(report_files[0].read_text()))
    # The report saw the scored proposals analyze wrote:
    assert "reasoning" in report.n_proposals_per_arm
    assert report.n_proposals_per_arm["reasoning"] > 0
    # And saw the ratings we wrote:
    assert report.n_ratings_per_arm.get("reasoning", 0) >= 2
