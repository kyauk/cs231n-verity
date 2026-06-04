"""Integration test: Module 5 (Judge UI) → Module 6 (Evaluation).

Module 6 does not exist yet. This test uses a stub consumer that exercises
every field Module 6 will need from the Rating objects exported by Module 5.

TODO: replace stub_evaluation_consumer with the real Module 6 import when built.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from pipeline.interfaces.rating import Rating
from pipeline.interfaces.window import WindowKey

# ---------------------------------------------------------------------------
# Stub: Module 6 consumer
# Exercises every field of Rating that the Evaluation module will use.
# TODO: replace when Module 6 (Evaluation) is built.
# ---------------------------------------------------------------------------

def stub_evaluation_consumer(ratings: list[Rating]) -> dict[str, Any]:
    """Simulates what Module 6 will do with the exported ratings.

    Module 6 needs:
      - rater_id          → group ratings by rater for inter-rater agreement
      - proposal_id       → join with proposals for per-arm breakdown
      - arm               → compute per-arm mean scores
      - coherence_score   → compute mean_coherence per arm
      - usefulness_score  → compute mean_usefulness per arm
      - timestamp         → verify session ordering
      - seen_motivating_scenes → compute scene-view coverage metric

    Returns a summary dict that proves all fields were consumed without error.
    """
    assert len(ratings) > 0, "stub received empty ratings list"

    by_arm: dict[str, list[Rating]] = {}
    by_rater: dict[str, list[Rating]] = {}

    for r in ratings:
        # rater_id: str
        assert isinstance(r.rater_id, str) and r.rater_id, \
            f"rater_id must be non-empty str, got {r.rater_id!r}"

        # proposal_id: str
        assert isinstance(r.proposal_id, str) and r.proposal_id, \
            f"proposal_id must be non-empty str, got {r.proposal_id!r}"

        # arm: str — must be one of the known values
        assert r.arm in ("reasoning", "visual"), \
            f"arm must be 'reasoning' or 'visual', got {r.arm!r}"

        # coherence_score: int 1-5
        assert isinstance(r.coherence_score, int) and 1 <= r.coherence_score <= 5, \
            f"coherence_score out of range: {r.coherence_score}"

        # usefulness_score: int 1-5
        assert isinstance(r.usefulness_score, int) and 1 <= r.usefulness_score <= 5, \
            f"usefulness_score out of range: {r.usefulness_score}"

        # timestamp: parseable ISO-8601
        from datetime import datetime
        datetime.fromisoformat(r.timestamp)

        # free_text_note: str | None
        assert r.free_text_note is None or isinstance(r.free_text_note, str), \
            f"free_text_note wrong type: {type(r.free_text_note)}"

        # seen_motivating_scenes: list[WindowKey]
        assert isinstance(r.seen_motivating_scenes, list)
        for wk in r.seen_motivating_scenes:
            assert isinstance(wk, WindowKey), f"scene must be WindowKey, got {type(wk)}"
            assert isinstance(wk.segment_id, str)
            assert isinstance(wk.window_idx, int)

        # Group for downstream computations
        by_arm.setdefault(r.arm, []).append(r)
        by_rater.setdefault(r.rater_id, []).append(r)

    # Compute per-arm means (what Module 6 will do for mean_coherence/mean_usefulness)
    per_arm_means = {}
    for arm, arm_ratings in by_arm.items():
        mean_coh = sum(r.coherence_score for r in arm_ratings) / len(arm_ratings)
        mean_use = sum(r.usefulness_score for r in arm_ratings) / len(arm_ratings)
        per_arm_means[arm] = {"mean_coherence": mean_coh, "mean_usefulness": mean_use}

    return {
        "n_ratings": len(ratings),
        "arms_seen": sorted(by_arm.keys()),
        "raters_seen": sorted(by_rater.keys()),
        "per_arm_means": per_arm_means,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PROPOSALS = [
    {
        "composition_id": "integ_001",
        "constituents": ["weather:rain", "ego_task:turning"],
        "marginal_frequencies": {"weather:rain": 0.12, "ego_task:turning": 0.18},
        "pairwise_frequencies": {"ego_task:turning|weather:rain": 0.02},
        "expected_joint": 0.0216, "observed_joint": 0.001, "novelty_score": 3.07,
        "motivating_scene_ids": [{"segment_id": "seg001", "window_idx": 2}],
        "arm": "reasoning",
        "plausibility_score": 0.85, "plausibility_justification": "Coherent",
        "frontier_difficulty_score": 0.72, "frontier_difficulty_signals": {},
        "final_rank_score": 2.8, "accepted": True, "rejection_reason": None,
    },
    {
        "composition_id": "integ_002",
        "constituents": ["agents:pedestrian", "conditions:construction"],
        "marginal_frequencies": {"agents:pedestrian": 0.2, "conditions:construction": 0.08},
        "pairwise_frequencies": {"agents:pedestrian|conditions:construction": 0.015},
        "expected_joint": 0.016, "observed_joint": 0.002, "novelty_score": 2.1,
        "motivating_scene_ids": [],
        "arm": "visual",
        "plausibility_score": 0.76, "plausibility_justification": "Plausible",
        "frontier_difficulty_score": None, "frontier_difficulty_signals": {},
        "final_rank_score": 1.9, "accepted": True, "rejection_reason": None,
    },
]


@pytest.fixture()
def client_and_ratings_dir(tmp_path: Path):
    proposals_file = tmp_path / "proposals.json"
    proposals_file.write_text(json.dumps(PROPOSALS))
    ratings_dir = tmp_path / "ratings"

    overrides = {
        "JUDGE_PROPOSALS_PATH": str(proposals_file),
        "JUDGE_RATINGS_DIR": str(ratings_dir),
        "JUDGE_BUCKET_URI": "",
    }
    old = {k: os.environ.get(k) for k in overrides}
    os.environ.update(overrides)

    import pipeline.modules.judge_ui.server as srv
    import pipeline.modules.judge_ui.config as cfg
    importlib.reload(cfg); importlib.reload(srv)
    srv._proposals.clear(); srv._accepted_ids.clear(); srv._storage = None
    srv._load_proposals()

    client = TestClient(srv.app, raise_server_exceptions=True)
    yield client, srv

    for k, v in old.items():
        if v is None: os.environ.pop(k, None)
        else: os.environ[k] = v


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestJudgeUIToEvaluation:

    def test_ratings_flow_into_stub_consumer(self, client_and_ratings_dir):
        """Full flow: submit ratings via Module 5, export, feed to Module 6 stub."""
        client, _ = client_and_ratings_dir

        # Two raters rate both proposals (simulates a real rating session)
        for rater in ("alice", "bob"):
            client.post("/judge/ratings", json={
                "rater_id": rater, "proposal_id": "integ_001",
                "coherence_score": 4, "usefulness_score": 5,
                "seen_motivating_scenes": [{"segment_id": "seg001", "window_idx": 2}],
            })
            client.post("/judge/ratings", json={
                "rater_id": rater, "proposal_id": "integ_002",
                "coherence_score": 3, "usefulness_score": 4,
                "seen_motivating_scenes": [],
            })

        # Export via the Module 6 boundary endpoint
        exported = client.get("/judge/ratings/export").json()
        assert len(exported) == 4

        # Deserialize through the Rating interface (what Module 6 will do)
        ratings = [Rating.from_json(r) for r in exported]

        # Feed to stub consumer — exercises every field
        summary = stub_evaluation_consumer(ratings)

        assert summary["n_ratings"] == 4
        assert "reasoning" in summary["arms_seen"]
        assert "visual" in summary["arms_seen"]
        assert sorted(summary["raters_seen"]) == ["alice", "bob"]
        assert "reasoning" in summary["per_arm_means"]
        assert "visual" in summary["per_arm_means"]

    def test_both_arms_present_in_export(self, client_and_ratings_dir):
        """Export must contain ratings for both arms (Module 6 needs per-arm breakdown)."""
        client, _ = client_and_ratings_dir

        client.post("/judge/ratings", json={
            "rater_id": "carol", "proposal_id": "integ_001",
            "coherence_score": 5, "usefulness_score": 4,
        })
        client.post("/judge/ratings", json={
            "rater_id": "carol", "proposal_id": "integ_002",
            "coherence_score": 3, "usefulness_score": 3,
        })

        exported = client.get("/judge/ratings/export").json()
        ratings = [Rating.from_json(r) for r in exported]
        arms = {r.arm for r in ratings}
        assert "reasoning" in arms, "reasoning arm missing from export"
        assert "visual" in arms, "visual arm missing from export"

    def test_window_keys_in_seen_scenes_survive_boundary(self, client_and_ratings_dir):
        """WindowKey objects in seen_motivating_scenes must survive the Module 5→6 boundary."""
        client, _ = client_and_ratings_dir

        client.post("/judge/ratings", json={
            "rater_id": "dave", "proposal_id": "integ_001",
            "coherence_score": 4, "usefulness_score": 4,
            "seen_motivating_scenes": [
                {"segment_id": "seg001", "window_idx": 2},
                {"segment_id": "seg001", "window_idx": 3},
            ],
        })

        exported = client.get("/judge/ratings/export").json()
        rating = Rating.from_json(exported[0])
        assert len(rating.seen_motivating_scenes) == 2
        for wk in rating.seen_motivating_scenes:
            assert isinstance(wk, WindowKey)

    def test_session_export_consistent(self, client_and_ratings_dir):
        """Session endpoint and export endpoint must agree on rated proposal IDs."""
        client, _ = client_and_ratings_dir

        for pid in ("integ_001", "integ_002"):
            client.post("/judge/ratings", json={
                "rater_id": "eve", "proposal_id": pid,
                "coherence_score": 3, "usefulness_score": 3,
            })

        session = client.get("/judge/session/eve").json()
        exported = client.get("/judge/ratings/export").json()

        session_pids = set(session["rated_proposal_ids"])
        export_pids = {r["proposal_id"] for r in exported if r["rater_id"] == "eve"}
        assert session_pids == export_pids, \
            f"Session/export mismatch: session={session_pids} export={export_pids}"
