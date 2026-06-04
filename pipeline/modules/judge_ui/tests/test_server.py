"""Unit and contract tests for Module 5: Judge UI server.

Run with:
    python -m pytest pipeline/modules/judge_ui/tests/ -v
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURE_PROPOSALS = [
    {
        "composition_id": "proposal_a",
        "constituents": ["weather:rain", "ego_task:turning"],
        "marginal_frequencies": {"weather:rain": 0.12, "ego_task:turning": 0.18},
        "pairwise_frequencies": {"ego_task:turning|weather:rain": 0.02},
        "expected_joint": 0.0216,
        "observed_joint": 0.001,
        "novelty_score": 3.07,
        "motivating_scene_ids": [{"segment_id": "seg001", "window_idx": 2}],
        "arm": "reasoning",
        "plausibility_score": 0.85,
        "plausibility_justification": "Rain + turning is physically coherent",
        "frontier_difficulty_score": 0.72,
        "frontier_difficulty_signals": {"mean_confidence": 0.4},
        "final_rank_score": 2.8,
        "accepted": True,
        "rejection_reason": None,
    },
    {
        "composition_id": "proposal_b",
        "constituents": ["agents:pedestrian", "conditions:construction"],
        "marginal_frequencies": {"agents:pedestrian": 0.25, "conditions:construction": 0.08},
        "pairwise_frequencies": {"agents:pedestrian|conditions:construction": 0.015},
        "expected_joint": 0.02,
        "observed_joint": 0.003,
        "novelty_score": 1.9,
        "motivating_scene_ids": [],
        "arm": "visual",
        "plausibility_score": 0.78,
        "plausibility_justification": "Plausible",
        "frontier_difficulty_score": 0.55,
        "frontier_difficulty_signals": {},
        "final_rank_score": 2.1,
        "accepted": True,
        "rejection_reason": None,
    },
    {
        "composition_id": "proposal_rejected",
        "constituents": ["weather:fog"],
        "marginal_frequencies": {"weather:fog": 0.07},
        "pairwise_frequencies": {},
        "expected_joint": 0.07,
        "observed_joint": 0.07,
        "novelty_score": 0.0,
        "motivating_scene_ids": [],
        "arm": "reasoning",
        "plausibility_score": 0.3,
        "plausibility_justification": "Low plausibility",
        "frontier_difficulty_score": None,
        "frontier_difficulty_signals": {},
        "final_rank_score": 0.3,
        "accepted": False,
        "rejection_reason": "plausibility_score < threshold",
    },
]


@pytest.fixture()
def tmp_env(tmp_path: Path) -> Generator[dict[str, str], None, None]:
    """Set JUDGE_PROPOSALS_PATH and JUDGE_RATINGS_DIR to temp paths."""
    proposals_file = tmp_path / "proposals.json"
    proposals_file.write_text(json.dumps(FIXTURE_PROPOSALS), encoding="utf-8")
    ratings_dir = tmp_path / "ratings"

    env_overrides = {
        "JUDGE_PROPOSALS_PATH": str(proposals_file),
        "JUDGE_RATINGS_DIR": str(ratings_dir),
        "JUDGE_BUCKET_URI": "",  # no storage; video-url endpoint will 503
    }
    old = {k: os.environ.get(k) for k in env_overrides}
    os.environ.update(env_overrides)
    yield env_overrides
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture()
def client(tmp_env: dict[str, str]) -> TestClient:
    """TestClient with a freshly loaded server state."""
    import importlib
    import pipeline.modules.judge_ui.server as srv
    import pipeline.modules.judge_ui.config as cfg

    importlib.reload(cfg)
    importlib.reload(srv)

    srv._proposals.clear()
    srv._accepted_ids.clear()
    srv._storage = None
    srv._load_proposals()

    return TestClient(srv.app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# /judge/proposals
# ---------------------------------------------------------------------------

class TestListProposals:
    def test_returns_accepted_only(self, client: TestClient) -> None:
        resp = client.get("/judge/proposals")
        assert resp.status_code == 200
        ids = [p["proposal_id"] for p in resp.json()]
        assert "proposal_rejected" not in ids
        assert len(ids) == 2

    def test_sorted_by_rank_score_desc(self, client: TestClient) -> None:
        resp = client.get("/judge/proposals")
        scores = [p["scores"]["final_rank_score"] for p in resp.json()]
        assert scores == sorted(scores, reverse=True)

    def test_arm_not_present(self, client: TestClient) -> None:
        resp = client.get("/judge/proposals")
        for row in resp.json():
            assert "arm" not in row, "arm leaked into proposal list response"

    def test_motivating_scene_count(self, client: TestClient) -> None:
        resp = client.get("/judge/proposals")
        rows = {p["proposal_id"]: p for p in resp.json()}
        assert rows["proposal_a"]["motivating_scene_count"] == 1
        assert rows["proposal_b"]["motivating_scene_count"] == 0


# ---------------------------------------------------------------------------
# /judge/proposals/{id}
# ---------------------------------------------------------------------------

class TestGetProposal:
    def test_detail_accepted(self, client: TestClient) -> None:
        resp = client.get("/judge/proposals/proposal_a")
        assert resp.status_code == 200
        d = resp.json()
        assert d["proposal_id"] == "proposal_a"
        assert "arm" not in d, "arm leaked into proposal detail response"
        assert d["plausibility_justification"] == "Rain + turning is physically coherent"
        assert len(d["motivating_scenes"]) == 1

    def test_404_unknown(self, client: TestClient) -> None:
        resp = client.get("/judge/proposals/nonexistent")
        assert resp.status_code == 404

    def test_404_rejected_proposal(self, client: TestClient) -> None:
        resp = client.get("/judge/proposals/proposal_rejected")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /judge/ratings
# ---------------------------------------------------------------------------

class TestSubmitRating:
    def _submit(self, client: TestClient, **overrides) -> object:
        payload = {
            "rater_id": "alice",
            "proposal_id": "proposal_a",
            "coherence_score": 4,
            "usefulness_score": 5,
            "free_text_note": None,
            "seen_motivating_scenes": [],
            **overrides,
        }
        return client.post("/judge/ratings", json=payload)

    def test_valid_submission_returns_ok(self, client: TestClient) -> None:
        resp = self._submit(client)
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}

    def test_arm_injected_server_side(self, client: TestClient, tmp_env: dict) -> None:
        self._submit(client)
        path = Path(tmp_env["JUDGE_RATINGS_DIR"]) / "alice" / "proposal_a.json"
        assert path.exists()
        saved = json.loads(path.read_text())
        assert saved["arm"] == "reasoning"  # from fixture, not from rater input

    def test_both_scores_required(self, client: TestClient) -> None:
        payload = {"rater_id": "alice", "proposal_id": "proposal_a", "coherence_score": 3}
        resp = client.post("/judge/ratings", json=payload)
        assert resp.status_code == 422

    def test_score_out_of_range_rejected(self, client: TestClient) -> None:
        resp = self._submit(client, coherence_score=6)
        assert resp.status_code == 422
        resp2 = self._submit(client, usefulness_score=0)
        assert resp2.status_code == 422

    def test_duplicate_overwrites_previous(self, client: TestClient, tmp_env: dict) -> None:
        self._submit(client, coherence_score=4)
        self._submit(client, coherence_score=2)
        path = Path(tmp_env["JUDGE_RATINGS_DIR"]) / "alice" / "proposal_a.json"
        saved = json.loads(path.read_text())
        assert saved["coherence_score"] == 2

    def test_404_unknown_proposal(self, client: TestClient) -> None:
        resp = self._submit(client, proposal_id="nonexistent")
        assert resp.status_code == 404

    def test_seen_scenes_persisted(self, client: TestClient, tmp_env: dict) -> None:
        resp = self._submit(client, seen_motivating_scenes=[
            {"segment_id": "seg001", "window_idx": 2}
        ])
        assert resp.status_code == 200
        path = Path(tmp_env["JUDGE_RATINGS_DIR"]) / "alice" / "proposal_a.json"
        saved = json.loads(path.read_text())
        assert len(saved["seen_motivating_scenes"]) == 1
        assert saved["seen_motivating_scenes"][0]["segment_id"] == "seg001"


# ---------------------------------------------------------------------------
# GET /judge/session/{rater_id}
# ---------------------------------------------------------------------------

class TestSession:
    def test_empty_session(self, client: TestClient) -> None:
        resp = client.get("/judge/session/nobody")
        assert resp.status_code == 200
        data = resp.json()
        assert data["rated_proposal_ids"] == []
        assert data["total_accepted"] == 2
        assert data["mean_coherence"] is None

    def test_session_after_rating(self, client: TestClient) -> None:
        client.post("/judge/ratings", json={
            "rater_id": "bob",
            "proposal_id": "proposal_a",
            "coherence_score": 3,
            "usefulness_score": 4,
        })
        resp = client.get("/judge/session/bob")
        assert resp.status_code == 200
        data = resp.json()
        assert "proposal_a" in data["rated_proposal_ids"]
        assert data["mean_coherence"] == 3.0
        assert data["mean_usefulness"] == 4.0
        assert data["coherence_distribution"]["3"] == 1


# ---------------------------------------------------------------------------
# GET /judge/ratings/export
# ---------------------------------------------------------------------------

class TestExport:
    def test_empty_export(self, client: TestClient) -> None:
        resp = client.get("/judge/ratings/export")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_export_after_ratings(self, client: TestClient) -> None:
        client.post("/judge/ratings", json={
            "rater_id": "carol",
            "proposal_id": "proposal_a",
            "coherence_score": 5,
            "usefulness_score": 4,
        })
        client.post("/judge/ratings", json={
            "rater_id": "carol",
            "proposal_id": "proposal_b",
            "coherence_score": 3,
            "usefulness_score": 3,
        })
        resp = client.get("/judge/ratings/export")
        assert resp.status_code == 200
        ratings = resp.json()
        assert len(ratings) == 2

    def test_export_round_trips_rating_interface(self, client: TestClient) -> None:
        """Every exported rating must survive Rating.from_json(r.to_json())."""
        from pipeline.interfaces.rating import Rating

        client.post("/judge/ratings", json={
            "rater_id": "dave",
            "proposal_id": "proposal_a",
            "coherence_score": 4,
            "usefulness_score": 4,
            "free_text_note": "Interesting",
            "seen_motivating_scenes": [{"segment_id": "seg001", "window_idx": 2}],
        })
        resp = client.get("/judge/ratings/export")
        for r_json in resp.json():
            r = Rating.from_json(r_json)
            assert r.to_json() == r_json


# ---------------------------------------------------------------------------
# Video URL endpoint (storage not configured)
# ---------------------------------------------------------------------------

class TestVideoUrl:
    def test_falls_back_to_segment_proxy_when_no_bucket(self, client: TestClient) -> None:
        # With no WindowStorage configured, video-url falls back to the raw-segment
        # proxy route (/judge/segment-video) rather than returning 503.
        resp = client.get("/judge/video-url?segment_id=seg001&window_idx=2&camera=FRONT")
        assert resp.status_code == 200
        assert resp.json()["url"] == "/judge/segment-video/seg001?camera=FRONT"
