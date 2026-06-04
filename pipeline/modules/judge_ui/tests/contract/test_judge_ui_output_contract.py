"""Contract test for Module 5: Judge UI.

Validates that every Rating produced by the server:
  1. Conforms to the Rating interface type from pipeline/interfaces/rating.py
  2. Contains every field declared in the README Output Contract section
  3. Has every documented side effect (file written to ratings dir)
  4. Fails clearly when given malformed input

README Output Contract fields for Rating:
  rater_id: str
  proposal_id: str
  arm: str              — blinded to rater, recorded server-side
  coherence_score: int  — 1-5
  usefulness_score: int — 1-5
  timestamp: str        — ISO-8601
  free_text_note: str | None
  seen_motivating_scenes: list[WindowKey]
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pipeline.interfaces.rating import Rating
from pipeline.interfaces.window import WindowKey

FIXTURE_PROPOSALS = [
    {
        "composition_id": "contract_001",
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
        "composition_id": "contract_002",
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
def env_and_client(tmp_path: Path):
    proposals_file = tmp_path / "proposals.json"
    proposals_file.write_text(json.dumps(FIXTURE_PROPOSALS))
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
    importlib.reload(cfg)
    importlib.reload(srv)
    srv._proposals.clear(); srv._accepted_ids.clear(); srv._storage = None
    srv._load_proposals()

    client = TestClient(srv.app, raise_server_exceptions=True)
    yield client, ratings_dir, srv

    for k, v in old.items():
        if v is None: os.environ.pop(k, None)
        else: os.environ[k] = v


class TestRatingOutputContract:
    """Every Rating emitted by POST /judge/ratings must satisfy the interface contract."""

    def _submit(self, client, **overrides):
        payload = {
            "rater_id": "contract_tester",
            "proposal_id": "contract_001",
            "coherence_score": 4,
            "usefulness_score": 3,
            "free_text_note": "Contract test note",
            "seen_motivating_scenes": [{"segment_id": "seg001", "window_idx": 2}],
            **overrides,
        }
        resp = client.post("/judge/ratings", json=payload)
        assert resp.status_code == 200
        return resp

    def test_rating_file_written_as_side_effect(self, env_and_client):
        """Documented side effect: rating persisted to ratings/{rater_id}/{proposal_id}.json"""
        client, ratings_dir, _ = env_and_client
        self._submit(client)
        path = ratings_dir / "contract_tester" / "contract_001.json"
        assert path.exists(), "Side effect missing: rating file was not written"

    def test_all_readme_fields_present(self, env_and_client):
        """Every field in the README Output Contract section must be present."""
        client, ratings_dir, _ = env_and_client
        self._submit(client)
        raw = json.loads((ratings_dir / "contract_tester" / "contract_001.json").read_text())

        required_fields = [
            "rater_id", "proposal_id", "arm",
            "coherence_score", "usefulness_score",
            "timestamp", "free_text_note", "seen_motivating_scenes",
        ]
        for field in required_fields:
            assert field in raw, f"README contract field missing from output: {field!r}"

    def test_rating_round_trips_through_interface(self, env_and_client):
        """Output must survive Rating.from_json(rating.to_json()) round-trip."""
        client, ratings_dir, _ = env_and_client
        self._submit(client)
        raw = json.loads((ratings_dir / "contract_tester" / "contract_001.json").read_text())
        rating = Rating.from_json(raw)
        assert rating.to_json() == raw

    def test_rater_id_type(self, env_and_client):
        client, ratings_dir, _ = env_and_client
        self._submit(client)
        raw = json.loads((ratings_dir / "contract_tester" / "contract_001.json").read_text())
        assert isinstance(raw["rater_id"], str)

    def test_proposal_id_type(self, env_and_client):
        client, ratings_dir, _ = env_and_client
        self._submit(client)
        raw = json.loads((ratings_dir / "contract_tester" / "contract_001.json").read_text())
        assert isinstance(raw["proposal_id"], str)

    def test_arm_is_server_injected_not_rater_provided(self, env_and_client):
        """arm must come from the proposal store, not from rater input."""
        client, ratings_dir, _ = env_and_client
        self._submit(client)
        raw = json.loads((ratings_dir / "contract_tester" / "contract_001.json").read_text())
        assert raw["arm"] == "reasoning"  # from fixture, not from rater

    def test_arm_visual_proposal_correctly_recorded(self, env_and_client):
        """arm=visual is also correctly recorded for visual-arm proposals."""
        client, ratings_dir, _ = env_and_client
        self._submit(client, proposal_id="contract_002",
                     seen_motivating_scenes=[])
        raw = json.loads((ratings_dir / "contract_tester" / "contract_002.json").read_text())
        assert raw["arm"] == "visual"

    def test_arm_never_in_list_response(self, env_and_client):
        """arm must not appear in GET /judge/proposals response."""
        client, _, _ = env_and_client
        rows = client.get("/judge/proposals").json()
        for row in rows:
            assert "arm" not in row, f"arm leaked into list: {row}"

    def test_arm_never_in_detail_response(self, env_and_client):
        """arm must not appear in GET /judge/proposals/{id} response."""
        client, _, _ = env_and_client
        detail = client.get("/judge/proposals/contract_001").json()
        assert "arm" not in detail, f"arm leaked into detail: {detail}"

    def test_coherence_score_is_int_in_range(self, env_and_client):
        client, ratings_dir, _ = env_and_client
        self._submit(client)
        raw = json.loads((ratings_dir / "contract_tester" / "contract_001.json").read_text())
        assert isinstance(raw["coherence_score"], int)
        assert 1 <= raw["coherence_score"] <= 5

    def test_usefulness_score_is_int_in_range(self, env_and_client):
        client, ratings_dir, _ = env_and_client
        self._submit(client)
        raw = json.loads((ratings_dir / "contract_tester" / "contract_001.json").read_text())
        assert isinstance(raw["usefulness_score"], int)
        assert 1 <= raw["usefulness_score"] <= 5

    def test_timestamp_is_iso8601(self, env_and_client):
        from datetime import datetime
        client, ratings_dir, _ = env_and_client
        self._submit(client)
        raw = json.loads((ratings_dir / "contract_tester" / "contract_001.json").read_text())
        ts = raw["timestamp"]
        assert isinstance(ts, str)
        # Must parse as datetime
        datetime.fromisoformat(ts)

    def test_free_text_note_can_be_none(self, env_and_client):
        client, ratings_dir, _ = env_and_client
        self._submit(client, free_text_note=None)
        raw = json.loads((ratings_dir / "contract_tester" / "contract_001.json").read_text())
        assert raw["free_text_note"] is None

    def test_seen_motivating_scenes_are_window_keys(self, env_and_client):
        """seen_motivating_scenes must be list[WindowKey] shape."""
        client, ratings_dir, _ = env_and_client
        self._submit(client)
        raw = json.loads((ratings_dir / "contract_tester" / "contract_001.json").read_text())
        scenes = raw["seen_motivating_scenes"]
        assert isinstance(scenes, list)
        for s in scenes:
            assert "segment_id" in s and isinstance(s["segment_id"], str)
            assert "window_idx" in s and isinstance(s["window_idx"], int)
            # Must parse as WindowKey
            WindowKey.from_json(s)

    def test_export_matches_interface_contract(self, env_and_client):
        """Every rating returned by /judge/ratings/export must satisfy Rating.from_json."""
        client, _, _ = env_and_client
        self._submit(client)
        self._submit(client, proposal_id="contract_002", seen_motivating_scenes=[])
        exported = client.get("/judge/ratings/export").json()
        assert len(exported) == 2
        for r_json in exported:
            rating = Rating.from_json(r_json)
            # Re-serialized form must match
            assert rating.to_json() == r_json

    def test_malformed_score_fails_clearly(self, env_and_client):
        """Out-of-range score must return 422, not 200 or 500."""
        client, _, _ = env_and_client
        resp = client.post("/judge/ratings", json={
            "rater_id": "bad_rater", "proposal_id": "contract_001",
            "coherence_score": 99, "usefulness_score": 3,
        })
        assert resp.status_code == 422, f"Expected 422, got {resp.status_code}"

    def test_missing_usefulness_fails_clearly(self, env_and_client):
        """Omitting usefulness_score must return 422."""
        client, _, _ = env_and_client
        resp = client.post("/judge/ratings", json={
            "rater_id": "bad_rater", "proposal_id": "contract_001",
            "coherence_score": 3,
        })
        assert resp.status_code == 422
