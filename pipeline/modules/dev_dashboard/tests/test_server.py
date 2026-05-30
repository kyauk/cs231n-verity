"""Endpoint tests for the dev_dashboard FastAPI server.

Uses FastAPI's TestClient — no real network/GCS. The VERITY_DEV_MODE gate
is exercised explicitly. All round filesystem persistence goes under a
tmp_path-rooted DEV_DASHBOARD_ROUNDS_DIR (set via monkeypatch).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pipeline.interfaces.proposal import ScoredProposal
from pipeline.interfaces.schema_record import SchemaRecord
from pipeline.interfaces.window import WindowKey


# ---------------------------------------------------------------------------
# Fixture: a fresh server with VERITY_DEV_MODE=1 and a tmp rounds dir
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("VERITY_DEV_MODE", "1")
    monkeypatch.setenv("DEV_DASHBOARD_ROUNDS_DIR", str(tmp_path / "rounds"))
    # Bucket left unset → /video-url returns 503
    monkeypatch.delenv("DEV_DASHBOARD_BUCKET_URI", raising=False)
    # Re-import config + server with the new env
    import importlib
    from pipeline.modules.dev_dashboard import config as cfg
    importlib.reload(cfg)
    from pipeline.modules.dev_dashboard import server as srv
    importlib.reload(srv)
    return TestClient(srv.app)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _record(i: int, weather: str = "clear", agents: str = "car") -> dict:
    return SchemaRecord(
        window_id=WindowKey(segment_id=f"seg_{i:04d}", window_idx=0),
        arm="reasoning", schema_version="1.0",
        prompt_template_id="v1_describe",
        fields={
            "agents": [agents],
            "environment": {"weather": weather, "time_of_day": "day",
                            "lighting_condition": "well_lit"},
            "road": {"geometry": "straight", "lane_count": 2},
            "traffic_control": "none", "ego_task": "cruising",
            "conditions": [],
        },
        failure_mode=None,
    ).to_json()


def _scored(i: int, *, rank: float, accepted: bool = True) -> dict:
    return ScoredProposal(
        composition_id=f"comp_{i:04d}",
        constituents=["agents:car", "weather:clear"],
        marginal_frequencies={}, pairwise_frequencies={},
        expected_joint=0.0, observed_joint=0.0, novelty_score=1.0,
        motivating_scene_ids=[WindowKey(segment_id=f"verity_seg_{i:04d}", window_idx=0)],
        arm="reasoning",
        plausibility_score=0.8, plausibility_justification="",
        frontier_difficulty_score=None, frontier_difficulty_signals={},
        final_rank_score=rank, accepted=accepted, rejection_reason=None,
    ).to_json()


def _create_payload(pool_size: int = 5) -> dict:
    """Smaller pool_size for tests so the synthetic data is tractable."""
    n_records = pool_size * 10
    records = [_record(i, weather="fog") for i in range(n_records // 10)]
    records += [_record(i + 10) for i in range(n_records - len(records))]
    proposals = [_scored(i, rank=float(100 - i)) for i in range(pool_size * 2)]
    return {
        "dataset_label": "test_dataset",
        "pool_size": pool_size,
        "seed": 42,
        "top_k_rare_atoms": 2,
        "scored": proposals,
        "schema_records": records,
    }


# ---------------------------------------------------------------------------
# VERITY_DEV_MODE gate
# ---------------------------------------------------------------------------

def test_server_refuses_to_start_without_verity_dev_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VERITY_DEV_MODE", raising=False)
    import importlib
    from pipeline.modules.dev_dashboard import config as cfg
    importlib.reload(cfg)
    from pipeline.modules.dev_dashboard import server as srv
    importlib.reload(srv)
    # Attempting to use the app triggers the lifespan → raises
    with pytest.raises(cfg.DevModeNotEnabledError):
        with TestClient(srv.app):
            pass


# ---------------------------------------------------------------------------
# POST /dev/rounds
# ---------------------------------------------------------------------------

def test_create_round_happy_path(client: TestClient) -> None:
    response = client.post("/dev/rounds", json=_create_payload())
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["round_id"].startswith("round_")
    assert body["total_windows"] == 15  # 3 pools * 5
    assert isinstance(body["naive_rare_atoms"], list)


def test_create_round_writes_manifest_to_disk(
    client: TestClient, tmp_path: Path,
) -> None:
    response = client.post("/dev/rounds", json=_create_payload())
    round_id = response.json()["round_id"]
    manifest_path = tmp_path / "rounds" / round_id / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["round_id"] == round_id
    assert set(manifest["pools"].keys()) == {"verity", "random", "naive_rare"}


def test_create_round_too_few_records_returns_400(client: TestClient) -> None:
    payload = _create_payload()
    payload["pool_size"] = 100  # way more than synthetic data supports
    response = client.post("/dev/rounds", json=payload)
    assert response.status_code == 400
    assert "pool" in response.text.lower()


def test_create_round_malformed_scored_returns_422(client: TestClient) -> None:
    payload = _create_payload()
    payload["scored"] = [{"this": "is_not_a_valid_proposal"}]
    response = client.post("/dev/rounds", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /dev/rounds
# ---------------------------------------------------------------------------

def test_list_rounds_returns_newest_first(client: TestClient) -> None:
    r1 = client.post("/dev/rounds", json=_create_payload()).json()["round_id"]
    r2 = client.post("/dev/rounds", json=_create_payload()).json()["round_id"]
    response = client.get("/dev/rounds")
    assert response.status_code == 200
    ids = [r["round_id"] for r in response.json()]
    assert set(ids) == {r1, r2}


# ---------------------------------------------------------------------------
# GET /dev/rounds/{round_id}
# ---------------------------------------------------------------------------

def test_get_round_status(client: TestClient) -> None:
    rid = client.post("/dev/rounds", json=_create_payload()).json()["round_id"]
    response = client.get(f"/dev/rounds/{rid}")
    assert response.status_code == 200
    body = response.json()
    assert body["round_id"] == rid
    assert body["total_windows"] == 15
    assert body["rated_count"] == 0
    assert body["complete"] is False


def test_get_round_status_unknown_returns_404(client: TestClient) -> None:
    response = client.get("/dev/rounds/nonexistent")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /dev/rounds/{round_id}/next
# ---------------------------------------------------------------------------

def test_next_window_returns_first_unrated(client: TestClient) -> None:
    rid = client.post("/dev/rounds", json=_create_payload()).json()["round_id"]
    response = client.get(f"/dev/rounds/{rid}/next")
    assert response.status_code == 200
    body = response.json()
    assert body["complete"] is False
    assert body["window"] is not None
    assert body["progress_idx"] == 1
    assert body["total_windows"] == 15
    # CRITICAL: source_pool must NOT leak to the rater
    assert "source_pool" not in body


# ---------------------------------------------------------------------------
# POST /dev/rounds/{round_id}/ratings
# ---------------------------------------------------------------------------

def test_submit_rating_persists(client: TestClient) -> None:
    rid = client.post("/dev/rounds", json=_create_payload()).json()["round_id"]
    nxt = client.get(f"/dev/rounds/{rid}/next").json()
    payload = {
        "rater_id": "alice",
        "window": nxt["window"],
        "safety_relevance": 4,
        "perceived_rarity": 3,
        "free_text_note": "felt risky",
    }
    response = client.post(f"/dev/rounds/{rid}/ratings", json=payload)
    assert response.status_code == 200
    # Status now shows 1 rated
    status = client.get(f"/dev/rounds/{rid}").json()
    assert status["rated_count"] == 1


def test_submit_rating_window_not_in_round_returns_400(client: TestClient) -> None:
    rid = client.post("/dev/rounds", json=_create_payload()).json()["round_id"]
    payload = {
        "rater_id": "alice",
        "window": {"segment_id": "not_in_round", "window_idx": 0},
        "safety_relevance": 4,
        "perceived_rarity": 3,
    }
    response = client.post(f"/dev/rounds/{rid}/ratings", json=payload)
    assert response.status_code == 400


def test_submit_rating_out_of_range_score_returns_422(client: TestClient) -> None:
    rid = client.post("/dev/rounds", json=_create_payload()).json()["round_id"]
    nxt = client.get(f"/dev/rounds/{rid}/next").json()
    payload = {
        "rater_id": "alice", "window": nxt["window"],
        "safety_relevance": 9,  # out of 1-5
        "perceived_rarity": 3,
    }
    response = client.post(f"/dev/rounds/{rid}/ratings", json=payload)
    assert response.status_code == 422


def test_rating_reveals_source_pool_only_to_export(client: TestClient) -> None:
    """The server records the source pool in the persisted rating's `arm`
    field, but the /next endpoint never returns it. The /export endpoint
    reveals it."""
    rid = client.post("/dev/rounds", json=_create_payload()).json()["round_id"]
    nxt = client.get(f"/dev/rounds/{rid}/next").json()
    client.post(f"/dev/rounds/{rid}/ratings", json={
        "rater_id": "alice", "window": nxt["window"],
        "safety_relevance": 4, "perceived_rarity": 3,
    })
    exported = client.get(f"/dev/rounds/{rid}/export").json()
    assert len(exported["ratings"]) == 1
    assert exported["ratings"][0]["source_pool"] in {"verity", "random", "naive_rare"}


# ---------------------------------------------------------------------------
# GET /dev/rounds/{round_id}/export
# ---------------------------------------------------------------------------

def test_export_returns_revealed_source_labels(client: TestClient) -> None:
    rid = client.post("/dev/rounds", json=_create_payload()).json()["round_id"]
    # Submit 3 ratings
    for _ in range(3):
        nxt = client.get(f"/dev/rounds/{rid}/next").json()
        client.post(f"/dev/rounds/{rid}/ratings", json={
            "rater_id": "alice", "window": nxt["window"],
            "safety_relevance": 4, "perceived_rarity": 3,
        })
    response = client.get(f"/dev/rounds/{rid}/export")
    assert response.status_code == 200
    body = response.json()
    assert body["round_id"] == rid
    assert len(body["ratings"]) == 3
    assert all(
        r["source_pool"] in {"verity", "random", "naive_rare"}
        for r in body["ratings"]
    )
    assert body["complete"] is False
    assert body["seed"] == 42


def test_export_complete_flag_true_when_all_rated(client: TestClient) -> None:
    rid = client.post(
        "/dev/rounds", json=_create_payload(pool_size=2),
    ).json()["round_id"]  # 6 windows total
    for _ in range(6):
        nxt = client.get(f"/dev/rounds/{rid}/next").json()
        if nxt["complete"]:
            break
        client.post(f"/dev/rounds/{rid}/ratings", json={
            "rater_id": "alice", "window": nxt["window"],
            "safety_relevance": 4, "perceived_rarity": 3,
        })
    body = client.get(f"/dev/rounds/{rid}/export").json()
    assert body["complete"] is True


# ---------------------------------------------------------------------------
# GET /dev/rounds/{round_id}/video-url
# ---------------------------------------------------------------------------

def test_video_url_503_when_bucket_not_configured(client: TestClient) -> None:
    rid = client.post("/dev/rounds", json=_create_payload()).json()["round_id"]
    nxt = client.get(f"/dev/rounds/{rid}/next").json()
    response = client.get(
        f"/dev/rounds/{rid}/video-url",
        params={"segment_id": nxt["window"]["segment_id"],
                "window_idx": nxt["window"]["window_idx"]},
    )
    assert response.status_code == 503


# ---------------------------------------------------------------------------
# GET /dev/accuracy/template
# ---------------------------------------------------------------------------

def test_accuracy_template_returns_valid_gold_set(client: TestClient) -> None:
    response = client.get("/dev/accuracy/template")
    assert response.status_code == 200
    body = response.json()
    assert "schema_version" in body
    assert "labels" in body
    assert len(body["labels"]) >= 1


# ---------------------------------------------------------------------------
# POST /dev/accuracy/diff
# ---------------------------------------------------------------------------

def test_accuracy_diff_round_trip(client: TestClient) -> None:
    # Build gold + matching records
    record = _record(0)
    gold = {
        "schema_version": "1.0",
        "labels": [{
            "window_id": record["window_id"],
            "fields": record["fields"],
            "label_source": "human:test",
        }],
    }
    response = client.post(
        "/dev/accuracy/diff",
        json={"gold": gold, "schema_records": [record]},
    )
    assert response.status_code == 200
    body = response.json()
    assert "windows" in body
    assert "field_aggregates" in body
    assert len(body["windows"]) == 1
    # Perfect match → every field counts 1/1
    for path, agg in body["field_aggregates"].items():
        assert agg == [1, 1], f"{path}: {agg}"


def test_accuracy_diff_malformed_gold_returns_422(client: TestClient) -> None:
    response = client.post(
        "/dev/accuracy/diff",
        json={"gold": {"no_labels_key": True}, "schema_records": []},
    )
    assert response.status_code == 422
