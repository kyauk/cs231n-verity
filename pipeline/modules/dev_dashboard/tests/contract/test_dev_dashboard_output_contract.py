"""Contract tests for Module 7: Dev Dashboard outputs.

Two output surfaces this module owns:
  1. DevRoundManifest persisted JSON shape (consumed by the server itself
     on subsequent calls — must round-trip cleanly).
  2. /dev/rounds/{round_id}/export response shape (consumed by an offline
     Mann-Whitney analysis the operator runs on the CSV/JSON). Source-pool
     labels must be REVEALED here (opposite of /next which hides them).

If this test fails, EITHER the README/plan is stale OR the implementation
diverged. Reconcile per Step 5 of the hygiene protocol.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pipeline.interfaces.dev_round import DevRoundManifest
from pipeline.interfaces.proposal import ScoredProposal
from pipeline.interfaces.schema_record import SchemaRecord
from pipeline.interfaces.window import WindowKey


# ---------------------------------------------------------------------------
# Fixture: fresh server under VERITY_DEV_MODE=1
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("VERITY_DEV_MODE", "1")
    monkeypatch.setenv("DEV_DASHBOARD_ROUNDS_DIR", str(tmp_path / "rounds"))
    monkeypatch.delenv("DEV_DASHBOARD_BUCKET_URI", raising=False)
    from pipeline.modules.dev_dashboard import config as cfg
    importlib.reload(cfg)
    from pipeline.modules.dev_dashboard import server as srv
    importlib.reload(srv)
    return TestClient(srv.app)


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


def _scored(i: int, rank: float) -> dict:
    return ScoredProposal(
        composition_id=f"comp_{i:04d}",
        constituents=["agents:car", "weather:clear"],
        marginal_frequencies={}, pairwise_frequencies={},
        expected_joint=0.0, observed_joint=0.0, novelty_score=1.0,
        motivating_scene_ids=[WindowKey(segment_id=f"v{i:04d}", window_idx=0)],
        arm="reasoning",
        plausibility_score=0.8, plausibility_justification="",
        frontier_difficulty_score=None, frontier_difficulty_signals={},
        final_rank_score=rank, accepted=True, rejection_reason=None,
    ).to_json()


def _create_round_payload(pool_size: int = 3) -> dict:
    n = pool_size * 10
    records = [_record(i, weather="fog") for i in range(n // 10)]
    records += [_record(i + 10) for i in range(n - len(records))]
    return {
        "dataset_label": "contract_test",
        "pool_size": pool_size,
        "seed": 7,
        "top_k_rare_atoms": 2,
        "scored": [_scored(i, rank=float(100 - i)) for i in range(pool_size * 2)],
        "schema_records": records,
    }


# ---------------------------------------------------------------------------
# Contract 1: DevRoundManifest filesystem JSON shape
# ---------------------------------------------------------------------------

def test_persisted_manifest_round_trips_through_interface_type(
    client: TestClient, tmp_path: Path,
) -> None:
    rid = client.post("/dev/rounds",
                      json=_create_round_payload()).json()["round_id"]
    manifest_path = tmp_path / "rounds" / rid / "manifest.json"
    raw = json.loads(manifest_path.read_text())
    # Must parse back through the interface type without information loss
    restored = DevRoundManifest.from_json(raw)
    assert restored.round_id == rid
    assert restored.pool_size == 3
    assert restored.seed == 7
    assert set(restored.pools.keys()) == {"verity", "random", "naive_rare"}
    assert len(restored.shuffled_order) == 9   # 3 * pool_size


def test_persisted_manifest_has_every_documented_field(
    client: TestClient, tmp_path: Path,
) -> None:
    """Every field declared on DevRoundManifest must appear in the JSON."""
    rid = client.post("/dev/rounds",
                      json=_create_round_payload()).json()["round_id"]
    raw = json.loads((tmp_path / "rounds" / rid / "manifest.json").read_text())
    for field in (
        "round_id", "created_at", "dataset_label", "pool_size",
        "seed", "pools", "shuffled_order", "naive_rare_atoms",
    ):
        assert field in raw, f"manifest missing documented field: {field}"


# ---------------------------------------------------------------------------
# Contract 2: /dev/rounds/{round_id}/export shape
# ---------------------------------------------------------------------------

def test_export_response_shape_is_stable(client: TestClient) -> None:
    """Frozen export shape — offline analysis scripts pin against this."""
    rid = client.post("/dev/rounds",
                      json=_create_round_payload()).json()["round_id"]
    # Submit one rating
    nxt = client.get(f"/dev/rounds/{rid}/next").json()
    client.post(f"/dev/rounds/{rid}/ratings", json={
        "rater_id": "alice", "window": nxt["window"],
        "safety_relevance": 4, "perceived_rarity": 3,
    })
    body = client.get(f"/dev/rounds/{rid}/export").json()
    # Top-level keys
    for k in ("round_id", "dataset_label", "pool_size", "seed",
              "naive_rare_atoms", "complete", "ratings"):
        assert k in body, f"export missing top-level key: {k}"
    # Each row shape
    row = body["ratings"][0]
    for k in ("rater_id", "window", "source_pool",
              "safety_relevance", "perceived_rarity",
              "timestamp", "free_text_note"):
        assert k in row, f"export row missing field: {k}"


def test_export_source_pool_values_are_documented_set(client: TestClient) -> None:
    """source_pool must be one of the three documented labels."""
    rid = client.post("/dev/rounds",
                      json=_create_round_payload()).json()["round_id"]
    # Submit all 9 ratings
    for _ in range(9):
        nxt = client.get(f"/dev/rounds/{rid}/next").json()
        if nxt["complete"]:
            break
        client.post(f"/dev/rounds/{rid}/ratings", json={
            "rater_id": "alice", "window": nxt["window"],
            "safety_relevance": 3, "perceived_rarity": 3,
        })
    body = client.get(f"/dev/rounds/{rid}/export").json()
    for row in body["ratings"]:
        assert row["source_pool"] in {"verity", "random", "naive_rare"}, (
            f"unexpected source_pool: {row['source_pool']!r}"
        )


# ---------------------------------------------------------------------------
# Contract 3: /next response must NOT leak source_pool
# ---------------------------------------------------------------------------

def test_next_endpoint_never_reveals_source_pool(client: TestClient) -> None:
    """Blinding contract: the rater never sees which pool a window came from."""
    rid = client.post("/dev/rounds",
                      json=_create_round_payload()).json()["round_id"]
    body = client.get(f"/dev/rounds/{rid}/next").json()
    # No top-level source_pool
    assert "source_pool" not in body
    # And not inside the window object either
    if body["window"] is not None:
        assert "source_pool" not in body["window"]
        assert "arm" not in body["window"]


# ---------------------------------------------------------------------------
# Contract 4: accuracy diff response shape
# ---------------------------------------------------------------------------

def test_accuracy_diff_response_shape(client: TestClient) -> None:
    record = _record(0)
    gold = {
        "schema_version": "1.0",
        "labels": [{
            "window_id": record["window_id"],
            "fields": record["fields"],
            "label_source": "human:test",
        }],
    }
    body = client.post(
        "/dev/accuracy/diff",
        json={"gold": gold, "schema_records": [record]},
    ).json()
    for k in ("schema_version", "windows", "field_aggregates", "missing_entries"):
        assert k in body
    # field_aggregates is a dict of (matches, total)
    for path, agg in body["field_aggregates"].items():
        assert isinstance(agg, list) and len(agg) == 2
