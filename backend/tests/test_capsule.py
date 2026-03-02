"""
Tests for Feature 2 failure capsule endpoints.
"""

from datetime import datetime, timezone

from fastapi.testclient import TestClient

import api.capsule as capsule_module
import main as main_module
from models.db_records import FailureCapsuleRecord


class FakePool:
    """
    Simple fake pool object used by tests.
    """


async def _fake_initialize_db_pool() -> FakePool:
    '''
    Purpose: Provide a test-safe replacement for startup DB pool creation.
    Parameters:
    None: This function does not require runtime arguments.
    Returns:
    FakePool: In-memory placeholder object for app.state.db_pool.
    Called by: backend/main.py -> lifespan() during tests
    Calls: None
    '''
    return FakePool()


async def _fake_close_db_pool(pool: FakePool) -> None:
    '''
    Purpose: Provide a no-op replacement for DB pool shutdown in tests.
    Parameters:
    pool (FakePool): Placeholder pool initialized in test startup.
    Returns:
    None: This function performs no operation.
    Called by: backend/main.py -> lifespan() during tests
    Calls: None
    '''
    return None


def _sample_capsule(ticket_id: str = "ticket-123") -> FailureCapsuleRecord:
    '''
    Purpose: Build a deterministic capsule record fixture used by API route tests.
    Parameters:
    ticket_id (str): Ticket identifier to associate with the generated capsule fixture.
    Returns:
    FailureCapsuleRecord: Typed sample capsule record with stable field values.
    Called by: backend/tests/test_capsule.py -> test cases in this module
    Calls: None
    '''
    return FailureCapsuleRecord(
        capsule_id="capsule-123",
        ticket_id=ticket_id,
        triage_summary="Vehicle drifted after lane-marking loss.",
        scenario_type="urban_driving",
        failure_mode_hints=["lane_detection_dropout", "control_overshoot"],
        likely_subsystem="perception",
        severity_cue="high",
        key_timestamp=datetime(2026, 2, 15, 9, 12, tzinfo=timezone.utc),
        tags=["lane", "perception", "safety"],
        embedding=[0.1, 0.2, 0.3],
        created_at=datetime(2026, 2, 15, 9, 15, tzinfo=timezone.utc),
    )


def test_generate_capsule_happy_path(monkeypatch):
    """
    Happy-path generation returns a normalized capsule payload.
    """

    async def _fake_generate_capsule_node(ticket_id: str, db_pool) -> FailureCapsuleRecord:
        return _sample_capsule(ticket_id=ticket_id)

    monkeypatch.setattr(main_module, "initialize_db_pool", _fake_initialize_db_pool)
    monkeypatch.setattr(main_module, "close_db_pool", _fake_close_db_pool)
    monkeypatch.setattr(capsule_module, "generate_capsule_node", _fake_generate_capsule_node)

    with TestClient(main_module.app) as client:
        response = client.post("/generate/failure_capsule", json={"ticketId": "ticket-xyz"})

    assert response.status_code == 201
    body = response.json()
    assert body["capsuleId"] == "capsule-123"
    assert body["ticketId"] == "ticket-xyz"
    assert body["severityCue"] == "high"
    assert body["failureModeHints"]


def test_generate_capsule_missing_ticket_returns_404(monkeypatch):
    """
    Missing ticket surfaces as 404 from generation endpoint.
    """

    async def _fake_generate_capsule_node(ticket_id: str, db_pool) -> FailureCapsuleRecord:
        raise ValueError(f"Ticket not found for ticket_id={ticket_id}")

    monkeypatch.setattr(main_module, "initialize_db_pool", _fake_initialize_db_pool)
    monkeypatch.setattr(main_module, "close_db_pool", _fake_close_db_pool)
    monkeypatch.setattr(capsule_module, "generate_capsule_node", _fake_generate_capsule_node)

    with TestClient(main_module.app) as client:
        response = client.post("/generate/failure_capsule", json={"ticketId": "missing-ticket"})

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_fetch_capsule_happy_path(monkeypatch):
    """
    Fetch endpoint returns existing capsule by capsule ID.
    """

    async def _fake_get_capsule(db_pool, capsule_id: str) -> FailureCapsuleRecord | None:
        return _sample_capsule() if capsule_id == "capsule-123" else None

    monkeypatch.setattr(main_module, "initialize_db_pool", _fake_initialize_db_pool)
    monkeypatch.setattr(main_module, "close_db_pool", _fake_close_db_pool)
    monkeypatch.setattr(capsule_module, "get_capsule", _fake_get_capsule)

    with TestClient(main_module.app) as client:
        response = client.get("/fetch/failure_capsule/capsule-123")

    assert response.status_code == 200
    body = response.json()
    assert body["capsuleId"] == "capsule-123"
    assert body["likelySubsystem"] == "perception"


def test_fetch_capsule_missing_returns_404(monkeypatch):
    """
    Fetch endpoint returns 404 when capsule is not stored.
    """

    async def _fake_get_capsule(db_pool, capsule_id: str) -> FailureCapsuleRecord | None:
        return None

    monkeypatch.setattr(main_module, "initialize_db_pool", _fake_initialize_db_pool)
    monkeypatch.setattr(main_module, "close_db_pool", _fake_close_db_pool)
    monkeypatch.setattr(capsule_module, "get_capsule", _fake_get_capsule)

    with TestClient(main_module.app) as client:
        response = client.get("/fetch/failure_capsule/missing-capsule")

    assert response.status_code == 404
