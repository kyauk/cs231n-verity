"""
Tests for Feature 1.1 ingestion endpoint.
"""

from fastapi.testclient import TestClient

import api.ingest as ingest_module
import main as main_module


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


def test_ingest_happy_path(monkeypatch):
    """
    Happy-path ingestion returns ticket ID and received status.
    """

    async def _fake_save_ticket(pool, ticket) -> str:
        return ticket.ticket_id

    monkeypatch.setattr(main_module, "initialize_db_pool", _fake_initialize_db_pool)
    monkeypatch.setattr(main_module, "close_db_pool", _fake_close_db_pool)
    monkeypatch.setattr(ingest_module, "save_ticket", _fake_save_ticket)

    payload = {
        "sourceType": "jira",
        "sourceRef": "JIRA-123",
        "title": " Perception failure ",
        "rawText": " obstacle not detected ",
        "eventTimestamp": "2026-02-15T09:12:00Z",
        "agentId": "agent-01",
        "scenarioId": "scenario-01",
        "artifactsRef": [" replay://abc ", "log://xyz"],
    }

    with TestClient(main_module.app) as client:
        response = client.post("/ingest/failure_ticket", json=payload)

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "received"
    assert body["ticketId"]
    assert "stored" in body["message"].lower()


def test_ingest_missing_required_fields(monkeypatch):
    """
    Missing required fields should return validation errors.
    """
    monkeypatch.setattr(main_module, "initialize_db_pool", _fake_initialize_db_pool)
    monkeypatch.setattr(main_module, "close_db_pool", _fake_close_db_pool)

    payload = {
        "sourceType": "jira",
        "sourceRef": "JIRA-123",
    }

    with TestClient(main_module.app) as client:
        response = client.post("/ingest/failure_ticket", json=payload)

    assert response.status_code == 422


def test_ingest_invalid_source_type(monkeypatch):
    """
    Invalid sourceType value should be rejected.
    """
    monkeypatch.setattr(main_module, "initialize_db_pool", _fake_initialize_db_pool)
    monkeypatch.setattr(main_module, "close_db_pool", _fake_close_db_pool)

    payload = {
        "sourceType": "email",
        "sourceRef": "MAIL-1",
        "title": "Bad source",
        "rawText": "invalid source test",
        "eventTimestamp": "2026-02-15T09:12:00Z",
    }

    with TestClient(main_module.app) as client:
        response = client.post("/ingest/failure_ticket", json=payload)

    assert response.status_code == 422
