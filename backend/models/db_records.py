"""
Pydantic records for database boundary contracts.
"""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel


class RawTicketRecord(BaseModel):
    """
    Immutable raw ticket record persisted at ingestion time.
    """

    ticket_id: str
    source_type: str
    source_ref: str
    title: str
    raw_text: str
    event_timestamp: datetime
    agent_id: Optional[str] = None
    scenario_id: Optional[str] = None
    artifacts_ref: Optional[list[str]] = None
    created_at: datetime


class FailureCapsuleRecord(BaseModel):
    """
    Stored normalized failure capsule for downstream retrieval.
    """

    capsule_id: str
    ticket_id: str
    triage_summary: str
    scenario_type: Optional[str] = None
    failure_mode_hints: list[str]
    likely_subsystem: Optional[str] = None
    severity_cue: Literal["critical", "high", "medium", "low", "unknown"]
    key_timestamp: Optional[datetime] = None
    tags: list[str]
    embedding: list[float]
    created_at: datetime
