"""
Pydantic schemas for API boundary contracts.
"""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict


def to_camel(string: str) -> str:
    """
    Convert snake_case field names to camelCase aliases.
    """
    parts = string.split("_")
    return parts[0] + "".join(word.capitalize() for word in parts[1:])


class FailureTicketRequest(BaseModel):
    """
    Request schema for failure ticket ingestion.
    """

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    source_type: Literal["jira", "slack", "manual"]
    source_ref: str
    title: str
    raw_text: str
    event_timestamp: datetime
    agent_id: Optional[str] = None
    scenario_id: Optional[str] = None
    artifacts_ref: Optional[list[str]] = None


class FailureTicketResponse(BaseModel):
    """
    Response schema for failure ticket ingestion.
    """

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    ticket_id: str
    status: Literal["received", "processing", "failed"]
    message: str


class GenerateCapsuleRequest(BaseModel):
    """
    Request schema for failure capsule generation from a stored ticket.
    """

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    ticket_id: str


class FailureCapsuleResponse(BaseModel):
    """
    Response schema for generated or fetched failure capsules.
    """

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    capsule_id: str
    ticket_id: str
    triage_summary: str
    scenario_type: Optional[str] = None
    failure_mode_hints: list[str]
    likely_subsystem: Optional[str] = None
    severity_cue: Literal["critical", "high", "medium", "low", "unknown"]
    key_timestamp: Optional[datetime] = None
    tags: list[str]
    created_at: datetime
