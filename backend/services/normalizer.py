"""
Normalization helpers for ingestion inputs.
"""

from datetime import datetime, timezone
from uuid import uuid4

from models.api_schemas import FailureTicketRequest
from models.db_records import RawTicketRecord

VALID_SOURCE_TYPES = {"jira", "slack", "manual"}
MAX_RAW_TEXT_LENGTH = 100_000


def normalize_failure_ticket(payload: FailureTicketRequest) -> RawTicketRecord:
    '''
    Purpose: Normalize validated API payload fields into an immutable DB record.
    Parameters:
    payload (FailureTicketRequest): Request body parsed by FastAPI/Pydantic.
    Returns:
    RawTicketRecord: Normalized ticket ready for repository persistence.
    Called by: backend/api/ingest.py -> ingest_failure_ticket()
    Calls: uuid.uuid4()
    '''
    normalized_source_type = payload.source_type.strip().lower()
    if normalized_source_type not in VALID_SOURCE_TYPES:
        raise ValueError("sourceType must be one of: jira, slack, manual")

    normalized_source_ref = payload.source_ref.strip()
    normalized_title = payload.title.strip()
    normalized_raw_text = payload.raw_text.strip()

    if not normalized_source_ref:
        raise ValueError("sourceRef must not be empty")
    if not normalized_title:
        raise ValueError("title must not be empty")
    if not normalized_raw_text:
        raise ValueError("rawText must not be empty")
    if len(normalized_raw_text) > MAX_RAW_TEXT_LENGTH:
        raise ValueError(f"rawText exceeds max length of {MAX_RAW_TEXT_LENGTH}")

    normalized_event_timestamp = payload.event_timestamp
    if normalized_event_timestamp.tzinfo is None:
        normalized_event_timestamp = normalized_event_timestamp.replace(tzinfo=timezone.utc)
    else:
        normalized_event_timestamp = normalized_event_timestamp.astimezone(timezone.utc)

    normalized_artifacts = (
        [artifact.strip() for artifact in payload.artifacts_ref if artifact and artifact.strip()]
        if payload.artifacts_ref
        else None
    )

    return RawTicketRecord(
        ticket_id=str(uuid4()),
        source_type=normalized_source_type,
        source_ref=normalized_source_ref,
        title=normalized_title,
        raw_text=normalized_raw_text,
        event_timestamp=normalized_event_timestamp,
        agent_id=payload.agent_id.strip() if payload.agent_id else None,
        scenario_id=payload.scenario_id.strip() if payload.scenario_id else None,
        artifacts_ref=normalized_artifacts,
        created_at=datetime.now(timezone.utc),
    )
