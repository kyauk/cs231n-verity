"""
Agent node for Feature 2 failure capsule generation.
"""

from datetime import datetime, timezone
from uuid import uuid4

import asyncpg

from db.capsule_repo import save_capsule
from db.ticket_repo import get_ticket
from models.db_records import FailureCapsuleRecord
from services.llm_client import generate_embedding, generate_triage_summary


async def generate_capsule_node(ticket_id: str, db_pool: asyncpg.Pool) -> FailureCapsuleRecord:
    '''
    Purpose: Generate and persist a normalized failure capsule from a raw ticket record.
    Parameters:
    ticket_id (str): UUID string identifying the raw ticket to transform into a capsule.
    db_pool (asyncpg.Pool): Shared asyncpg pool used for ticket/capsule repository calls.
    Returns:
    FailureCapsuleRecord: Persisted typed capsule record including embedding and metadata.
    Called by: backend/api/capsule.py -> generate_failure_capsule()
    Calls: db.ticket_repo.get_ticket(), services.llm_client.generate_triage_summary(), services.llm_client.generate_embedding(), db.capsule_repo.save_capsule()
    '''
    ticket = await get_ticket(db_pool, ticket_id)
    if ticket is None:
        raise ValueError(f"Ticket not found for ticket_id={ticket_id}")

    triage_summary = await generate_triage_summary(
        raw_text=ticket.raw_text,
        metadata={
            "title": ticket.title,
            "source_type": ticket.source_type,
            "source_ref": ticket.source_ref,
            "event_timestamp": ticket.event_timestamp,
            "agent_id": ticket.agent_id,
            "scenario_id": ticket.scenario_id,
            "artifacts_ref": ticket.artifacts_ref,
        },
    )
    embedding_input = "\n".join(
        [
            f"title: {ticket.title}",
            f"summary: {triage_summary.summary}",
            f"failure_mode_hints: {', '.join(triage_summary.failure_mode_hints)}",
            f"likely_subsystem: {triage_summary.likely_subsystem or ''}",
            f"tags: {', '.join(triage_summary.tags)}",
        ]
    )
    embedding = await generate_embedding(embedding_input)

    capsule = FailureCapsuleRecord(
        capsule_id=str(uuid4()),
        ticket_id=ticket.ticket_id,
        triage_summary=triage_summary.summary,
        scenario_type=triage_summary.scenario_type,
        failure_mode_hints=triage_summary.failure_mode_hints,
        likely_subsystem=triage_summary.likely_subsystem,
        severity_cue=triage_summary.severity_cue,
        key_timestamp=ticket.event_timestamp,
        tags=triage_summary.tags,
        embedding=embedding,
        created_at=datetime.now(timezone.utc),
    )
    await save_capsule(db_pool, capsule)
    return capsule
