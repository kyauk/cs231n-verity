"""
Repository functions for raw ticket persistence.
"""

from datetime import datetime, timezone

import asyncpg

from models.db_records import RawTicketRecord


async def save_ticket(pool: asyncpg.Pool, ticket: RawTicketRecord) -> str:
    '''
    Purpose: Persist an immutable raw ticket record for downstream triage pipeline use.
    Parameters:
    pool (asyncpg.Pool): Shared asyncpg connection pool.
    ticket (RawTicketRecord): Typed ticket record to insert into raw_tickets.
    Returns:
    str: Persisted ticket UUID string.
    Called by: backend/api/ingest.py -> ingest_failure_ticket()
    Calls: asyncpg.Pool.acquire(), asyncpg.Connection.execute()
    '''
    query = """
    INSERT INTO raw_tickets (
        ticket_id,
        source_type,
        source_ref,
        title,
        raw_text,
        event_timestamp,
        agent_id,
        scenario_id,
        artifacts_ref,
        created_at
    )
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
    """

    async with pool.acquire() as connection:
        await connection.execute(
            query,
            ticket.ticket_id,
            ticket.source_type,
            ticket.source_ref,
            ticket.title,
            ticket.raw_text,
            ticket.event_timestamp,
            ticket.agent_id,
            ticket.scenario_id,
            ticket.artifacts_ref,
            ticket.created_at,
        )

    return ticket.ticket_id


async def get_ticket(pool: asyncpg.Pool, ticket_id: str) -> RawTicketRecord | None:
    '''
    Purpose: Fetch a raw ticket by ticket_id for readback and downstream processing.
    Parameters:
    pool (asyncpg.Pool): Shared asyncpg connection pool.
    ticket_id (str): UUID string identifying the ticket row.
    Returns:
    RawTicketRecord | None: Typed ticket record if found, otherwise None.
    Called by: backend/tests/test_ingest.py
    Calls: asyncpg.Pool.fetchrow()
    '''
    query = """
    SELECT
        ticket_id,
        source_type,
        source_ref,
        title,
        raw_text,
        event_timestamp,
        agent_id,
        scenario_id,
        artifacts_ref,
        created_at
    FROM raw_tickets
    WHERE ticket_id = $1
    """

    row = await pool.fetchrow(query, ticket_id)
    if row is None:
        return None

    event_timestamp = row["event_timestamp"]
    created_at = row["created_at"]
    if event_timestamp.tzinfo is None:
        event_timestamp = event_timestamp.replace(tzinfo=timezone.utc)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)

    return RawTicketRecord(
        ticket_id=str(row["ticket_id"]),
        source_type=row["source_type"],
        source_ref=row["source_ref"],
        title=row["title"],
        raw_text=row["raw_text"],
        event_timestamp=event_timestamp,
        agent_id=row["agent_id"],
        scenario_id=row["scenario_id"],
        artifacts_ref=row["artifacts_ref"],
        created_at=created_at if isinstance(created_at, datetime) else datetime.now(timezone.utc),
    )
