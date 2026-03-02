"""
Repository functions for failure capsule persistence.
"""

from datetime import datetime, timezone
from typing import Any

import asyncpg

from models.db_records import FailureCapsuleRecord


def _to_vector_literal(vector: list[float]) -> str:
    '''
    Purpose: Convert a Python float list into pgvector literal syntax for SQL inserts.
    Parameters:
    vector (list[float]): Embedding values produced by the embedding model call.
    Returns:
    str: pgvector-compatible literal string in bracketed comma-separated format.
    Called by: backend/db/capsule_repo.py -> save_capsule()
    Calls: None
    '''
    return "[" + ",".join(str(value) for value in vector) + "]"


def _parse_embedding(value: Any) -> list[float]:
    '''
    Purpose: Normalize embedding values returned by asyncpg into list[float] shape.
    Parameters:
    value (Any): Raw DB column value for the pgvector embedding field.
    Returns:
    list[float]: Parsed float embedding vector, or empty list if null/unsupported.
    Called by: backend/db/capsule_repo.py -> get_capsule()
    Calls: None
    '''
    if value is None:
        return []
    if isinstance(value, list):
        return [float(item) for item in value]
    if isinstance(value, tuple):
        return [float(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip().strip("[]")
        if not stripped:
            return []
        return [float(item) for item in stripped.split(",")]
    return []


async def save_capsule(pool: asyncpg.Pool, capsule: FailureCapsuleRecord) -> str:
    '''
    Purpose: Persist a normalized failure capsule record for retrieval and triage workflows.
    Parameters:
    pool (asyncpg.Pool): Shared asyncpg connection pool.
    capsule (FailureCapsuleRecord): Typed capsule payload to insert into failure_capsules.
    Returns:
    str: Persisted capsule UUID string.
    Called by: backend/agents/capsule_generation.py -> generate_capsule_node()
    Calls: asyncpg.Pool.acquire(), asyncpg.Connection.execute()
    '''
    query = """
    INSERT INTO failure_capsules (
        capsule_id,
        ticket_id,
        triage_summary,
        scenario_type,
        failure_mode_hints,
        likely_subsystem,
        severity_cue,
        key_timestamp,
        tags,
        embedding,
        created_at
    )
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::vector, $11)
    """

    async with pool.acquire() as connection:
        await connection.execute(
            query,
            capsule.capsule_id,
            capsule.ticket_id,
            capsule.triage_summary,
            capsule.scenario_type,
            capsule.failure_mode_hints,
            capsule.likely_subsystem,
            capsule.severity_cue,
            capsule.key_timestamp,
            capsule.tags,
            _to_vector_literal(capsule.embedding),
            capsule.created_at,
        )

    return capsule.capsule_id


async def get_capsule(pool: asyncpg.Pool, capsule_id: str) -> FailureCapsuleRecord | None:
    '''
    Purpose: Fetch a stored failure capsule by capsule_id for API readback.
    Parameters:
    pool (asyncpg.Pool): Shared asyncpg connection pool.
    capsule_id (str): UUID string identifying the capsule row.
    Returns:
    FailureCapsuleRecord | None: Typed capsule record if found, otherwise None.
    Called by: backend/api/capsule.py -> fetch_failure_capsule()
    Calls: asyncpg.Pool.fetchrow()
    '''
    query = """
    SELECT
        capsule_id,
        ticket_id,
        triage_summary,
        scenario_type,
        failure_mode_hints,
        likely_subsystem,
        severity_cue,
        key_timestamp,
        tags,
        embedding,
        created_at
    FROM failure_capsules
    WHERE capsule_id = $1
    """
    row = await pool.fetchrow(query, capsule_id)
    if row is None:
        return None

    key_timestamp = row["key_timestamp"]
    created_at = row["created_at"]
    if key_timestamp is not None and key_timestamp.tzinfo is None:
        key_timestamp = key_timestamp.replace(tzinfo=timezone.utc)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)

    return FailureCapsuleRecord(
        capsule_id=str(row["capsule_id"]),
        ticket_id=str(row["ticket_id"]),
        triage_summary=row["triage_summary"],
        scenario_type=row["scenario_type"],
        failure_mode_hints=row["failure_mode_hints"] or [],
        likely_subsystem=row["likely_subsystem"],
        severity_cue=row["severity_cue"],
        key_timestamp=key_timestamp,
        tags=row["tags"] or [],
        embedding=_parse_embedding(row["embedding"]),
        created_at=created_at if isinstance(created_at, datetime) else datetime.now(timezone.utc),
    )
