"""
Failure capsule generation and fetch API routes.
"""

import asyncpg
from fastapi import APIRouter, HTTPException, Request, status

from agents.capsule_generation import generate_capsule_node
from db.capsule_repo import get_capsule
from models.api_schemas import (
    FailureCapsuleResponse,
    GenerateCapsuleRequest,
)
from models.db_records import FailureCapsuleRecord

router = APIRouter()


def _to_capsule_response(capsule: FailureCapsuleRecord) -> FailureCapsuleResponse:
    '''
    Purpose: Convert DB capsule records into API response schema objects.
    Parameters:
    capsule (FailureCapsuleRecord): Typed database record returned from repository or agent.
    Returns:
    FailureCapsuleResponse: API-safe response model with camelCase aliases on output.
    Called by: backend/api/capsule.py -> generate_failure_capsule(), fetch_failure_capsule()
    Calls: None
    '''
    return FailureCapsuleResponse(
        capsule_id=capsule.capsule_id,
        ticket_id=capsule.ticket_id,
        triage_summary=capsule.triage_summary,
        scenario_type=capsule.scenario_type,
        failure_mode_hints=capsule.failure_mode_hints,
        likely_subsystem=capsule.likely_subsystem,
        severity_cue=capsule.severity_cue,
        key_timestamp=capsule.key_timestamp,
        tags=capsule.tags,
        created_at=capsule.created_at,
    )


@router.post(
    "/generate/failure_capsule",
    response_model=FailureCapsuleResponse,
    status_code=status.HTTP_201_CREATED,
)
async def generate_failure_capsule(
    payload: GenerateCapsuleRequest,
    request: Request,
) -> FailureCapsuleResponse:
    '''
    Purpose: Generate and persist a failure capsule from an existing raw ticket.
    Parameters:
    payload (GenerateCapsuleRequest): Request containing the target ticket_id.
    request (Request): FastAPI request object containing app-scoped DB pool.
    Returns:
    FailureCapsuleResponse: Generated capsule fields for immediate client use.
    Called by: HTTP clients invoking POST /generate/failure_capsule
    Calls: agents.capsule_generation.generate_capsule_node()
    '''
    db_pool: asyncpg.Pool = request.app.state.db_pool

    try:
        capsule = await generate_capsule_node(payload.ticket_id, db_pool)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate capsule: {error}",
        ) from error

    return _to_capsule_response(capsule)


@router.get(
    "/fetch/failure_capsule/{capsule_id}",
    response_model=FailureCapsuleResponse,
    status_code=status.HTTP_200_OK,
)
async def fetch_failure_capsule(capsule_id: str, request: Request) -> FailureCapsuleResponse:
    '''
    Purpose: Fetch a previously generated failure capsule by capsule identifier.
    Parameters:
    capsule_id (str): UUID string identifying the stored failure capsule.
    request (Request): FastAPI request object containing app-scoped DB pool.
    Returns:
    FailureCapsuleResponse: Stored capsule fields if the capsule exists.
    Called by: HTTP clients invoking GET /fetch/failure_capsule/{capsule_id}
    Calls: db.capsule_repo.get_capsule()
    '''
    db_pool: asyncpg.Pool = request.app.state.db_pool
    capsule = await get_capsule(db_pool, capsule_id)
    if capsule is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Capsule not found for capsule_id={capsule_id}",
        )
    return _to_capsule_response(capsule)
