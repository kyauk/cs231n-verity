"""
Ingestion API routes.
"""

import asyncpg
from fastapi import APIRouter, HTTPException, Request, status

from db.ticket_repo import save_ticket
from models.api_schemas import FailureTicketRequest, FailureTicketResponse
from services.normalizer import normalize_failure_ticket

router = APIRouter()


@router.post(
    "/ingest/failure_ticket",
    response_model=FailureTicketResponse,
    status_code=status.HTTP_201_CREATED,
)
async def ingest_failure_ticket(
    payload: FailureTicketRequest,
    request: Request,
) -> FailureTicketResponse:
    '''
    Purpose: Accept a failure ticket payload, normalize it, and persist it as a raw ticket.
    Parameters:
    payload (FailureTicketRequest): Structured request body from client.
    request (Request): FastAPI request object containing app-scoped DB pool.
    Returns:
    FailureTicketResponse: Ticket ID and ingestion status for caller tracking.
    Called by: HTTP clients invoking POST /ingest/failure_ticket
    Calls: services.normalizer.normalize_failure_ticket(), db.ticket_repo.save_ticket()
    '''
    db_pool: asyncpg.Pool = request.app.state.db_pool

    try:
        normalized_ticket = normalize_failure_ticket(payload)
        ticket_id = await save_ticket(db_pool, normalized_ticket)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest ticket: {error}",
        ) from error

    return FailureTicketResponse(
        ticket_id=ticket_id,
        status="received",
        message="Failure ticket accepted and stored for downstream triage.",
    )
