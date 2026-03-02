"""
FastAPI application entrypoint for Triage Refinery backend.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.capsule import router as capsule_router
from api.ingest import router as ingest_router
from db.connection import close_db_pool, initialize_db_pool


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize and tear down shared app resources.
    """
    app.state.db_pool = await initialize_db_pool()
    try:
        yield
    finally:
        await close_db_pool(app.state.db_pool)


app = FastAPI(title="Triage Refinery API", version="0.1.0", lifespan=lifespan)
app.include_router(ingest_router)
app.include_router(capsule_router)
