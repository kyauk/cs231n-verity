"""
Database connection utilities for asyncpg.
"""

import os

import asyncpg


async def initialize_db_pool() -> asyncpg.Pool:
    '''
    Purpose: Create an asyncpg pool for shared DB access across FastAPI requests.
    Parameters:
    None: Configuration is read from the DATABASE_URL environment variable.
    Returns:
    asyncpg.Pool: Active connection pool for PostgreSQL operations.
    Called by: backend/main.py -> lifespan()
    Calls: asyncpg.create_pool()
    '''
    database_url = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/triage_refinery",
    )
    return await asyncpg.create_pool(dsn=database_url)


async def close_db_pool(pool: asyncpg.Pool) -> None:
    '''
    Purpose: Gracefully close the asyncpg pool during application shutdown.
    Parameters:
    pool (asyncpg.Pool): The connection pool initialized at startup.
    Returns:
    None: This function closes resources and does not return data.
    Called by: backend/main.py -> lifespan()
    Calls: asyncpg.Pool.close()
    '''
    await pool.close()
