"""Centralised configuration: DB connection and OpenAI client."""

from __future__ import annotations

import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from openai import OpenAI
from pgvector.psycopg import register_vector

_ENV_LOADED = False

# Ensure the environment variables are loaded
def _ensure_env() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)
    _ENV_LOADED = True


def get_database_url() -> str:
    """Return the database URL."""
    _ensure_env()
    user = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    db = os.environ.get("POSTGRES_DB", "verity")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def get_db_conn() -> psycopg.Connection:
    """Return a new psycopg3 connection with pgvector type registered. Lets python code query the database and get results back"""
    conn = psycopg.connect(get_database_url(), autocommit=True)
    register_vector(conn)
    return conn


def get_openai_client() -> OpenAI:
    """Return a new OpenAI client."""
    _ensure_env()
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])
