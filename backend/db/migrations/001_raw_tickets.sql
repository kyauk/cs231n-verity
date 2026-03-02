CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS raw_tickets (
    ticket_id UUID PRIMARY KEY,
    source_type TEXT NOT NULL,
    source_ref TEXT NOT NULL,
    title TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    event_timestamp TIMESTAMPTZ NOT NULL,
    agent_id TEXT NULL,
    scenario_id TEXT NULL,
    artifacts_ref TEXT[] NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_raw_tickets_source_type ON raw_tickets (source_type);
CREATE INDEX IF NOT EXISTS idx_raw_tickets_event_timestamp ON raw_tickets (event_timestamp);
