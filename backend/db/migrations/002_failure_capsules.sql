CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS failure_capsules (
    capsule_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ticket_id UUID NOT NULL REFERENCES raw_tickets(ticket_id),
    triage_summary TEXT NOT NULL,
    scenario_type TEXT NULL,
    failure_mode_hints TEXT[] NOT NULL DEFAULT '{}',
    likely_subsystem TEXT NULL,
    severity_cue TEXT NOT NULL DEFAULT 'unknown',
    key_timestamp TIMESTAMPTZ NULL,
    tags TEXT[] NOT NULL DEFAULT '{}',
    embedding vector(1536),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_capsules_ticket_id ON failure_capsules (ticket_id);
CREATE INDEX IF NOT EXISTS idx_capsules_severity ON failure_capsules (severity_cue);
CREATE INDEX IF NOT EXISTS idx_capsules_embedding_hnsw
    ON failure_capsules USING hnsw (embedding vector_cosine_ops);
