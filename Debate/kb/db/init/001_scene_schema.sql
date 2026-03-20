CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS scene_embeddings (
    id SERIAL PRIMARY KEY,
    scene_token TEXT NOT NULL,
    log_id TEXT NOT NULL,
    scenario_tags TEXT[],
    embedding vector(256) NOT NULL,
    quality_stats JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_scene_embeddings_hnsw
    ON scene_embeddings USING hnsw (embedding vector_cosine_ops);
