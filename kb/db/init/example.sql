-- db/initexample.sql
-- Minimal pgvector schema + tiny seed data for quick sanity checks.
-- Run inside your target database (e.g., verity).

-- 1) Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- 2) Table
CREATE TABLE IF NOT EXISTS chunks (
  id BIGSERIAL PRIMARY KEY,
  source_id TEXT,
  chunk_text TEXT NOT NULL,
  embedding VECTOR(1536),
  cluster_id INTEGER,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3) Metadata indexes
CREATE INDEX IF NOT EXISTS idx_chunks_source_id ON chunks (source_id);
CREATE INDEX IF NOT EXISTS idx_chunks_created_at ON chunks (created_at);
CREATE INDEX IF NOT EXISTS idx_chunks_cluster_id ON chunks (cluster_id);

-- 4) HNSW vector index for fast cosine similarity search
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
  ON chunks USING hnsw (embedding vector_cosine_ops);

-- 5) Seed rows (only if the table is empty)
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM chunks LIMIT 1) THEN
    INSERT INTO chunks (source_id, chunk_text, embedding)
    VALUES
      (
        'demo:overheat',
        'Battery overheated during fast charging; device shut down to prevent damage.',
        -- Fake embedding: 1536 dimensions of 0.0
        -- This is just to prove inserts work; replace with real embeddings from your pipeline.
        array_fill(0.0::real, ARRAY[1536])::vector
      ),
      (
        'demo:sensor-drift',
        'IMU sensor drift increased in humid conditions, causing navigation error accumulation.',
        array_fill(0.0::real, ARRAY[1536])::vector
      );
  END IF;
END $$;

-- 6) Sanity check
SELECT id, source_id, created_at, left(chunk_text, 80) AS preview
FROM chunks
ORDER BY id;