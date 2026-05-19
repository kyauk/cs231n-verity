"""Waymo discovery pipeline package.

Mirrors the logic of the legacy ``pipeline/`` package but adapted for the
Waymo Open Dataset. Assumes Waymo segment MP4s are already reconstructed and
available in GCS (see ``waymo_video_pipeline.py`` in the legacy package).

Stages:
  1. waymo_extract_scene_windows  -- slice segments into encoder-agnostic scene windows
  2. waymo_embed_scenes           -- embed windows via Cosmos Embed1 NIM
  3. waymo_cluster_embeddings     -- two-pass UMAP + HDBSCAN clustering
  4. waymo_populate_pgvector      -- load embeddings into PostgreSQL pgvector

Serving:
  waymo_runner  -- FastAPI app exposing the interface the frontend consumes
                   (batch launch/list, cluster space, scenes, SSE agentic analysis).
"""

__version__ = "1.0.0"
