# Waymo Discovery Pipeline

Waymo-dataset counterpart of the legacy `pipeline/` package. It mirrors the
same discovery -> embed -> cluster -> reason flow, adapted for the Waymo Open
Dataset v2 (five-camera rig, GCS storage), and adds a FastAPI runner that
serves the Verity frontend with live data.

## Layout

| File | Mirrors (legacy) | Purpose |
|------|------------------|---------|
| `waymo_video_pipeline.py` | `pipeline/waymo_video_pipeline.py` | Waymo Parquet -> per-camera MP4 -> GCS, signed-URL `segment_index.json` |
| `waymo_extract_scene_windows.py` | `pipeline/extract_scene_windows.py` + `retrieve_scene_windows_s3.py` | Slice Waymo segments into encoder-agnostic scene windows |
| `waymo_embed_scenes.py` | `pipeline/embed_scenes.py` | Embed camera clips via the Cosmos Embed1 NIM (5 x 256 = 1280-d) |
| `waymo_cluster_embeddings.py` | `pipeline/cluster_embeddings.py` | Two-pass UMAP + HDBSCAN; emits `flagged_windows.jsonl` |
| `waymo_populate_pgvector.py` | `pipeline/populate_pgvector.py` | Load embeddings into `waymo_window_embeddings` pgvector table |
| `waymo_describe_and_debate.py` | `relevant_video_debate_files/pipeline/stage_describe_and_debate.py` | VLM scene description + proponent/critic/judge debate |
| `waymo_runner.py` | `relevant_video_debate_files/pipeline/remote_gpu_runner.py` | FastAPI server consumed by the frontend |
| `store.py` | — | JSON-file-backed state for batch jobs |

> The directory is `waymo-pipeline/` (as requested). A `waymo_pipeline` symlink
> at the repo root makes the package importable by Python (`-` is not valid in
> a module name). Keep the symlink when committing.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r waymo-pipeline/requirements.txt
cp .env.example .env   # fill NVIDIA_API_KEY, DATABASE_URL, etc.
```

GCS access (Waymo buckets) uses Application Default Credentials — run on a BREV
instance with the `waymo-video-pipeline` service account attached, or
`gcloud auth application-default login` locally.

## Running the runner

```bash
# from the repo root, so `python -m waymo_pipeline.*` resolves
uvicorn waymo_pipeline.waymo_runner:app --host 0.0.0.0 --port 8000
```

The frontend talks to this server via `NEXT_PUBLIC_API_URL` (default
`http://localhost:8000`). The runner serves permissive CORS.

## Endpoints

| Method | Path | Frontend tab |
|--------|------|--------------|
| GET | `/health` | connectivity probe |
| GET | `/batches` | Ingest — batch history |
| POST | `/batches` | Ingest — launch batch (runs extract/embed/cluster) |
| GET | `/cluster-space` | Cluster Space — 3D points + cluster stats |
| GET | `/scenes/{id}` | Cluster Space — scene modal video + annotations |
| GET | `/scenarios` | Dashboard — flagged scenarios |
| POST | `/analysis/run-stream` | Analysis — SSE-streamed agentic debate |

## Pipeline (manual / batch use)

```bash
python -m waymo_pipeline.waymo_video_pipeline       --num-segments 20
python -m waymo_pipeline.waymo_extract_scene_windows --output-jsonl outputs/waymo/waymo_scene_windows.jsonl --max-segments 5
python -m waymo_pipeline.waymo_embed_scenes          --input-jsonl outputs/waymo/waymo_scene_windows.jsonl --output-jsonl outputs/waymo/waymo_window_embeddings.jsonl
python -m waymo_pipeline.waymo_cluster_embeddings    --input-jsonl outputs/waymo/waymo_window_embeddings.jsonl --output-npz outputs/waymo/waymo_clusters.npz --output-jsonl outputs/waymo/waymo_clusters.jsonl --flagged-jsonl outputs/waymo/flagged_windows.jsonl
python -m waymo_pipeline.waymo_populate_pgvector     --input-jsonl outputs/waymo/waymo_window_embeddings.jsonl --embedding-dim 1280
```

The runner's `POST /batches` runs the extract/embed/cluster stages
automatically as a background task.
