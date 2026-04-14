# Video Upload + Description + Debate (Frontend + Backend Pack)

This folder contains a runnable subset for:

1. upload video from frontend UI
2. scene description (Cosmos HF local model)
3. multi-agent debate (NVIDIA NIM text model)

## Included runtime files

- `pipeline/remote_gpu_runner.py` - FastAPI upload endpoint (`POST /run-video`)
- `pipeline/stage_describe_and_debate.py` - description + debate stage runner
- `pipeline/models/handoff_contracts.py` - typed handoff contracts
- `pipeline/__init__.py`, `pipeline/models/__init__.py` - package markers
- `requirements.txt`, `.env.example`

Frontend is included and runnable (`frontend/` Next.js app + API routes).

## Teammate handoff setup

Use `TEAM_SETUP.md` for a from-scratch Brev GPU runbook, including terminal-by-terminal startup commands, port forwarding, and troubleshooting.

## Backend quick start (FastAPI upload endpoint)

1) Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies:

```bash
pip install -r requirements.txt
```

3) Configure environment:

```bash
cp .env.example .env
```

Fill at least:
- `NVIDIA_API_KEY`
- (optional) model overrides

4) Start the upload API from this folder:

```bash
uvicorn pipeline.remote_gpu_runner:app --host 0.0.0.0 --port 8001
```

5) Upload one video:

```bash
curl -X POST "http://localhost:8001/run-video" \
  -F "video=@/absolute/path/to/clip.mp4"
```

## Notes

- Docker is not required for this flow.
- The description stage loads a large local HF model (`nvidia/Cosmos-Reason2-8B`), so GPU memory and model download access are needed.
- The API writes artifacts to:
  - `inputs/`
  - `outputs/flagged_windows.jsonl`
  - `outputs/flagged_visuals/manifest.jsonl`
  - `outputs/reasoning/*.jsonl`

## Frontend quick start (UI upload flow)

1) Open a second terminal and go to the frontend:

```bash
cd frontend
```

2) Install frontend dependencies:

```bash
npm install
```

3) Start frontend:

```bash
npm run dev
```

4) Open:
- `http://localhost:3000` (main dashboard)
- `http://localhost:3000/video-lab` (standalone upload page)

The frontend upload endpoint is `frontend/app/api/workspace/run-video/route.ts`.

## Frontend runtime modes

- **Local mode (default):** frontend route spawns `python -m pipeline.stage_describe_and_debate` directly.
- **Remote mode (optional):** set `REMOTE_GPU_RUN_URL` in `frontend/.env.local` and frontend will forward uploads to remote runner.

When using local mode, create `.venv` in this folder root so the route auto-detects `../.venv/bin/python`.
