# Verity — Adversarial Environment Generator

Autonomous Vehicle safety validation pipeline. Ingests fleet footage, embeds scenes with NVIDIA Cosmos, clusters them in UMAP space, and runs a multi-agent LLM debate to surface the highest-priority adversarial scenarios for sim testing.

[Wiki](https://github.com/cs210/NVIDIA-ADR-1/wiki)

---

## How it works

```
GCS driving data
      │
      ▼
[Stage 1] MP4 conversion       waymo_video_pipeline.py
      │
      ▼
[Stage 2] Scene window extraction   waymo_extract_scene_windows.py
      │
      ▼
[Stage 3] Cosmos Embed1 NIM         waymo_embed_scenes.py
          (1280-d per scene, concurrent)
      │
      ▼
[Stage 4] UMAP + HDBSCAN            waymo_cluster_embeddings.py
          → cluster JSONL + flagged scenarios
      │
      ▼
FastAPI backend  ←──SSE──→  Next.js frontend (Verity UI)
```

The UI has four tabs:

| Tab | Purpose |
|-----|---------|
| **Ingest** | Point Verity at a GCS dataset path and launch a batch |
| **Cluster Space** | 3-D UMAP scatter plot — click any point to inspect the scene |
| **Analysis** | Proponent → Critic → Judge multi-agent debate on a selected scene |
| **Dashboard** | Ranked list of flagged high-priority scenarios |

---

## Repo layout

```
NVIDIA-ADR-1/
├── frontend/               Next.js 14 app (TypeScript, Tailwind v4, shadcn/ui)
│   ├── app/page.tsx        Root page — tab state, data fetching
│   ├── components/         Tab components + shared UI
│   └── lib/
│       ├── api.ts          All fetch calls to FastAPI backend
│       └── types.ts        Shared TypeScript types
│
├── waymo_pipeline/         Python backend package
│   ├── waymo_runner.py     FastAPI server (entry point)
│   ├── waymo_video_pipeline.py       Stage 1 — Parquet → MP4
│   ├── waymo_extract_scene_windows.py Stage 2 — scene window extraction
│   ├── waymo_embed_scenes.py          Stage 3 — Cosmos embedding
│   └── waymo_cluster_embeddings.py   Stage 4 — UMAP + HDBSCAN
│
├── pipeline/               Original pipeline (reference implementation)
├── smoke_test.py           End-to-end smoke test with synthetic embeddings
├── .env.example            Required environment variables
└── README.md               This file
```

---

## Prerequisites

- Python 3.10+
- Node.js 18+
- `gcloud` CLI (for GCS access)
- NVIDIA NIM API key (Cosmos Embed1 + LLM endpoints)
- GCS bucket with write access (for processed MP4s)

---

## Setup

### 1. Clone and create virtualenv

```bash
git clone https://github.com/cs210/NVIDIA-ADR-1.git
cd NVIDIA-ADR-1
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r waymo_pipeline/requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env with your values
```

Key variables:

| Variable | Description |
|----------|-------------|
| `NVIDIA_API_KEY` | NIM API key from build.nvidia.com |
| `COSMOS_EMBED_URL` | Cosmos Embed1 NIM endpoint URL |
| `NVIDIA_LLM_MODEL` | LLM model ID for the debate agents |
| `WAYMO_DEST_BUCKET` | GCS bucket for storing processed MP4s |
| `WAYMO_SOURCE_BUCKET` | Source dataset bucket (default: waymo_open_dataset_v_2_0_1) |
| `WAYMO_SOURCE_PREFIX` | Path prefix inside source bucket |

### 3. Authenticate with Google Cloud

The pipeline reads from and writes to GCS using Application Default Credentials — no credentials are entered in the UI.

```bash
gcloud auth application-default login
```

Re-run every ~60 days, or configure a service account key for production.

### 4. Install frontend dependencies

```bash
cd frontend
npm install
```

---

## Running locally

Open two terminals from the repo root.

**Terminal 1 — Backend**

```bash
source .venv/bin/activate
cd waymo_pipeline
uvicorn waymo_runner:app --reload --port 8000
```

**Terminal 2 — Frontend**

```bash
cd frontend
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

---

## Smoke test (no real data needed)

Generates 88 synthetic 1280-d embeddings (4 tight clusters + 8 noise points), runs the full clustering stage, and writes outputs that the UI can load:

```bash
source .venv/bin/activate
python smoke_test.py
```

Then start the backend and frontend — Cluster Space will show the synthetic clusters immediately. Analysis won't return real outputs (no actual video), but you can verify the agent stepper UI and history panel.

---

## UI walkthrough

### Ingest tab

1. Expand **"How to connect your dataset"** for setup instructions and example URIs.
2. Paste a `gs://bucket/path` URI or click **"Use this"** next to a dataset example.
3. Set a **Batch Label** (e.g., `Phoenix Q4 Highway`) and pick a **Region**.
4. Click **Launch Batch**. The batch appears in the history table below with a live status.
5. When status turns **Completed**, click **View Clusters** to jump to Cluster Space.

### Cluster Space tab

- Drag to rotate, scroll to zoom.
- Click any point to open the **Scene Details** modal — shows the video clip, environment metadata, and detected events.
- Click **Analyze this scene** to send the scene to the Analysis tab.
- The right panel shows cluster statistics and a per-cluster density bar.
- Noise points (red) are scenes that didn't fit any cluster — often the most unusual.

### Analysis tab

- A scene must be selected first (via Cluster Space → "Analyze this scene").
- Click **Generate Analysis** to start the three-agent debate:
  - **Proposer** — proposes an adversarial scenario based on the scene
  - **Critic** — challenges the proposal for safety coverage gaps
  - **Judge** — weighs the debate and issues a verdict + priority score (0–100)
- Each agent's terminal streams output live as the pipeline runs.
- Completed analyses are saved to browser `localStorage` and appear under **History**.

### Dashboard tab

- Shows all flagged high-priority scenarios across all ingested batches.
- Sorted by priority score descending.
- Click **View** on any row to jump to its Analysis.

---

## Deploying on Brev

1. Provision a Brev instance (GPU recommended for Cosmos embedding; CPU-only works for clustering + debate).
2. Clone the repo and follow the [Setup](#setup) steps above.
3. Set your NIM API key and GCS credentials on the instance.
4. Run the backend with `--host 0.0.0.0`:
   ```bash
   uvicorn waymo_pipeline.waymo_runner:app --host 0.0.0.0 --port 8000
   ```
5. Update `NEXT_PUBLIC_API_URL` in `frontend/.env.local` to point at the Brev instance's public URL.
6. Run the frontend (`npm run dev` or `npm run build && npm start`).

---

## Architecture notes

- **Embedding concurrency**: `waymo_embed_scenes.py` uses a `ThreadPoolExecutor` with a `threading.Lock` on the JSONL write so workers don't corrupt output.
- **Two-pass UMAP**: a 50-d pass feeds HDBSCAN for clustering; a separate 3-d pass produces the visualization coordinates. This avoids the crowding artifacts that come from clustering directly in 3-d.
- **SSE progress**: the runner pipes subprocess stdout line-by-line over Server-Sent Events so the UI updates in real time without polling.
- **No credentials in UI**: the Ingest tab accepts a GCS URI but never touches credentials — the backend machine's ADC handles auth transparently.
- **Scene modal video**: the backend generates a short-lived signed URL for each scene clip; `cluster-space-tab.tsx` passes it directly to a `<video>` element.
