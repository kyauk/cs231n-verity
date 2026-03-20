# Hand-off: UMAP/HDBSCAN → Cosmos-Reason2 (from existing embeddings)

**Audience:** Fellow engineer taking over. Goal: run from existing embeddings, prove the UMAP → HDBSCAN → Cosmos-Reason2 pipeline. No S3 retrieval or embed step required.

---

## 1. Docker setup (PostgreSQL and related)

Local run depends on Docker for PostgreSQL (and optionally pgAdmin). Do this first.

### Services

- **PostgreSQL + pgvector:** The `db` service in [docker-compose.yml](docker-compose.yml) uses image `pgvector/pgvector:pg16`, loads env from `.env`, exposes port 5432, and mounts init scripts from [kb/db/init/](kb/db/init/) (e.g. [001_scene_schema.sql](kb/db/init/001_scene_schema.sql) for `uuid-ossp`, `vector` extension, and tables).
- **pgAdmin (optional):** The `pgadmin` service uses `dpage/pgadmin4`, maps host port 5050 to container port 80, and reads `PGADMIN_DEFAULT_EMAIL` and `PGADMIN_DEFAULT_PASSWORD` from `.env`.

### Commands

1. Copy env and set at least PostgreSQL variables:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and set (see [.env.example](.env.example)):
   - `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_HOST`, `POSTGRES_PORT`
   - Optionally `DATABASE_URL` (e.g. `postgresql+asyncpg://postgres:postgres@localhost:5432/verity`)
   - If using pgAdmin: `PGADMIN_DEFAULT_EMAIL`, `PGADMIN_DEFAULT_PASSWORD`

2. Start the database (and optionally pgAdmin):
   ```bash
   docker compose up -d db
   docker compose up -d pgadmin   # optional; then open http://localhost:5050
   ```

**Optional reference:** The same [docker-compose.yml](docker-compose.yml) defines a `cosmos-embed1` service under the `gpu` profile (`docker compose --profile gpu up -d`). You do **not** need it for this hand-off path (you use existing embeddings).

---

## 2. Prerequisites (Python, deps, input = embeddings)

- **Python:** 3.11+, venv recommended.
- **Dependencies:** From [requirements.txt](requirements.txt): `umap-learn`, `hdbscan`, `numpy`, `matplotlib`, `scikit-learn`, and optionally `plotly` for interactive HTML. No GPU required for UMAP/HDBSCAN.
- **Input = existing embeddings:** You do **not** run S3 retrieval or the embed stage. Your starting point is an existing **window embeddings JSONL** (e.g. `outputs/window_embeddings_cosmos.jsonl`). You can use the **Cosmos** window embeddings the team is not using for the main pipeline—either shared in the repo or from a teammate. Each line must have `embedding` (list of float), `window_id`, `scene_token_hex`, `log_id`, and related fields; see [pipeline/models/scene_window.py](pipeline/models/scene_window.py) `WindowEmbeddingRecord`. If you have no embeddings file, see **Optional: Regenerate embeddings locally** below.

---

## 3. Step 1: UMAP + HDBSCAN (visualization and flagged points)

Script: [pipeline/visualize_embeddings.py](pipeline/visualize_embeddings.py). Input: your existing embeddings JSONL (e.g. `outputs/window_embeddings_cosmos.jsonl`).

### Commands

Basic 2D plot:
```bash
python -m pipeline.visualize_embeddings \
  --input-jsonl outputs/window_embeddings_cosmos.jsonl \
  --output outputs/embedding_umap.png
```

With HDBSCAN and interactive HTML (recommended):
```bash
python -m pipeline.visualize_embeddings \
  --input-jsonl outputs/window_embeddings_cosmos.jsonl \
  --run-hdbscan \
  --output outputs/embedding_umap.png \
  --output-html outputs/embedding_umap.html
```

Optional flags: `--reducer pca` for a faster run; `--max-points 5000` to cap the number of points.

### How to interpret

- **Noise (label -1):** Points HDBSCAN did not assign to any cluster—candidate anomalies.
- **Small clusters:** Also of interest for rarity.
- Use the HTML hover (`window_id`, `log_id`) to map points back to scenes.

### Flagged points for Reason2

The script does **not** currently write a machine-readable list of flagged `window_id`s. For the Cosmos-Reason2 step you will need either to (a) add an optional `--output-flagged-jsonl` to this script to export window_ids and labels, or (b) run a small script that loads the same JSONL, runs HDBSCAN again, and writes flagged IDs. Treat "flagged points" as HDBSCAN noise plus optionally small-cluster members; the next step consumes a list of `window_id`s and their metadata for reasoning.

---

## 4. Step 2: Attach Cosmos-Reason2 (integration point)

**Goal:** For each flagged point (or a subset), call Cosmos-Reason2 to produce:

- A **scene description** (what occurred in the window/scene).
- A short **high-fidelity rationale** for why this might be flagged as an anomaly (e.g. rarity, unusual layout, edge case).

**Inputs to provide from the pipeline:** For each flagged `window_id`: at least scene/window metadata (`scene_token_hex`, `log_id`, `scenario_tags`, time range). Ideally also frames or a clip path so Reason2 can do vision-based reasoning; metadata-only prompts are enough to prove the wiring.

**Output:** Structured text or JSON per window, e.g. `scene_description`, `anomaly_rationale`, optional `confidence`.

**Implementation status:** Not implemented in this repo. Your next step is to integrate with NVIDIA Cosmos-Reason2 (e.g. via NIM for VLMs or a vLLM server). Reference: [NVIDIA Cosmos-Reason2](https://docs.nvidia.com/cosmos/latest/reason2/index.html) / NIM API; the model accepts images/video and text prompts for chain-of-thought reasoning.

**Caveat:** With existing Cosmos (or fake) embeddings, anomaly labels come only from UMAP/HDBSCAN; Cosmos-Reason2’s descriptions and rationales may be inaccurate or generic. The objective here is **proof of pipeline** (reduce → cluster → flag → reason), not accuracy of Reason2 output.

---

## 5. Optional: Regenerate embeddings locally (no S3)

If you have no embeddings file, run the **fake** encoder (no GPU, no Cosmos NIM) to produce a valid JSONL for visualization:

1. Copy or symlink the fixture to the expected input path:
   ```bash
   mkdir -p outputs
   cp pipeline/scene_windows_fixture.jsonl outputs/scene_windows.jsonl
   ```
2. Run the embed step with the fake encoder:
   ```bash
   python -m pipeline.embed_scenes \
     --input-jsonl outputs/scene_windows.jsonl \
     --output-jsonl outputs/window_embeddings_cosmos.jsonl \
     --encoder fake
   ```
3. Use `outputs/window_embeddings_cosmos.jsonl` as input to Step 1. No S3 retrieval is involved.

---

## 6. End-to-end flow

```mermaid
flowchart LR
  A[Existing window embeddings JSONL]
  B[UMAP 2D + HDBSCAN]
  C[Flagged points]
  D[Cosmos-Reason2 to be wired]
  E[Scene description + anomaly rationale]
  A --> B
  B --> C
  C --> D
  D --> E
```

If you have no embeddings: **Fake encoder + fixture** → embeddings JSONL → same pipeline from Step 1.

---

## 7. Quick reference

**Docker**
```bash
cp .env.example .env
# Edit .env: set POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB, etc.
docker compose up -d db
docker compose up -d pgadmin   # optional
```

**Visualize (from existing embeddings)**
```bash
python -m pipeline.visualize_embeddings \
  --input-jsonl outputs/window_embeddings_cosmos.jsonl \
  --run-hdbscan \
  --output outputs/embedding_umap.png \
  --output-html outputs/embedding_umap.html
```

**Next:** Implement the Reason2 step that consumes flagged `window_id`s and metadata (and optionally frames/clip) and calls Cosmos-Reason2 to return scene descriptions and anomaly rationales.
