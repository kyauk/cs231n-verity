# Verity — AV Safety Scenario Discovery

Verity takes your fleet data and finds the edge cases your system hasn't been tested against — driving conditions that are statistically under-represented in your data but physically plausible and likely to expose gaps in your AV stack.

Bring your fleet data. Verity surfaces what's underrepresented, ranks the highest-risk gaps, and gives your team a structured way to validate and prioritize them.

---

## What You Get

After a run on your fleet data, Verity delivers:

- **A ranked gap list** — underrepresented scene combinations (e.g. "pedestrian in fog at uncontrolled intersection, ego turning left") sorted by how much rarer they are in your data than chance would predict, filtered for physical plausibility and AV difficulty
- **Supporting evidence** — the actual data windows where each scenario occurs, surfaced in the review UI so your team can validate whether the gap is real
- **A coverage report** — seeded recall at K=30, inter-rater agreement, and per-scenario scores you can use to prioritize simulation campaigns or targeted data collection

---

## What You Need

| | |
|---|---|
| **Fleet data** | Waymo Open Dataset format (Parquet or TFRecord) |
| **Cloud storage** | A Google Cloud Storage bucket you control |
| **NVIDIA API Key** | Free tier available at [build.nvidia.com](https://build.nvidia.com) |
| **Python 3.10+** | For the pipeline |
| **pnpm 8+** | For the review UI |

**Optional — to run models on your own GPU instead of NVIDIA's cloud:**

| | |
|---|---|
| **NGC API Key** | Same NVIDIA account as above |
| **GPU instance** | A100 40 GB+ (one per model, or time-share) |
| **Docker + Docker Compose 24+** | For the NIM containers |

---

## Setup

### 1. Install

```bash
git clone <repo-url>
cd NVIDIA-ADR-1

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cd frontend && pnpm install && cd ..
```

### 2. Configure

```bash
cp .env.example .env
```

Open `.env` and fill in the three lines that matter most:

```bash
NVIDIA_API_KEY=nvapi-...                        # your NVIDIA API key
NUPLAN_DATA_ROOT=/path/to/your/waymo/footage    # where your footage lives
```

For your GCS bucket, authenticate once:

```bash
gcloud auth application-default login
```

Or if you're running on a VM, set `GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json`.

### 3. Decide where the models run

**Option A — NVIDIA's cloud (easiest, no GPU needed)**

The default. Your `NVIDIA_API_KEY` is all you need. NVIDIA hosts Cosmos-Reason2 and you pay per call. Leave `NVIDIA_BASE_URL` as-is in `.env`.

**Option B — Your own GPU instance**

If you're on a GPU box (Brev, EC2, bare metal) and want lower latency or higher throughput, run the models locally:

```bash
# Add your NGC_API_KEY to .env first, then:
docker compose --profile gpu up -d
```

This starts Cosmos-Embed1 (port 8080) and Cosmos-Reason2 (port 8081) as local NIM containers. Then update one line in `.env`:

```bash
NVIDIA_BASE_URL=http://localhost:8081/v1
```

Swap back to the cloud at any time by reverting that line.

---

## Running a Discovery Session

### Step 1 — Load your fleet data

Point Verity at your data directory and your GCS bucket. It will segment the data into 8-second windows and store them for analysis:

```bash
python -m pipeline.run ingest \
  --source /path/to/waymo/data \
  --bucket gs://your-bucket/verity
```

This is a one-time step per dataset. Existing windows are skipped on re-runs.

### Step 2 — Analyze

Run the full analysis — Cosmos-Reason2 reads each window and extracts structured scene attributes, then Verity identifies which combinations of conditions are statistically underrepresented and scores them:

```bash
python -m pipeline.run analyze \
  --bucket gs://your-bucket/verity \
  --output outputs/session-1
```

Typical runtime: ~30 seconds per window (hosted NIM), ~5 seconds on a local A100.

### Step 3 — Review findings in the browser

Start the review server and open the UI:

```bash
# Terminal 1 — backend
JUDGE_BUCKET_URI=gs://your-bucket/verity \
JUDGE_PROPOSALS_PATH=outputs/session-1/proposals.json \
uvicorn pipeline.modules.judge_ui.server:app --port 8001

# Terminal 2 — frontend
cd frontend && pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) and go to the **Judge** tab.

Your team sees a ranked list of candidate edge cases. For each one, you can:
- Inspect the data windows that motivated the discovery
- Score it for **coherence** (is this a real scenario your system could encounter?) and **usefulness** (does closing this gap matter for your safety case?)
- Add a free-text note

Multiple reviewers can rate independently — ratings are merged automatically in the final report.

### Step 4 — Export your report

```bash
python -m pipeline.run report \
  --output outputs/session-1
```

This writes `outputs/session-1/report.json`, `report.md`, and `report.html` with:
- Scenario recall at K=10, K=30, and full set
- Per-scenario plausibility and frontier difficulty scores
- Inter-rater agreement (Krippendorff's α)
- Differential examples where reviewers disagreed most

---

## What the Scenario Scores Mean

| Score | What it measures | Range |
|---|---|---|
| **Novelty** | How much rarer the scenario is than chance predicts | Higher = rarer |
| **Plausibility** | Whether Cosmos-Reason2 judges the scenario physically possible | 0–1 |
| **Frontier difficulty** | Whether the scenario is likely to challenge current AV systems | 0–1 |
| **Final rank** | `novelty × 0.4 + plausibility × 0.3 + difficulty × 0.3` | Higher = higher priority |

Scenarios that score high on all three are the ones most worth prioritizing for simulation or targeted data collection.

---

## Troubleshooting

**"VLMUnavailableError" at annotation time**
Check that `NVIDIA_API_KEY` is set and valid. If using local NIM, confirm containers are healthy: `docker compose ps`.

**"WindowStorageError" when reading fleet data**
Your GCS credentials may have expired. Re-run `gcloud auth application-default login` or check your service account key.

**Annotation is slow**
Switch to Option B (local GPU). On a hosted A100, annotation runs ~6× faster than the NVIDIA cloud free tier.

**The review UI shows no proposals**
The analysis step must complete before starting the Judge server. Check that `outputs/session-1/proposals.json` exists and is non-empty.

**No gaps found after analysis**
Verity needs enough data volume to detect statistical underrepresentation. If your dataset is small (< 50 windows), try running against a larger slice of your fleet data or contact us to adjust the detection thresholds.

---

## Configuration Reference

A full list of settings is in [`.env.example`](.env.example). The ones you're most likely to change:

| Setting | What it controls |
|---|---|
| `NVIDIA_API_KEY` | Authentication for NVIDIA NIM (hosted) |
| `NVIDIA_BASE_URL` | Switch between NVIDIA cloud and local NIM |
| `NUPLAN_DATA_ROOT` | Root directory of your fleet data |
| `NGC_API_KEY` | Authentication for local NIM containers |

---

## Technical Reference

For output schemas, failure modes, module contracts, and pipeline internals, see [`pipeline/README.md`](pipeline/README.md).
