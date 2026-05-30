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

Point Verity at your data and your GCS bucket. It segments the data into 8-second windows and uploads them in the canonical layout that the rest of the pipeline reads:

```bash
# Waymo TFRecord (local files):
python -m pipeline.run ingest \
  --source-format waymo_tfrecord \
  --source-root /data/waymo \
  --bucket gs://your-bucket/verity \
  --segments all

# Waymo Parquet (GCS-resident source):
python -m pipeline.run ingest \
  --source-format waymo_parquet \
  --source-root gs://waymo-bucket/validation/camera_image \
  --bucket gs://your-bucket/verity \
  --segments all
```

`--segments` accepts `all`, a comma-separated list, or `@path/to/file.txt` (one ID per line). Existing windows are skipped on re-runs unless you pass `--force`.

### Step 2 — Analyze

Run the analysis — Cosmos-Reason2 annotates each window, Verity finds underrepresented compositions, and Scorer ranks them:

```bash
python -m pipeline.run analyze \
  --bucket gs://your-bucket/verity \
  --output outputs/session-1
```

Typical runtime: ~30 seconds per window (hosted NIM), ~5 seconds on a local A100. Useful flags: `--stub` (offline / CI), `--no-visual` (skip the embedding arm), `--max-workers N` (concurrency), `--sign-as <sa-email>` (see signing options below).

Outputs: `schema_records.json`, `proposals.json`, `scored.json` in the `--output` directory.

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
  --scored outputs/session-1/scored.json \
  --ratings outputs/ratings/ \
  --seeds outputs/seeds.json \
  --output outputs/session-1
```

Or read ratings directly from a running Judge UI server instead of the filesystem:

```bash
python -m pipeline.run report \
  --scored outputs/session-1/scored.json \
  --ratings-url http://localhost:8001 \
  --seeds outputs/seeds.json \
  --output outputs/session-1
```

The **seeds file** is a JSON document you pre-register before any analysis run — it lists the windows you want recall measured against and labels each as `familiar` or `unfamiliar`:

```json
{
  "seeded_windows": [
    {"window": "seg_001/0000", "subset": "familiar"},
    {"window": "seg_002/0001", "subset": "unfamiliar"}
  ]
}
```

The report writes `report.json`, `report.md`, and `report.html` (in a timestamped subdirectory) with:
- Scenario recall at K=10, K=30, and full set
- Per-scenario plausibility and frontier difficulty scores
- Inter-rater agreement (Krippendorff's α)
- Differential examples where reviewers disagreed most

### GCS signed URLs — three working setups

Step 2 (`analyze`) needs to give Cosmos-Reason2 a URL it can fetch each video from. v4 signed-URL generation **requires a private-key signer** — a user refresh-token ADC cannot sign. Pick one of:

| Setup | When to use | How |
|---|---|---|
| **Service-account key file** | Local dev, single-operator runs | Export `GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa-key.json` instead of using `gcloud auth login`. |
| **Service-account impersonation** | Production / shared infra, no keys on disk | Keep user ADC, grant your user `roles/iam.serviceAccountTokenCreator` on a reader SA, then pass `--sign-as reader@your-project.iam.gserviceaccount.com` to `analyze`. |
| **Public-read bucket** | Quick PoCs, non-sensitive data only | `gsutil iam ch allUsers:objectViewer gs://your-bucket` — *do not use for production data.* |

If you see `WindowStorageError: Signed URL generation failed`, you're missing one of these.

### Quick-analysis path: I already have MP4s

If you have a flat GCS bucket of MP4 segment videos (filenames as IDs) and want a fast read on what's underrepresented in them, skip `ingest` entirely:

```bash
# Single-camera bucket: files like gs://my-mp4s/drives/drive_001.mp4
python -m pipeline.run analyze \
  --bucket gs://my-mp4s/drives \
  --storage-mode flat_mp4 \
  --cameras FRONT \
  --output outputs/quick

# Multi-camera bucket: files like gs://my-mp4s/drives/drive_001_FRONT.mp4
python -m pipeline.run analyze \
  --bucket gs://my-mp4s/drives \
  --storage-mode flat_mp4 \
  --cameras FRONT,FRONT_LEFT,FRONT_RIGHT,SIDE_LEFT,SIDE_RIGHT \
  --output outputs/full
```

**`--cameras` is required in flat mode** — you must declare which cameras your MP4s contain so the visual-arm embedding dimensionality is explicit. Filename convention is decided by your camera count:

| `--cameras` count | Filename pattern | Example |
|---|---|---|
| 1 | `<segment_id>.mp4` | `drive_001.mp4` |
| 2+ | `<segment_id>_<camera>.mp4` | `drive_001_FRONT.mp4` |

Constraints of flat mode: each MP4 becomes one Verity "window" (no auto-slicing). For windowed analysis of large drives, use the canonical `ingest` → `analyze` path instead.

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

**"WindowStorageError: Signed URL generation failed"**
You're authenticated against GCS but the credentials cannot sign v4 URLs (e.g. a user refresh-token has no private key). See **GCS signed URLs** above — pick one of the three setups.

**"NVIDIA_API_KEY is not set" from `analyze`**
Either set the env var (or load `.env`) or pass `--stub` to run offline with the canned stub clients (useful for CI and smoke tests).

**Lots of `vlm_unavailable` failures on a fresh NIM key (HTTP 429)**
The default `--max-workers 8` is too aggressive for build.nvidia.com free-tier rate limits. Drop to `--max-workers 1` or `2` for your first runs. Each failed window is logged to stderr; the batch continues, no silent data loss.

**`Ctrl-C` during `analyze` takes a while to actually exit**
The `ThreadPoolExecutor` waits for in-flight VLM HTTP calls to finish before shutting down. The OpenAI client has no default timeout, so an interrupt may block until the slowest in-flight call returns or fails. Expected behavior; not a hang. To bail faster, kill the process directly.

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
