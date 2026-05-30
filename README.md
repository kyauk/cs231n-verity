# Verity — AV Safety Scenario Discovery

Verity takes your fleet data and finds the edge cases your system hasn't been tested against — driving conditions that are statistically under-represented in your data but physically plausible and likely to expose gaps in your AV stack.

Bring your fleet data. Verity surfaces what's underrepresented, ranks the highest-risk gaps, and gives your team a structured way to validate and prioritize them.

---

## Status (2026-05-30)

Pipeline is end-to-end runnable. **657 tests passing, full hygiene protocol signed off for every module.**

| Component | Status |
|---|---|
| Module 1 — Storage (canonical layout + ingestion) | ✅ Complete |
| Module 1 sibling — `FlatMP4Storage` (skip-ingest path for bare MP4 buckets) | ✅ Complete |
| Module 2 — Encoder (reasoning + visual arms) | ✅ Complete |
| Module 3 — Hypothesizer | ✅ Complete |
| Module 4 — Scorer (with `NIMTextClient` for production) | ✅ Complete |
| Module 5 — Judge UI (customer-facing rater) | ✅ Complete |
| Module 6 — Evaluation | ✅ Complete |
| Module 7 — Dev Dashboard (private operator eval surface) | ✅ Complete |
| `pipeline.run` CLI (`ingest` / `analyze` / `report`) | ✅ Complete |

Two valid input paths:
1. **Canonical** — Waymo Parquet or TFRecord → `ingest` → `analyze` → `report` (full windowing + pose).
2. **Flat MP4** — bucket of bare MP4 files → `analyze --storage-mode flat_mp4` (no ingest, one window per MP4).

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

**Recommended: self-hosted NIM on your GPU.** If you already have a GPU box (Brev, EC2, bare metal) — which you probably do, since you're running an AV pipeline — host the models yourself. Reasons:

- ~10× faster per batch (a 100-window `analyze` takes ~2 min locally vs ~1–2 hours on the hosted free tier with retries).
- No rate-limit babysitting (hosted free tier 429s at `--max-workers 2+`).
- Sidesteps any account-level access issues with the hosted API (e.g. specific model slugs returning 404).
- Marginal cost of running the containers ≈ 0 if the GPU is already paid for.

```bash
# Set NGC_API_KEY in .env (same NVIDIA account; separate token from NVIDIA_API_KEY), then:
docker compose --profile gpu up -d
```

This pulls and starts Cosmos-Embed1 (port 8080) and Cosmos-Reason2 (port 8081). One-time pull is ~15 min; after that, containers come up in seconds. Update `.env` to point both clients at local:

```bash
NVIDIA_BASE_URL=http://localhost:8081/v1
```

**GPU memory note.** Both the encoder (Cosmos-Reason2-7B, ~14 GB FP16) and the scorer's text model run as NIM containers. The scorer defaults to `meta/llama-3.1-70b-instruct` (~40 GB INT4). If you're on a single A100 40 GB, use the smaller scorer variant:

```bash
SCORER_NIM_MODEL_ID=meta/llama-3.1-8b-instruct
```

(~5 GB INT4 alongside the encoder ≈ 20 GB total — fits comfortably.) On A100 80 GB or H100, keep the 70b.

**Fallback: hosted NIM (zero setup, slower).** If you don't want to deal with docker, set only `NVIDIA_API_KEY` and leave `NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1`. Expect 1–2 hour batches with `--max-workers 1` to avoid rate limits. Suitable for a one-shot run; painful for iterating.

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

## Evaluating the Pipeline (Dev Dashboard)

Distinct from the customer-facing Judge UI (which collects ratings from your team on candidate scenarios), the **Dev Dashboard** is the private operator surface for evaluating Verity *itself*. Two evals, two tabs:

| Eval | Question it answers |
|---|---|
| **VLM Accuracy** | "How accurately does Cosmos-Reason2 describe scenes against my hand-labeled gold set?" — per-field match counts, precision/recall/F1 for multi-value fields. |
| **Discrimination Test** | "Does Verity's compositional discovery actually beat random sampling?" — 90 blind-shuffled windows from three pools (Verity / Random / Naive-rare), you rate each on safety relevance + rarity, export reveals source labels for offline Mann-Whitney. |

The discrimination test is the **headline eval** — it directly tests Verity's central claim against a baseline. Use this for CS231N-style writeups, internal reviews, or any "does this actually work?" investigation.

```bash
# Backend (refuses to start without VERITY_DEV_MODE=1)
VERITY_DEV_MODE=1 \
DEV_DASHBOARD_BUCKET_URI=gs://your-bucket/verity \
uvicorn pipeline.modules.dev_dashboard.server:app --port 8002

# Frontend (Dev tabs only render when this env is set at build time)
NEXT_PUBLIC_DEV_DASHBOARD_URL=http://localhost:8002 cd frontend && pnpm dev
```

Both gates (`VERITY_DEV_MODE` server-side, `NEXT_PUBLIC_DEV_DASHBOARD_URL` build-side) keep the dev surface off customer deployments. Full contract: [`pipeline/README.md`](pipeline/README.md) → Module 7.

**Pool sizing.** The discrimination test needs **30 accepted proposals** for the Verity pool. With ~100 windows worth of analyze output you should have plenty; with <30 windows the round-creation request returns 400. For the VLM accuracy eval, n=50–100 labeled windows is the CS231N-defensible minimum (n=10 is too few).

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

**Lots of `vlm_unavailable` failures on a fresh hosted NIM key (HTTP 429)**
The hosted free tier rate-limits aggressively. Two fixes, in order of preference:
1. **Switch to self-hosted NIM** — see Setup §3. Eliminates the rate limit entirely.
2. **Drop `--max-workers 1`** if you must stay on hosted. Each failed window is logged to stderr; the batch continues, no silent data loss.

**Annotation is slow**
Same answer: self-host. Cosmos-Reason2 on a local A100 runs each 8-second window in ~3–8 s vs ~10–30 s through the hosted API, with no rate limits to throttle batches.

**`Ctrl-C` during `analyze` takes a while to actually exit**
NIM clients have a 10-min default timeout (`NVIDIA_NIM_TIMEOUT_SECONDS=600`). The `ThreadPoolExecutor` waits for in-flight calls to either return or hit that timeout before shutting down. Lower the timeout if you want faster interrupts, or `kill -9` the process to bail immediately.

**The review UI shows no proposals**
The analysis step must complete before starting the Judge server. Check that `outputs/session-1/proposals.json` exists and is non-empty.

**No gaps found after analysis**
Verity needs enough data volume to detect statistical underrepresentation. If your dataset is small (< 50 windows), try running against a larger slice of your fleet data or contact us to adjust the detection thresholds.

---

## Configuration Reference

A full list of settings is in [`.env.example`](.env.example). The ones you're most likely to change:

| Setting | What it controls |
|---|---|
| `NVIDIA_API_KEY` | Hosted NIM authentication. Required for both hosted and (some) self-hosted use. |
| `NGC_API_KEY` | Container-pull authentication for self-hosted NIM. Same NVIDIA account, separate token. |
| `NVIDIA_BASE_URL` | `https://integrate.api.nvidia.com/v1` for hosted, `http://localhost:8081/v1` for self-hosted. |
| `SCORER_NIM_MODEL_ID` | Scorer's text model. Default `meta/llama-3.1-70b-instruct`. Drop to `meta/llama-3.1-8b-instruct` if GPU-memory tight. |
| `NVIDIA_NIM_TIMEOUT_SECONDS` | Per-call timeout, default 600 (10 min). Prevents a stuck NIM call from hanging a worker indefinitely. |
| `VERITY_DEV_MODE` | Must be `1` for the Dev Dashboard server to start. |
| `NEXT_PUBLIC_DEV_DASHBOARD_URL` | Set at frontend build time to expose the Dev tabs. Leave unset for customer builds. |

---

## Technical Reference

For output schemas, failure modes, module contracts, and pipeline internals, see [`pipeline/README.md`](pipeline/README.md).
