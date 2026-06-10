# Verity — Autonomous-Vehicle Safety Scenario Discovery

Verity mines real driving footage to surface the **edge cases your AV stack has not been validated against** — operationally difficult and under-represented scenes — and turns them into **structured, simulator-ready scenario specifications** your team can prioritize, generate, and test.

You bring driving data. Verity annotates it, discovers the rare and difficult scenes, ranks them by how hard they are for an automated driver, and presents them for human review with supporting evidence and a generation-ready description for each.

---

## What you get

After a run on your data, Verity delivers:

- **A ranked list of edge-case scenarios** — surfaced from your footage by *model-judged operational difficulty* and *rarity of behavior / conditions*, not by hand-written rules.
- **Grounded evidence** — for every surfaced scenario, the real scene(s) it was drawn from, viewable and playable in the review UI, with a pointer from each structured tag back to the observation that justified it.
- **Simulator-ready scenario descriptions** — a vivid, complete spec for each scenario (road layout, signals, weather, lighting, agents and their intent), suitable for driving-scene generation.
- **An evolving scene taxonomy** — an emergent label vocabulary that grows richer as you add data, with no fixed schema to maintain.
- **A blinded human-review UI** — raters score each scenario; ratings persist for calibration.

---

## How it works

```
  Driving data ─▶ Annotate ─▶ Build taxonomy ─▶ Select ─▶ Synthesize ─▶ Review
   (clips)        (VLM)        (emergent)        (rank)    (scenario)     (UI)
```

1. **Ingest** — driving data (Waymo Parquet / TFRecord, or a bucket of MP4s) becomes per-scene video clips.
2. **Annotate** — a vision-language model watches each clip and writes a free-form analysis, then a structuring pass extracts typed descriptors. Each descriptor carries (a) a **salience** score (how operationally critical it is) and (b) a **span pointer** back to the sentence that justified it. This is immutable evidence.
3. **Build the taxonomy** — descriptors are clustered into **emergent canonical labels** that persist and refine across runs. The vocabulary is not fixed; it grows and sharpens as more data arrives, and past data is re-projected onto the richer taxonomy automatically.
4. **Select** — scenes are ranked by **model-judged difficulty (primary)** and **behavior-novelty (refining)**, with an independent difficulty cross-check that flags possible over-reports.
5. **Synthesize** — each surfaced scene is turned into a **novel, generation-ready scenario** derived from its components (not a caption of one clip).
6. **Review** — the surfaced scenarios appear in the **Judge** tab for human rating (coherence + usefulness).

Verity is built as a set of independent, swappable modules (the [pipeline reference](pipeline/README.md) is the technical guide). Annotation, taxonomy, and selection are decoupled, so any stage — including the reasoning model — can be replaced without touching the others.

---

## What you need

| | |
|---|---|
| **Driving data** | Waymo Open Dataset (Parquet or TFRecord), or a GCS bucket of MP4 clips |
| **Cloud storage** | A Google Cloud Storage bucket you control |
| **Reasoning model** | An NVIDIA GPU to self-host the model (recommended), or an [NVIDIA API key](https://build.nvidia.com) |
| **Runtime** | Python 3.10+ (pipeline), pnpm 8+ (review UI) |

---

## Setup

### 1. Install

```bash
git clone <repo-url> && cd Verity
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd frontend && pnpm install && cd ..
```

### 2. Configure

```bash
cp .env.example .env
```

Fill in your NVIDIA key and (optionally) point the model at a self-hosted endpoint:

```bash
NVIDIA_API_KEY=nvapi-...
NVIDIA_BASE_URL=http://localhost:8081/v1   # local NIM; or the hosted endpoint
GCS_PROJECT=your-gcp-project
```

Authenticate to your bucket once:

```bash
gcloud auth application-default login
# or, on a VM: GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

### 3. Run the reasoning model

**Recommended — self-hosted NIM on your GPU** (≈10× faster, no rate limits):

```bash
# set NGC_API_KEY in .env (same NVIDIA account), then:
docker compose --profile gpu up -d        # Cosmos-Reason on :8081
```

A single L40S / A100 (40 GB+) runs the reasoning model comfortably. **Fallback:** leave `NVIDIA_BASE_URL` on the hosted endpoint and set only `NVIDIA_API_KEY` (zero setup, slower). Text embeddings for the taxonomy use a hosted endpoint by default and do not compete for the GPU.

---

## Run a discovery session

```bash
# 1. Ingest driving data into per-scene clips (canonical path):
python -m pipeline.run ingest \
  --source-format waymo_parquet \
  --source-root gs://waymo-bucket/validation/camera_image \
  --bucket gs://your-bucket/verity --segments all

# 2. Review server + UI:
uvicorn pipeline.modules.judge_ui.server:app --port 8001
cd frontend && pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) → the **Judge** tab. For each surfaced scenario your reviewers can play the source scene, read the generation-ready description, and score it. Multiple reviewers rate independently; ratings merge automatically.

> The emergent-taxonomy + salience discovery flow (annotation → taxonomy → selection → synthesis) is driven by the composition-root scripts in [`drivers/`](drivers/), run as `python -m drivers.<name>` (e.g. `python -m drivers.verity_hardnovel`); they wire the modules together and write the feed the Judge UI reads. See the [pipeline reference](pipeline/README.md) for how they compose.

---

## Repository layout

| Path | What |
|---|---|
| `pipeline/interfaces/` | Shared, frozen data contracts (the only thing modules share) |
| `pipeline/modules/` | The lego-block modules — storage, extractor, curator, selection, hypothesizer, scorer, judge UI, evaluation |
| `pipeline/run.py` | CLI composition root for the canonical ingest / analyze / report path |
| `frontend/` | The Next.js review UI (single origin; proxies to the judge server) |
| `drivers/` | Composition-root scripts that wire modules into the discovery flow — run as `python -m drivers.<name>` |

For the full architecture, module contracts, and data flow, see **[`pipeline/README.md`](pipeline/README.md)**.
