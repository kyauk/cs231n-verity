# First End-to-End Run — Findings & README Gaps

**Date:** 2026-06-02
**Goal:** Run the Verity pipeline end-to-end from Waymo Parquet → ingest → analyze → Judge UI, self-hosted NIM, following the root + `pipeline/` READMEs as a first-time operator.
**Outcome:** Reached a live, working pipeline (16 segments → 48 windows → 43 annotated → 30 proposals → **27 accepted**, served in the Judge UI). Getting there required **6 fixes** the docs didn't anticipate. One was a host driver issue (documented separately); the other five — plus several documentation/process gaps — are analyzed below so the README can be corrected.

This document deliberately **excludes** the GPU-driver/CUDA wall (that's a host-provisioning issue). Everything here is a code, config, prompt, or documentation gap in the project itself.

---

## Environment for context

| Item | Value |
|---|---|
| GPU | NVIDIA L40S, 46 GB |
| Encoder model actually run | `nvidia/cosmos-reason1-7b` (self-hosted NIM) — see Issue 0 |
| Source data | `gs://waymo_open_dataset_v_2_0_1/validation/camera_image` (Waymo Open Dataset v2 Parquet) |
| Output bucket | `gs://nvidia-adr-waymo-segment-videos/verity` |
| Python | 3.10.12 |
| Credentials | User ADC (read on Waymo source + write on output) + a `verity-signer` service-account **key** (signing only) |

---

## Issue 0 — Encoder model: docs name a model that isn't reachable as written

**Severity:** High (blocks the documented happy path)
**Type:** Model availability / documentation

**What the docs say:** the encoder uses **Cosmos-Reason2** (`pipeline/README.md` Module 2; `.env.example` defaults `NVIDIA_MODEL_ID`, code default `COSMOS_REASON2_MODEL_ID=nvidia/cosmos-reason2-7b`). Self-host path: `docker compose --profile gpu up` pulls `nvcr.io/nim/nvidia/cosmos-reason2-7b:latest`.

**What actually happens for a fresh NVIDIA/NGC account:**
- **Hosted API** (`integrate.api.nvidia.com`): `nvidia/cosmos-reason2-8b` *appears in* `/v1/models` but returns **HTTP 404 "Function not found for account"** on inference (even text-only). `nvidia/vila` too. So the hosted Reason 2 is not actually callable.
- **Self-host container** `cosmos-reason2-7b`: NGC returns **UNAUTHORIZED** on the repo (gated, not granted to a standard key). `cosmos-reason2-8b` *container* is pullable, but its `1.7.0` image needs a **CUDA 12.8** driver (the driver issue, out of scope here).
- **What works today:** `cosmos-reason1-7b` (NGC repo is accessible; older NIM builds against CUDA ≤12.6) and `meta/llama-3.1-70b-instruct` (hosted, for the scorer).

**Recommendation for README:**
1. State the **exact entitlement** Reason 2 requires (NGC repo access / hosted function provisioning) and link to the access-request flow. A standard free-tier key does **not** have it.
2. Document a **verification step** before any run: `curl $NVIDIA_BASE_URL/v1/models` for hosted, and `curl https://nvcr.io/v2/nim/nvidia/<repo>/tags/list` (with token) to confirm the container is pullable.
3. List **fallback models** explicitly (`cosmos-reason1-7b`) and note that all model IDs are pure env config (`COSMOS_REASON2_MODEL_ID`, `NVIDIA_BASE_URL`, `SCORER_NIM_MODEL_ID`) — no code change to swap.
4. Pin a **real container tag** in `docker-compose.yml` (it ships `cosmos-reason2-7b:latest`, and `:latest` 404s — the tag doesn't resolve).

---

## Issue 1 — ffmpeg MP4 encode crashes on every window (ingest produces 0 videos)

**Severity:** Critical (ingest writes no usable windows)
**Type:** Code bug
**File:** `pipeline/modules/storage/ingestion.py` (`_encode_mp4`, ~line 322)

**Symptom:** every camera of every window logs `ffmpeg FAILED ... flush of closed file`; `segments_ok=0`.

**Root cause:** the encoder does
```python
proc.stdin.close()
_, stderr_bytes = proc.communicate()   # <-- communicate() re-flushes the already-closed stdin
```
On Python 3.10+, `communicate()` unconditionally flushes `self.stdin`; since it was just closed, it raises `ValueError: flush of closed file`. ffmpeg/libx264 are fine — this is purely the close-then-communicate antipattern.

**Fix applied:**
```python
proc.stdin.close()
stderr_bytes = proc.stderr.read() if proc.stderr is not None else b""
proc.wait()
```

**Recommendation:** fix in code; add a one-segment ingest smoke test to CI (the existing tests apparently mock ffmpeg, so this never surfaced).

---

## Issue 2 — `analyze` builds the embedding client even with `--no-visual`, and with a wrong kwarg

**Severity:** Critical (any non-stub `analyze` crashes at startup)
**Type:** Code bug
**File:** `pipeline/run.py` (`_build_encoder`, ~line 277)

**Symptom:** `TypeError: CosmosEmbed1Client.__init__() got an unexpected keyword argument 'url'` — even when `--no-visual` is passed.

**Root cause (two bugs):**
1. `embed_client = CosmosEmbed1Client(url=...)` — the constructor parameter is **`cosmos_url`**, not `url` (`pipeline/modules/encoder/visual_arm.py:117`).
2. The client is constructed **unconditionally**, *before* the `if no_visual:` check — so `--no-visual` (reasoning-only) can't avoid it, and it requires the Cosmos-Embed1 endpoint even when the visual arm is off.

**Fix applied:** make the embed client **lazy** — only construct it when the visual arm is actually used, and use the correct `cosmos_url=` kwarg.

**Recommendation:** fix in code. A `--no-visual --stub`-free smoke test of `analyze` wiring would have caught both.

---

## Issue 3 — NIM rejects Waymo video: pixel budget exceeded (47/48 windows → HTTP 400)

**Severity:** Critical (annotation fails for full-res footage)
**Type:** Config / documentation
**Where:** NIM container env, not in repo

**Symptom:** `Video pixel budget 78643200 px exceeds server limit VLLM_MAX_TOTAL_VIDEO_PIXELS=25690112 (frames=32, resolution=1920x1280)` → HTTP 400 on 47 of 48 windows; classified as `vlm_unavailable`. (The one success was a shorter window.)

**Root cause:** Waymo FRONT camera is **1920×1280**. At the NIM's 4 fps sampling, an 8 s window = 32 frames = **78.6 M px**, ~3× the NIM's default `VLLM_MAX_TOTAL_VIDEO_PIXELS` (25.7 M).

**Fix applied:** set on the container:
```yaml
VLLM_MAX_TOTAL_VIDEO_PIXELS: "104857600"   # ~100M px, covers 1920x1280 x 32 frames + headroom
```
(also reduced `analyze --max-workers` to 2 to keep concurrent video decodes within VRAM.)

**Recommendation for README/compose:**
1. Set `VLLM_MAX_TOTAL_VIDEO_PIXELS` in `docker-compose.yml` for the encoder service (Waymo is a documented input — its resolution is known).
2. Document the relationship: `pixels = frames × W × H`, `frames ≈ window_seconds × sampled_fps`. Note the alternative levers (downscale at ingest, lower sampled fps).
3. Surface the NIM 400 body in the encoder's `vlm_unavailable` log — right now a clear "reduce pixels" message from the NIM is hidden behind a generic unavailable classification.

---

## Issue 4 — Vocabulary validator crashes (not retries) when the model returns a dict for a scalar field

**Severity:** High (windows fail as "unknown" and skip the retry path)
**Type:** Code bug / robustness
**File:** `pipeline/modules/encoder/vocabulary.py` (`validate_fields`)

**Symptom:** `[Encoder] UNEXPECTED ERROR ... TypeError: unhashable type: 'dict'` on some windows; classified as `unknown` (not retried).

**Root cause:** scalar checks do `value not in self.<frozenset>`. When the VLM occasionally emits a **dict** where a scalar tag is expected (e.g. `traffic_control: {...}`), `dict in frozenset` computes `hash(dict)` → `TypeError`. Because it's an unexpected exception (not a `VocabularyViolation`), it **escapes the retry loop** that would otherwise re-prompt and recover the window.

**Fix applied:** a `_not_in_vocab(value, vocab)` helper that treats any **non-string** scalar as a normal vocabulary violation (so it's caught, reported, and retried with a stricter prompt) instead of crashing. Applied to all 8 membership checks (agents, weather, time_of_day, lighting, road.geometry, traffic_control, ego_task, conditions).

**Recommendation:** fix in code; add a validator unit test with malformed (dict/list/int) field values. Models *will* return off-spec JSON — the validator must degrade to "violation," never raise.

---

## Issue 5 — Plausibility prompt leaks the rarity statistic → model rejects every rare scenario (0 accepted)

**Severity:** Critical (defeats the product's core purpose; UI shows nothing)
**Type:** Prompt-design bug
**File:** `pipeline/modules/scorer/plausibility.py` (`describe_composition`)

**Symptom:** all 30 proposals scored `plausibility_score = 0.0` → all below the 0.5 threshold → **0 accepted** → empty Judge UI. Justifications literally read *"implausible because the observed joint frequency ... is zero."*

**Root cause:** `describe_composition` appended a **"Statistical context"** block to the plausibility prompt:
```
Statistical context:
  Expected joint frequency (under independence): ...
  Observed joint frequency: 0.0000
  Novelty: these conditions are individually common but jointly rare.
```
The Hypothesizer's entire job is to surface **jointly-rare** combos (low/zero observed frequency). Feeding that statistic into the *plausibility* judge makes the model equate "rare in this dataset" with "physically impossible," so it rejects exactly the scenarios we want. Rarity is **already** captured separately as `novelty_score` (weighted 0.4 in `final_rank_score`); the prompt was double-counting it as implausibility.

**Fix applied:** `describe_composition` now lists only the conditions and instructs the model to judge **physical/behavioral co-occurrence**, explicitly: *"Do NOT consider how rare or common the combination is — statistical rarity is not the same as implausibility."* Result: plausibility mean **0.86**, **27/30 accepted**, with genuinely odd combos still filtered.

**Recommendation:** fix in code. This is the single most important correctness bug — it silently turns a working pipeline into one that accepts nothing, with no error. Add a regression test: a plausible-but-rare composition (e.g. `weather:clear + lighting:dim`) must score > threshold.

---

## Documentation / process gaps (no code change, but README should cover)

### A. `analyze` does not load `.env`
`python -m pipeline.run analyze` reads `NVIDIA_API_KEY` from the **environment** and aborts (`NVIDIA_API_KEY is not set`) if you only have it in `.env`. The README implies `.env` is consumed automatically. **Fix doc** (or load dotenv in `run.py`): the run commands must be preceded by `set -a; source .env; set +a` (and `GOOGLE_APPLICATION_CREDENTIALS` exported separately — it's not in `.env`).

### B. Encoder and scorer share `NVIDIA_BASE_URL`
`run.py` builds both `CosmosReason2Client` and `NIMTextClient` from the same `NVIDIA_BASE_URL` with no per-client override. So you **cannot** run the encoder self-hosted while the scorer uses the hosted API. Self-host means the scorer's text model must also be served by the local container — set `SCORER_NIM_MODEL_ID` to the local model (we used `nvidia/cosmos-reason1-7b` for both). Document this, or add `SCORER_NIM_BASE_URL`.

### C. GCS signing needs a service-account **key**, and a credential split
The README covers that user-ADC can't sign v4 URLs. What it omits: if you set `GOOGLE_APPLICATION_CREDENTIALS` to the signer SA **globally**, that SA usually **can't read the Waymo public source bucket** (403 — Waymo access is granted per-account). Working pattern: run **`ingest` under user ADC** (reads Waymo + writes output), and **`analyze` with the SA key** (reads output + signs). The signer SA only needs `Storage Object Viewer` on the **output** bucket.

### D. Judge UI input file: README example points at the wrong JSON
Root README Step 3 sets `JUDGE_PROPOSALS_PATH=outputs/session-1/proposals.json`. The server loads **`ScoredProposal`** and filters `accepted` (`pipeline/modules/judge_ui/server.py:99`), so it must point at **`scored.json`** (Scorer output). `proposals.json` is the Hypothesizer's `CompositionProposal` list — no scores/`accepted`, so the UI shows nothing.

### E. Frontend: package manager + two-port forwarding
- README says `pnpm`; the repo also has `package-lock.json` and `pnpm` isn't always installed — `npm run dev` works against the existing `node_modules`.
- The Judge tab fetches `NEXT_PUBLIC_JUDGE_API_URL` (default `http://localhost:8001`) **from the browser**, so a remote/VS-Code operator must forward **two** ports (3000 + 8001). We added a Next.js rewrite proxying `/judge/*` → `:8001` and run with `NEXT_PUBLIC_JUDGE_API_URL=''`, so only **port 3000** needs forwarding. Recommend shipping that rewrite (or documenting both-port forwarding prominently).

### F. `report` needs a pre-registered `seeds.json` you must create
`report` requires `--seeds seeds.json` (`{"seeded_windows":[{"window":"<seg>/0000","subset":"familiar"|"unfamiliar"}]}`). There's no generator; we built one from the ingested window IDs. Document the schema and ideally ship a `make-seeds` helper.

### G. Small-N behavior: no gaps / no motivating clips
With 48 windows, one window = 2.1% frequency, but the Hypothesizer keeps only `observed_joint < 0.5%` — so **only combos that appear in zero windows survive**, and those have **empty `motivating_scene_ids`** (no clips to inspect in the UI). This is correct by construction but confusing. The README's "<50 windows" warning should be sharpened: below a few hundred windows, accepted gaps will have no example clips because they're (correctly) absent from the data. Tuning `max_joint_frequency` or ingesting more windows surfaces rare-but-present combos that *do* have clips.

---

## Summary table

| # | Issue | Type | Fixed in |
|---|---|---|---|
| 0 | Reason 2 not reachable (hosted 404 / NGC gated); compose `:latest` doesn't resolve | availability/docs | `.env`, `docker-compose.yml` (→ Reason 1) |
| 1 | ffmpeg `stdin.close()`+`communicate()` → encode crash | code | `storage/ingestion.py` |
| 2 | embed client built eagerly + wrong `url=` kwarg under `--no-visual` | code | `run.py` |
| 3 | NIM video pixel budget too low for 1920×1280 | config | `docker-compose.yml` |
| 4 | validator `TypeError` on dict scalar (escapes retry) | code | `encoder/vocabulary.py` |
| 5 | plausibility prompt leaks rarity → 0 accepted | prompt | `scorer/plausibility.py` |
| A–G | `.env` load, shared base URL, signing split, Judge input file, frontend ports, seeds, small-N | docs/process | this doc |

**Net:** the architecture is sound and every fix was small. The blockers were (a) one model-availability mismatch with the docs, (b) two latent code bugs on the real (non-stub, non-mocked) path, (c) one NIM config value for real-resolution video, and (d) one prompt-design bug that silently zeroed the output. Items 1, 2, 4, 5 are code fixes worth upstreaming; 0, 3, A–G are documentation/config corrections for the README.
