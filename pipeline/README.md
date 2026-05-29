# Verity Pipeline — Architecture Reference

> **Getting started?** See the [root README](../README.md) for installation, configuration, and quick-start instructions.

Six-module lego-block pipeline for AV safety scenario discovery.
Each module is a standalone Python package that imports shared types from
`pipeline/interfaces/` and nothing from other modules' internals.

---

## Pipeline overview

```
Fleet footage (Waymo Parquet / TFRecord)
         │
         ▼
┌──────────────────┐
│  Module 1        │  WindowStorage client + IngestionPipeline
│  Storage         │  Reads source footage → windows frames into 8-s clips
│                  │  Uploads canonical bucket structure (GCS)
│  Output:         │  WindowManifest, PoseData, WindowKey, signed video URLs
└────────┬─────────┘
         │  WindowStorage (read-only client)
         ▼
┌──────────────────┐
│  Module 2        │  Encoder (reasoning arm only — Phase 1)
│  Encoder         │  Calls Cosmos-Reason2 via NVIDIA NIM
│                  │  Extracts structured JSON; validates against vocabulary
│  Output:         │  SchemaRecord (one per window, success or failure_mode set)
└────────┬─────────┘
         │  list[SchemaRecord]
         ▼
┌──────────────────┐
│  Module 3        │  Hypothesizer
│  ✅ complete     │  Finds compositionally novel scenario combinations
│  Output:         │  list[CompositionProposal]
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Module 4        │  Scorer
│  ✅ complete     │  Rates plausibility + frontier difficulty; filters proposals
│  Output:         │  list[ScoredProposal]
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Module 5        │  Judge UI
│                  │  Human raters evaluate proposals; records coherence/usefulness
│  Output:         │  list[Rating]
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Module 6        │  Evaluation
│  ✅ complete     │  Seeded recall, rating stats, inter-rater agreement
│  Output:         │  EvaluationReport
└──────────────────┘
```

---

## Shared interfaces (`pipeline/interfaces/`)

All cross-module types are frozen dataclasses with `to_json()` / `from_json()`.

| File | Types |
|---|---|
| `window.py` | `WindowKey`, `PoseRecord`, `PoseData`, `WindowManifest`, `DatasetManifest` |
| `schema_record.py` | `SchemaRecord` |
| `proposal.py` | `CompositionProposal`, `ScoredProposal` |
| `rating.py` | `Rating` |
| `report.py` | `EvaluationReport`, `DifferentialExample` |

Round-trip tests: `pipeline/interfaces/tests/test_roundtrip.py` (24 tests).

---

## Module 1: Storage — Output Contract

### `WindowManifest` (written to GCS at `windows/{segment_id}/{window_idx:04d}/manifest.json`)

| Field | Type | Notes |
|---|---|---|
| `segment_id` | `str` | Source segment identifier |
| `window_idx` | `int` | 0-indexed window position within segment |
| `source_format` | `str` | `"waymo_parquet"` or `"waymo_tfrecord"` |
| `source_schema_version` | `str` | Version stamp of the source adapter |
| `window_start_ts_us` | `int` | First FRONT frame timestamp (microseconds) |
| `window_end_ts_us` | `int` | Last FRONT frame timestamp (microseconds) |
| `frame_count` | `int` | Number of FRONT frames in this window |
| `cameras` | `list[str]` | Camera names present (e.g. `["FRONT", "FRONT_LEFT", …]`) |
| `ingested_at` | `str` | ISO-8601 UTC timestamp |
| `pose_summary` | `str \| None` | Natural-language pose summary for VLM prompts; None if no pose data |
| `extra` | `dict` | Adapter-specific metadata (default `{}`) |

`window_key` property returns `WindowKey(segment_id, window_idx)`.

### `PoseData` (returned by `WindowStorage.get_window_pose()`)

| Field | Type |
|---|---|
| `segment_id` | `str` |
| `window_idx` | `int` |
| `records` | `list[PoseRecord]` |

Returns `records=[]` if `pose.parquet` was not written (best-effort at ingest time).

### `WindowKey` (returned by `WindowStorage.list_windows()`)

Frozen `(segment_id: str, window_idx: int)`. Hashable, usable as dict key.
String form: `f"{segment_id}/{window_idx:04d}"`.

### Bucket layout

```
{bucket}/windows/{segment_id}/{window_idx:04d}/
    manifest.json
    camera_FRONT.mp4
    camera_FRONT_LEFT.mp4
    camera_FRONT_RIGHT.mp4
    camera_SIDE_LEFT.mp4
    camera_SIDE_RIGHT.mp4
    pose.parquet          (best-effort — may be absent)
    pose_summary.json
{bucket}/segments/{segment_id}/index.json
{bucket}/index/manifest.json
{bucket}/errors/ingestion/{YYYY-MM-DD}/{segment_id}.json
```

### Error contract

All errors inherit from `StorageError` and print to stderr immediately on construction.

| Error | Trigger | Caller behavior |
|---|---|---|
| `SourceUnreachableError` | GCS/source cannot be reached | Abort — fix connectivity |
| `SourceSchemaVersionError` | Adapter schema version mismatch | Abort — fix adapter or re-export |
| `SourceAdapterError` | Per-segment fetch failure | Segment skipped, logged, continue |
| `WindowStorageError` | Retrieval request cannot be fulfilled | Caller must handle |
| `IngestionError` | Fatal ingestion setup failure | Abort |

---

## Module 2: Encoder — Output Contract

### `SchemaRecord` (one per window; written to local cache)

| Field | Type | Notes |
|---|---|---|
| `window_id` | `WindowKey` | Identifies the annotated window |
| `arm` | `str` | `"reasoning"` (Phase 1 only) |
| `schema_version` | `str` | `"1.0"` |
| `prompt_template_id` | `str \| None` | `"v1_describe"` |
| `fields` | `dict` | All 6 vocabulary keys present on success; may be null-filled on failure |
| `failure_mode` | `str \| None` | `None` = success; else one of the `FAILURE_*` constants |
| `cached` | `bool` | `True` if result came from cache |
| `created_at` | `str` | ISO-8601 UTC |

`succeeded` property: `failure_mode is None`.

### `fields` shape (schema v1.0)

```json
{
  "agents": ["car", "pedestrian"],
  "environment": {
    "weather": "clear",
    "time_of_day": "day",
    "lighting_condition": "well_lit"
  },
  "road": {
    "geometry": "straight",
    "lane_count": 2
  },
  "traffic_control": "none",
  "ego_task": "cruising",
  "conditions": []
}
```

All values drawn from the locked v1.0 vocabulary (`pipeline/modules/encoder/vocabulary.py`).

### Failure modes

| `failure_mode` | Cause |
|---|---|
| `"invalid_json"` | VLM response had no parseable JSON after `max_retries` |
| `"vocabulary_violation"` | JSON parsed but values violated locked vocabulary after retries |
| `"vlm_unavailable"` | VLM endpoint unreachable (network, auth) |
| `"storage_error"` | `WindowStorage` could not provide video URL or manifest |
| `"unknown"` | Unexpected exception |

`vlm_unavailable` results are **never cached** — they retry after recovery.

### Cache

- Location: `{cache_root}/encoder/reasoning/{sha256}.json`
- Key: `sha256(segment_id|window_idx|arm|schema_version|prompt_template_id|model_id)`
- Write: atomic (`{key}.json.tmp` → `{key}.json`)

---

## Module 3: Hypothesizer — Output Contract

### `CompositionProposal` (one per discovered composition; returned by `Hypothesizer.propose()`)

| Field | Type | Notes |
|---|---|---|
| `composition_id` | `str` | 16-char hex sha256 of sorted constituents |
| `constituents` | `list[str]` | Qualified atoms: `"prefix:value"` (≥2 per composition) |
| `marginal_frequencies` | `dict[str, float]` | Per-atom fraction; keys = `constituents` |
| `pairwise_frequencies` | `dict[str, float]` | Per-pair fraction; key = `"atom_a\|atom_b"` (sorted) |
| `expected_joint` | `float` | Product of marginal frequencies (independence assumption) |
| `observed_joint` | `float` | Fraction of windows containing all constituents |
| `novelty_score` | `float` | `ln(expected / max(observed, ε))` where `ε = 1/(10*N)` |
| `motivating_scene_ids` | `list[WindowKey]` | Windows where all atoms co-occur |
| `arm` | `str` | `"reasoning"` (Phase 1) |

`succeeded` property (inherited from proposal contract): `novelty_score > 0` indicates expected >> observed.

### Qualified atom format

Atoms are `"prefix:value"` strings. The prefix maps to a `SchemaRecord.fields` path:

| Prefix | Schema path | Type |
|---|---|---|
| `agents` | `fields["agents"]` | list (multi-value) |
| `weather` | `fields["environment"]["weather"]` | scalar |
| `time_of_day` | `fields["environment"]["time_of_day"]` | scalar |
| `lighting` | `fields["environment"]["lighting_condition"]` | scalar |
| `road_geometry` | `fields["road"]["geometry"]` | scalar |
| `traffic_control` | `fields["traffic_control"]` | scalar |
| `ego_task` | `fields["ego_task"]` | scalar |
| `conditions` | `fields["conditions"]` | list (multi-value) |

`road.lane_count` is excluded (numeric, not categorical).

### Mutual exclusivity

Two atoms with the same prefix from a scalar field (weather, time_of_day, lighting, road_geometry, traffic_control, ego_task) cannot appear in the same composition — a window cannot have two weather values. Multi-value fields (agents, conditions) allow same-prefix pairs.

### Filtering pipeline

1. **Marginal frequency** ≥ `min_marginal_frequency` (default 0.05)
2. **Min pairwise co-occurrence** ≥ `min_pairwise_frequency` (default 0.01)
3. **Observed joint** < `max_joint_frequency` (default 0.005)
4. **Mutual exclusivity** check (scalar-field same-prefix pairs rejected)
5. **Ranked** by `novelty_score` DESC, `composition_id` ASC (tie-breaker)
6. **Truncated** to `top_k` (default 30)

### Public interface

```python
class Hypothesizer:
    def __init__(self, config: HypothesizerConfig = HypothesizerConfig()) -> None: ...
    def propose(self, records: list[SchemaRecord], arm: str = "reasoning") -> list[CompositionProposal]: ...
```

### Side effects

- None. `propose()` is a pure function (stateless, no I/O).

### Failure modes

| Failure | Behavior |
|---|---|
| Empty record list | Raise `HypothesizerEmptyInputError` |
| All records have `failure_mode` set | Raise `HypothesizerEmptyInputError`; count logged to stderr |
| Record has succeeded=True but all fields null | Skipped; count logged to stderr separately |
| Atom value not in `valid_atoms` | Raise `VocabularyMismatchError` immediately (strict) |
| No compositions pass filters | Return empty list |

---

## Module 4: Scorer — Output Contract

Produces `ScoredProposal` objects from `pipeline/interfaces/proposal.py`.

### Input

`CompositionProposal` list from Module 3, plus two injectable `TextClient`
adapters (plausibility VLM, difficulty proxy reasoner).

### Output fields

| Field | Type | Notes |
|---|---|---|
| `composition_id` | `str` | Preserved from input |
| `constituents` | `list[str]` | Preserved from input |
| `marginal_frequencies` | `dict[str, float]` | Preserved from input |
| `pairwise_frequencies` | `dict[str, float]` | Preserved from input |
| `expected_joint` | `float` | Preserved from input |
| `observed_joint` | `float` | Preserved from input |
| `novelty_score` | `float` | Preserved from input |
| `motivating_scene_ids` | `list[WindowKey]` | Preserved from input |
| `arm` | `str` | Preserved from input |
| `plausibility_score` | `float` | 0.0–1.0; 0.0 on failure |
| `plausibility_justification` | `str` | VLM text; empty string if `rejection_reason == "plausibility_check_failed"` |
| `frontier_difficulty_score` | `float \| None` | `None` when difficulty_client=None or all 3 runs fail |
| `frontier_difficulty_signals` | `dict[str, float]` | `{}` when difficulty unavailable; otherwise `mean_confidence`, `action_variance`, `reasoning_action_mismatch` |
| `final_rank_score` | `float` | `novelty*0.4 + plausibility*0.3 + difficulty*0.3` (difficulty term is 0.0 when None) |
| `accepted` | `bool` | True iff plausibility succeeded AND score ≥ threshold |
| `rejection_reason` | `str \| None` | `None` if accepted; `"plausibility_check_failed"` or `"plausibility_below_threshold"` |

### Public interface

```python
class Scorer:
    def __init__(
        self,
        plausibility_client: TextClient,
        difficulty_client: TextClient | None = None,
        config: ScorerConfig = ScorerConfig(),
        cache_root: Path | None = None,
    ) -> None: ...

    def score(self, proposal: CompositionProposal) -> ScoredProposal: ...
    def score_batch(self, proposals: list[CompositionProposal]) -> list[ScoredProposal]: ...
```

### Cache

Key: `sha256(composition_id | p_model_id | d_model_id | p_prompt_v | d_prompt_v)`
Location: `{cache_root}/scorer/{key}.json`
Write: atomic (`.json.tmp → .json` via `os.replace()`)
Failures are NOT cached — they will be retried on next call.

### Difficulty signals

Three runs per proposal with different constituent orderings (deterministic
seed from `sha256(composition_id)[:8]`). Signals:

```
frontier_difficulty_score =
    action_variance * 0.5
    + (1 - mean_confidence) * 0.3
    + reasoning_action_mismatch * 0.2
```

If all 3 difficulty runs fail: `frontier_difficulty_score=None`, `frontier_difficulty_signals={}`.
Proposal is still accepted/rejected on plausibility alone.

### Failure modes

| Failure | Behavior |
|---|---|
| All 3 plausibility runs fail | `accepted=False`, `rejection_reason="plausibility_check_failed"`, not cached |
| Plausibility score below threshold | `accepted=False`, `rejection_reason="plausibility_below_threshold"`, cached |
| Partial plausibility failure (1-2/3 succeed) | Conservative aggregation: 2/3 → lower score; 1/3 → single result |
| All difficulty runs fail | `frontier_difficulty_score=None`, proposal still scored on plausibility |
| Cache write fails (disk full, permissions) | Logged to stderr, not fatal — next call re-invokes VLM |

---

## Module 5: Judge UI — Output Contract

### `Rating` (one per rater per proposal; written to `ratings/{rater_id}/{proposal_id}.json`)

| Field | Type | Notes |
|---|---|---|
| `rater_id` | `str` | Identifier provided by rater at session start |
| `proposal_id` | `str` | Matches `CompositionProposal.composition_id` |
| `arm` | `str` | `"reasoning"` or `"visual"` — injected server-side, never from rater |
| `coherence_score` | `int` | 1–5 |
| `usefulness_score` | `int` | 1–5 |
| `timestamp` | `str` | ISO-8601 UTC |
| `free_text_note` | `str \| None` | Optional rater observation |
| `seen_motivating_scenes` | `list[WindowKey]` | Which motivating videos the rater watched |

### Server endpoints (port 8001)

| Endpoint | Description |
|---|---|
| `GET /judge/proposals` | Ranked list of accepted proposals, arm blinded |
| `GET /judge/proposals/{id}` | Full proposal detail, arm blinded |
| `GET /judge/video-url` | Pre-signed GCS URL for a motivating-scene video |
| `POST /judge/ratings` | Submit a rating; arm injected server-side |
| `GET /judge/session/{rater_id}` | Rated proposal IDs + score distributions (session resumability) |
| `GET /judge/ratings/export` | All ratings across all raters — consumed by Module 6 |

### Blinding contract

The `arm` field is **never sent to the rater** in any response from this server.
It is read from the proposal store at submission time and written only to the
persisted rating file. Module 6 reads `arm` from the export; raters never see it.
A runtime assertion in `_blind_row()` and `_blind_detail()` enforces this.

### Storage layout

```
pipeline/modules/judge_ui/
  proposals.json          # input: list[ScoredProposal] from Module 4
  ratings/
    {rater_id}/
      {proposal_id}.json  # output: one Rating per submission (atomic write)
```

### Run

```bash
JUDGE_BUCKET_URI=gs://my-bucket/verity \
JUDGE_PROPOSALS_PATH=pipeline/modules/judge_ui/proposals.json \
uvicorn pipeline.modules.judge_ui.server:app --port 8001 --reload
```

### Known assumptions

- Proposals are loaded once at server startup. Restart to pick up new proposals from Module 4.
- `composition_id` values must be flat hex strings (no `/` characters); enforced at write time.

---

## Module 6: Evaluation — Output Contract

Produces `EvaluationReport` and `DifferentialExample` from `pipeline/interfaces/report.py`.

### `EvaluationReport` (returned by `Evaluator.evaluate()`)

| Field | Type | Notes |
|---|---|---|
| `seeded_recall` | `dict[str, dict[str, dict[str, float]]]` | `{arm: {subset: {k: recall}}}` — subset ∈ {overall, familiar, unfamiliar}; k ∈ {@10, @30, @all} |
| `recall_k_primary` | `int` | = 30, pre-registered before any results are seen |
| `mean_coherence` | `dict[str, float]` | arm → mean coherence (1–5); `float("nan")` in-memory when arm has no ratings; serializes to `null` in report.json |
| `mean_usefulness` | `dict[str, float]` | arm → mean usefulness (1–5); same NaN/null behavior |
| `coherence_ci_95` | `dict[str, tuple \| None]` | 95% bootstrap CI; None when n\_ratings < 30 |
| `usefulness_ci_95` | `dict[str, tuple \| None]` | Same |
| `n_ratings_per_arm` | `dict[str, int]` | Always present alongside every CI |
| `inter_rater_agreement_coherence` | `float \| None` | Krippendorff's α (ordinal); None when < 2 overlapping raters |
| `inter_rater_agreement_usefulness` | `float \| None` | Same |
| `n_raters_overlapping` | `int` | Raters with ≥1 rating on the same proposal |
| `differential_examples` | `list[DifferentialExample]` | Top compositions where arms diverged by rank; empty for single-arm runs |
| `failure_mode_distribution` | `dict[str, Any]` | Encoder failure counts (requires `schema_records` in input) |
| `n_proposals_per_arm` | `dict[str, int]` | Accepted proposals per arm |
| `n_proposals_filtered` | `dict[str, int]` | Rejected proposals per arm |
| `n_raters` | `int` | Distinct rater IDs |
| `seeded_set_size` | `dict[str, int]` | `{"familiar": N, "unfamiliar": N}` |

### Public interface

```python
from pipeline.modules.evaluation import Evaluator, EvaluationInput

class EvaluationInput:
    proposals_by_arm: dict[str, list[ScoredProposal]]
    ratings: list[Rating]
    seeded_window_ids: list[WindowKey]         # non-empty; pre-registered
    seeded_subset_labels: dict[WindowKey, Literal["familiar", "unfamiliar"]]
    schema_records: list[SchemaRecord] | None  # optional
    recall_k: int                               # = 30, pre-register before running

class Evaluator:
    def evaluate(self, input: EvaluationInput) -> EvaluationReport: ...  # pure, no I/O
    def save(self, report: EvaluationReport, output_dir: Path) -> Path: ...
```

### Failure modes

| Failure | Behavior |
|---|---|
| `seeded_window_ids` empty | Raise `MissingSubsetLabelsError` |
| `seeded_subset_labels` missing keys | Raise `MissingSubsetLabelsError` |
| Rating references unknown arm | Raise `ArmMismatchError` |
| Fewer than 2 overlapping raters | `inter_rater_agreement_*=None`, flagged; other metrics still computed |
| `n_ratings < 30` for an arm | CI fields are `None`; mean is still reported |

---

## Running tests

```bash
# All pipeline tests
python -m pytest pipeline/ -v

# By layer
python -m pytest pipeline/interfaces/tests/           # round-trip contracts (21)
python -m pytest pipeline/modules/storage/tests/      # Module 1 smoke + contract (35)
python -m pytest pipeline/modules/encoder/tests/      # Module 2 smoke + contract (56)
python -m pytest pipeline/tests/integration/          # cross-module boundary (4+)
```

---

## Lego-block rule

**No module may import from another module's internals.**

```
# FORBIDDEN
from pipeline.modules.storage.adapters.base import WindowKey  # in encoder

# REQUIRED
from pipeline.interfaces.window import WindowKey               # in encoder
```

The `pipeline/interfaces/` package is the only shared surface area between modules.
