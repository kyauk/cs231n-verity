# Verity — CHANGELOG

## 2026-05-29 — Bug-fix batch: NIM timeouts, GCS thread safety, MP4 size validation

Three bugs that had been documented as "accepted risks" in earlier hygiene
cycles, now fixed because each turned out to be a real reliability issue
rather than the cosmetic concern I originally classified them as.

**Fixed:**
- **NIM client timeouts** (`CosmosReason2Client`, `NIMTextClient`) — both
  now accept a `timeout` constructor arg (default 600 s = 10 min) and read
  the shared `NVIDIA_NIM_TIMEOUT_SECONDS` env var. The timeout is passed to
  the underlying `OpenAI(...)` client. Before this fix a stuck NIM call had
  no upper bound and would block a `ThreadPoolExecutor` worker (and the
  whole `pipeline.run analyze` process, including Ctrl-C) indefinitely.
- **Thread-safe lazy GCS init** in `WindowStorage._get_bucket()` and
  `FlatMP4Storage._get_bucket()` via double-checked locking on a
  per-instance `threading.Lock`. Hot path is still lock-free after first
  init. Concurrent first-callers no longer race to construct the GCS
  client twice.
- **MP4 size validation** in `FlatMP4Storage.get_window_video_url()`.
  Switched from `blob.exists()` to `blob.reload()` (one HTTP HEAD-equivalent
  that returns both existence AND `blob.size`) and added a `< 1024-byte`
  floor (`_MIN_MP4_BYTES`). Truncated uploads, zero-byte placeholders, and
  mis-renamed text files are now caught at retrieval time with a clear
  error instead of producing an opaque VLM-fetch failure.

**Tests added (16 new):**
- `pipeline/modules/scorer/tests/test_nim_client.py` — 4 timeout tests
  (default 600 s; constructor override; `NVIDIA_NIM_TIMEOUT_SECONDS` env
  override; timeout reaches the OpenAI client).
- `pipeline/modules/encoder/tests/test_smoke.py` — 4 parallel timeout
  tests for `CosmosReason2Client`.
- `pipeline/modules/storage/tests/test_thread_safety.py` — 4 tests
  (16 concurrent first-callers → exactly 1 client construction; pre-warmed
  cache → 0 constructions; same for both `WindowStorage` and
  `FlatMP4Storage`).
- `pipeline/modules/storage/tests/test_flat_mp4.py` — 4 size-validation
  tests (zero bytes / 512 bytes / `size=None` rejected; healthy 100 KB
  blob accepted).

**Test helper updates:**
- `_fake_blob` in `test_flat_mp4.py` now sets `blob.reload`, `blob.size`,
  and `blob.content_type` so existing tests work under the new code path.
- Contract and integration fixtures (`test_flat_mp4_output_contract.py`,
  `test_flat_mp4_to_encoder.py`) updated likewise.
- `_mock_storage` in `test_storage_output_contract.py` now sets
  `_bucket_lock` since the test bypasses `__init__` via `__new__`.

**Docs:**
- `flat_mp4.py` module docstring — replaced the now-stale "no validation"
  and "not thread-safe" accepted-risk entries with the new "Validation +
  safety" section. The greedy filename-parser caveat remains (genuine,
  unchanged).

**Re-hygiene scope:** lightweight, per the protocol's "Bug fixes that
don't change contracts" rule. Smoke + contract + reconciliation only;
no full new hygiene cycle. All existing contract tests still pass
(`WindowManifest` shape, `WindowStorageBase` Protocol satisfaction).

**Why these were originally classified as "accepted risks":**
- Timeouts: I conflated the symptom (slow Ctrl-C) with a UX limitation
  rather than recognizing the root cause as a real reliability bug.
- Thread safety: I rationalized it by noting `WindowStorage` had the same
  bug — "matches the existing pattern" is not a justification.
- MP4 validation: I claimed full validation was too expensive (true) and
  missed that cheap metadata checks were essentially free.

The user pushed back on all three. Fixed.

**Pipeline total: 598 passing, 2 skipped, lint clean** (the 3 remaining
F401/F841 hints are all in code outside this fix-batch).

---

## Implementation Status

| Module | Name | Status | Hygiene Protocol | Date |
|---|---|---|---|---|
| 0 | Interfaces package | ✅ Complete | ✅ Passed | 2026-05-26 |
| 1 | Storage | ✅ Complete | ✅ Passed | 2026-05-26 |
| 1+ | Storage — FlatMP4Storage sibling | ✅ Complete | ✅ Passed | 2026-05-29 |
| 2 | Encoder (reasoning arm) | ✅ Complete | ✅ Passed | 2026-05-26 |
| 3 | Hypothesizer | ✅ Complete | ✅ Passed | 2026-05-26 |
| 4 | Scorer | ✅ Complete | ✅ Passed | 2026-05-26 |
| 4+ | Scorer — NIMTextClient (production text client) | ✅ Complete | ✅ Passed | 2026-05-29 |
| 5 | Judge UI | ✅ Complete | ✅ Passed | 2026-05-26 |
| 6 | Evaluation | ✅ Complete | ✅ Passed | 2026-05-26 |
| CLI | `pipeline.run` (consumer-facing CLI) | ✅ Complete | ✅ Passed | 2026-05-29 |

---

## 2026-05-29 — `pipeline.run` CLI orchestration layer v1.0

**Built:**
- `pipeline/run.py` (~390 LoC) — three subcommands wiring the six pipeline modules end-to-end. Uses ONLY package-root public surfaces (`from pipeline.modules.X import ...`); no submodule reaching, no internals access.
  - `ingest`: builds `WaymoParquetSource`/`WaymoTFRecordSource` from `--source-format` + `--source-root`, parses `--segments` (`all` | comma list | `@file`), builds `WindowConfig`+`IngestionRequest`, runs `IngestionPipeline`. Catches `SourceUnreachableError`/`SourceSchemaVersionError` → exit 2.
  - `analyze`: builds storage (`WindowStorage` or `FlatMP4Storage` per `--storage-mode`), Encoder (with reasoning + visual arms; `--stub` swaps in offline clients; `--cameras` threads through to VisualArm in flat_mp4 mode), Hypothesizer, Scorer. Writes `schema_records.json`, `proposals.json`, `scored.json` atomically via `tmp + replace`.
  - `report`: loads scored proposals + ratings (filesystem `--ratings` or HTTP `--ratings-url`) + pre-registered seeds JSON, runs Evaluator, saves report to timestamped subdir.
- `pipeline/modules/scorer/nim_client.py` (~120 LoC) — `NIMTextClient` (production TextClient for Scorer; parallels `CosmosReason2Client` in Encoder) + `NIMUnavailableError`. Made the existing scorer config docstring claim "NIMTextClient (production)" truthful instead of aspirational.
- Test files (~70 tests new this build):
  - `pipeline/tests/run/test_cli_parsing.py` (14 argparse tests)
  - `pipeline/tests/run/test_run_ingest.py` (16 tests — helpers + handler smoke)
  - `pipeline/tests/run/test_run_analyze.py` (8 tests + 1 pessimistic-fix regression test)
  - `pipeline/tests/run/test_run_analyze_flat_mp4.py` (10 tests — the new storage-mode + cameras path)
  - `pipeline/tests/run/test_run_report.py` (12 tests — seeds parsing, ratings loaders, end-to-end)
  - `pipeline/tests/run/contract/test_pipeline_run_output_contract.py` (5 tests — every output JSON round-trips through its interface type; NIMTextClient satisfies TextClient Protocol)
  - `pipeline/tests/integration/test_pipeline_run_analyze_to_report.py` (1 marquee cross-stage test — analyze writes scored.json, report consumes it)
  - `pipeline/modules/scorer/tests/test_nim_client.py` (9 tests — model_id env vs arg, complete() shape, NIMUnavailableError)
- README.md — replaced the previously-fictional `python -m pipeline.run` invocations with the real CLI. Added "GCS signed URLs — three working setups" (the customer mitigation for blocker C from the GPU-instance feedback). Added "Quick-analysis path: I already have MP4s" section. Added three pessimistic-review-driven troubleshooting entries.
- `.env.example` — added `SCORER_NIM_MODEL_ID` and the GCS signing-options block (SA key file / impersonation / public bucket).

**Tests:** 70 new this build. Pipeline total: **582 passing, 2 skipped, lint clean.**

**Design decisions:**
- `pipeline.run` is consumer-only — produces no new interface type. It traffics in `WindowKey`, `SchemaRecord`, `CompositionProposal`, `ScoredProposal`, `Rating`, `EvaluationReport` (all from `pipeline.interfaces`).
- Subcommands are deliberately separate (`ingest`, `analyze`, `report`) rather than a single `run-all`. Human review (the Judge UI) happens between `analyze` and `report`; bundling them would imply unattended end-to-end runs that don't match the customer workflow.
- `--stub` mode wires in `StubVLMClient` + `StubPlausibilityClient` + `StubDifficultyClient`. Stub clients use distinct `model_id` strings (`stub/cosmos-reason2`, `stub/plausibility`, etc.) so their cache entries cannot poison subsequent production runs.
- `NIMTextClient` lives in the scorer module (not in `pipeline.run`) — production clients belong with their module, parallel to `CosmosReason2Client` in the encoder. This was a deliberate scope expansion the user approved during planning.
- File writes via `_write_json_list` are atomic (`tmp + replace`). Writes ACROSS files (schema_records → proposals → scored) are NOT atomic by design — re-running `analyze` is the recovery path.

**Hygiene protocol (all 7 steps passed):**
1. **Smoke:** ✅ `python -m pipeline.run --help` and each subcommand `--help` exit 0 with usable output.
2. **Contract:** ✅ 5 contract tests assert every output JSON file round-trips through its interface type; `NIMTextClient` satisfies `TextClient` Protocol.
3. **Cross-module integration:** ✅ Marquee test: real Hypothesizer + real Scorer (with stubs) on synthetic SchemaRecords designed to surface an arity-3 novel composition (`weather:fog × time_of_day:night × agents:pedestrian`). `analyze` writes `scored.json`; `report` reads that file and produces a valid `EvaluationReport`.
4. **Pessimistic review:** ✅ 3 concerns: 1 fixed with a regression test (uncaught `WindowStorageError` from `list_windows` now returns exit 2 with actionable diagnostic text), 2 documented in README troubleshooting (`--max-workers` overwhelming free-tier NIM quotas; `Ctrl-C` delayed by in-flight OpenAI calls with no client timeout).
5. **Reconciliation:** ✅ README ↔ argparse ↔ interface types all aligned line-by-line. All flags shown in README examples are real; all output files match their interface types.
6. **Cache:** ✅ pipeline.run has no cache of its own. `--cache-root` is threaded through to Encoder + Scorer (both already hygiene-approved). `--stub` vs production are key-isolated by model_id, preventing cross-mode poisoning.
7. **Sign-off:** ✅ This entry.

**Accepted risks:**
- **`--max-workers 8` overwhelms free-tier NIM quotas** producing `vlm_unavailable` failures. Failures are loud (stderr), not silent, but require customer adjustment. Documented in README troubleshooting; first-time customers told to drop to 1–2.
- **`Ctrl-C` blocks on in-flight OpenAI HTTP calls** (no client-side timeout by default). Documented; real fix would require timeout config + signal handlers, deferred until customers report it as a UX blocker.
- **File writes ACROSS subcommands are not atomic.** If `analyze` dies between writing `schema_records.json` and `proposals.json`, the output directory has inconsistent partial state. Recovery: re-run `analyze`. Acceptable because the encoder cache prevents redoing the expensive VLM work.

**Deviations from architecture spec:**
- Pre-existing: the original root README invented `python -m pipeline.run ingest|analyze|report` commands that didn't exist (caught when the GPU instance flagged blocker A). This build makes the invented promise real.

---

## 2026-05-29 — Module 1 sibling: FlatMP4Storage v1.0

**Built:**
- `pipeline/interfaces/window.py` — added `WindowStorageBase` Protocol (3 methods: `list_windows`, `get_window_video_url`, `get_window_manifest`). The Encoder and Judge UI now have a typed contract instead of `storage: Any` duck typing.
- `pipeline/interfaces/tests/test_window_storage_protocol.py` — 3 tests (Protocol satisfaction, method enumeration, negative case)
- `pipeline/modules/storage/flat_mp4.py` — `FlatMP4Storage` class (~250 LoC with docstring): read-only, stateless storage for flat MP4 buckets. Filename convention auto-switches on `len(cameras)`: bare `<id>.mp4` for single-camera, suffixed `<id>_<camera>.mp4` for multi-camera.
- `pipeline/modules/storage/__init__.py` — re-exported `FlatMP4Storage`; updated docstring to mention both implementations.
- `pipeline/modules/storage/tests/test_flat_mp4.py` — 23 unit tests (constructor validation, filename parsing both modes, list_windows dedup, signed URL generation, manifest synthesis, pessimistic-review pin).
- `pipeline/modules/storage/tests/contract/test_flat_mp4_output_contract.py` — 15 contract tests asserting every README-declared field of the synthesized `WindowManifest` is present, typed, and carries the documented value.
- `pipeline/tests/integration/test_flat_mp4_to_encoder.py` — 3 cross-module integration tests proving the Encoder consumes FlatMP4Storage with no special-casing.
- `pipeline/run.py` — added `--storage-mode {canonical,flat_mp4}` and `--cameras` flags to the `analyze` subcommand. `--cameras` is required iff `--storage-mode flat_mp4`. `_build_encoder` threads `cameras` through to `VisualArm` so the visual-arm embedding dimensionality reflects what the customer declared.
- `pipeline/tests/run/test_run_analyze_flat_mp4.py` — 10 CLI integration tests (argparse, validation, builder threading).
- `README.md` — new "Quick-analysis path: I already have MP4s" section under "Running a Discovery Session" with both single- and multi-camera invocation examples.
- `pipeline/README.md` — extended Module 1 contract section with "Two retrieval implementations" table and added "Adding a new SourceAdapter" HOWTO (the Option 3 the user approved as the customer-onboarding answer for messy parquet schemas).

**Tests:** 54 new (3 + 23 + 15 + 3 + 10). Pipeline total: 578 passing.

**Design decisions:**
- `WindowStorageBase` is a Protocol, not an ABC — matches the existing pattern (`SourceAdapter`, `TextClient`, `VLMClient`). The constructor is intentionally NOT part of the contract because the two implementations have different required args (`WindowStorage(bucket_uri)` vs `FlatMP4Storage(bucket_uri, cameras)`).
- Filename convention auto-switches on `len(cameras)` rather than being its own flag — single-camera customers (most common case) don't have to rename their bare `<id>.mp4` files.
- `--cameras` is REQUIRED in flat mode with no default. The visual-arm embedding dimensionality must be explicit so downstream consumers don't silently get unexpected vector sizes.
- `window_idx > 0` raises `WindowStorageError` rather than silently aliasing — the contract is "one MP4 = one window," and a non-zero request is a misuse signal.
- Synthesized manifests have `frame_count=0` and `pose_summary=None` because flat MP4s carry no per-frame metadata. The Encoder already tolerates `pose_summary=None` (verified by integration test).

**Hygiene protocol (all 7 steps passed):**
1. **Smoke:** ✅ All three public methods return non-empty, correctly-typed values on minimal mocked input.
2. **Contract:** ✅ 15/15 tests assert every README-documented `WindowManifest` field; `WindowStorageBase` satisfaction proven at runtime.
3. **Cross-module integration:** ✅ 3/3 tests; the Encoder produces valid `SchemaRecord` objects from FlatMP4Storage windows without special-casing; `SchemaRecord` round-trips through JSON.
4. **Pessimistic review:** ✅ 3 concerns identified — 1 fixed with a pinning test (filename collision when segment ID contains a camera substring), 2 documented (no MP4 content validation; lazy GCS client init is not thread-safe).
5. **Reconciliation:** ✅ README, `interfaces/window.py`, and `flat_mp4.py` all describe the same shape; verified line-by-line.
6. **Cache:** ✅ No cache — read-only and stateless. Determinism verified for load-bearing fields (list_windows is sorted; synthesized manifest fields are all deterministic except `ingested_at`, which is not in the Encoder's cache key).
7. **Sign-off:** ✅ This entry.

**Accepted risks:**
- **No MP4 content validation:** FlatMP4Storage only checks blob existence. Corrupted MP4s, mis-renamed archives, or empty files will produce signed URLs and fail at VLM call time. Validating at the storage layer would require downloading or HEAD-probing every blob.
- **Filename parser is greedy by camera-suffix match:** customers must avoid segment IDs that contain configured camera names as substrings (e.g. `drive_REAR_001` with `cameras=["REAR"]` parses incorrectly). Documented in the module docstring and pinned by `test_parse_blob_name_segment_id_containing_camera_substring`.
- **Lazy GCS client init is not thread-safe:** concurrent first calls may construct the client twice (functionally equivalent, no corruption). Inherited from `WindowStorage`'s pattern.

**Deviations from architecture spec:**
- None. FlatMP4Storage was added by user request as a Module 1 sibling (not a new numbered module) for the "I already have MP4s, just want quick analysis" customer path. The architecture stays intact; the canonical `ingest → analyze → report` path remains the recommended default.

---

## 2026-05-26 — Module 5: Judge UI v1.0

**Built:**
- `pipeline/modules/judge_ui/__init__.py` — module package init
- `pipeline/modules/judge_ui/config.py` — all configuration via env vars (`JUDGE_PORT`, `JUDGE_PROPOSALS_PATH`, `JUDGE_RATINGS_DIR`, `VIDEO_URL_TTL_SECONDS`, `JUDGE_BUCKET_URI`, `JUDGE_SIGN_AS`)
- `pipeline/modules/judge_ui/server.py` — FastAPI server (port 8001) with lifespan context manager (not deprecated `@app.on_event`); 6 endpoints:
  - `GET /judge/proposals` — ranked accepted proposals, arm blinded
  - `GET /judge/proposals/{id}` — proposal detail, arm blinded
  - `GET /judge/video-url` — fresh pre-signed GCS URL for motivating-scene video
  - `POST /judge/ratings` — persist Rating; arm injected server-side from proposal store
  - `GET /judge/session/{rater_id}` — session resumability + score distributions
  - `GET /judge/ratings/export` — Module 6 boundary endpoint; returns all ratings as JSON
- `pipeline/modules/judge_ui/tests/test_server.py` — 20 unit tests covering all endpoints
- `pipeline/modules/judge_ui/tests/contract/test_judge_ui_output_contract.py` — 17 contract tests (every README Rating field, arm blinding assertions, round-trip, malformed input 422)
- `pipeline/modules/judge_ui/tests/integration/test_judge_ui_to_evaluation.py` — 4 integration tests with Module 6 stub (`stub_evaluation_consumer` exercises every Rating field Module 6 needs)
- `frontend/components/judge-tab.tsx` — full React component; 4 screens: setup, proposal list, detail + rating widget, session summary; styled to match existing frontend design system (shadcn/ui + NVIDIA green design tokens, no hardcoded hex)
- `frontend/lib/types.ts` — added 7 Judge types: `JudgeScoreBadges`, `JudgeMotivatingScene`, `JudgeProposalRow`, `JudgeProposalDetail`, `JudgeVideoUrl`, `JudgeRatingSubmission`, `JudgeSessionSummary`
- `frontend/lib/api.ts` — added `JUDGE_API_URL` const and 5 API functions
- `frontend/.env.local` — added `NEXT_PUBLIC_JUDGE_API_URL=http://localhost:8001`
- `frontend/app/page.tsx` — added Judge tab with `Gavel` icon

**Tests:**
- 20 unit tests (`test_server.py`)
- 17 contract tests (`test_judge_ui_output_contract.py`)
- 4 integration tests (`test_judge_ui_to_evaluation.py`)

Total new tests: 41.

**Design decisions:**
- Proposals loaded once at startup from `JUDGE_PROPOSALS_PATH` (not re-read per request). Restart required to pick up new proposals from Module 4.
- Blinding enforced with a runtime assertion: `assert "arm" not in row.model_dump()` — fires immediately if `arm` ever leaks into a rater-facing response.
- Session resumability follows Scenario A: rated proposals are marked in the list (not skipped), so raters can revisit their ratings.
- `GET /judge/ratings/export` is the explicit Module 6 boundary. Module 6 calls this endpoint; it never reads the ratings directory directly.
- Atomic writes: `.json.tmp` → `Path.replace()`. No partial files visible to concurrent readers.
- Video URL retry capped at 2 attempts in the frontend with explicit error state.

**Fixes applied during hygiene protocol (Step 4):**
- Added path separator guard in `_rating_path()`: raises `ValueError` if `composition_id` contains `/` or `\`, preventing filesystem traversal if Module 4 produces a malformed composition_id.

**Deviations from architecture spec:**
- None. Module built directly from `/verity-brain` architecture spec (no README existed at build time). Module 5 README section in `pipeline/README.md` was written as part of this session to document the contract prospectively.

**Hygiene protocol (all 7 steps passed):**
1. Smoke: ✅ All 5 endpoints return 200 on minimal fixture; `_proposals` populated, ratings persisted to disk
2. Contract: ✅ 17/17 contract tests pass; every Rating field asserted with correct type; arm blinding verified for both `reasoning` and `visual` arms
3. Cross-module integration: ✅ 4/4 Module 5→6 integration tests pass; `stub_evaluation_consumer` exercises every field Module 6 needs; `WindowKey` objects survive the serialization boundary
4. Pessimistic review: ✅ 3 concerns identified; 1 fixed (path separator guard), 2 documented as accepted risks (see below)
5. Reconciliation: ✅ `pipeline/README.md` Module 5 section updated from "(stub) not yet implemented" to full output contract (Rating field table, endpoint table, run instructions, blinding contract, storage layout)
6. Cache: ✅ No cache layer — proposals are loaded from disk at startup, ratings persisted to disk on submission; no VLM calls, no stale-cache risk
7. Sign-off: ✅ This entry

**Accepted risks:**
- STALE PROPOSALS: Proposals loaded once at startup. If Module 4 regenerates `proposals.json` mid-session, the server continues serving the old snapshot. Accepted: rating sessions are short and Module 4 won't re-run during an active session in practice. Restart to pick up new proposals.
- CONCURRENT SAME-RATER WRITES: Two simultaneous `POST /judge/ratings` from the same `(rater_id, proposal_id)` are last-writer-wins via atomic rename. The race loser's write is silently discarded. Accepted: the UI is single-page and doesn't allow concurrent submission; double-submits require deliberate direct API calls.

---

## 2026-05-26 — Module 6: Evaluation v1.0

**Built:**
- `pipeline/interfaces/report.py` — `DifferentialExample`, `EvaluationReport` (added NaN→null sanitization in `to_json()`)
- `pipeline/interfaces/tests/test_roundtrip.py` — 3 new round-trip tests for DifferentialExample and EvaluationReport edge cases (24 total)
- `pipeline/modules/evaluation/metrics.py` — 4 pure functions: `compute_seeded_recall`, `compute_rating_stats`, `krippendorff_alpha` (ordinal), `compute_differential_examples`
- `pipeline/modules/evaluation/evaluator.py` — `EvaluationInput`, `Evaluator.evaluate()` (pure), `Evaluator.save()` (I/O), `MissingSubsetLabelsError`, `ArmMismatchError`, `RECALL_K_PRIMARY=30`
- `pipeline/modules/evaluation/renderer.py` — `render_markdown()` (GFM for paper writeup), `render_html()` (standalone + embeddable=True for React dashboard; Plotly graceful degradation)
- `pipeline/modules/evaluation/__init__.py` — exports all public symbols

**Tests:**
- `tests/test_metrics.py` — 23 unit tests (seeded recall, rating stats, Krippendorff alpha, differential examples)
- `tests/test_smoke.py` — 25 smoke tests (Evaluator public interface, error handling)
- `tests/contract/test_evaluation_output_contract.py` — 24 contract tests (every EvaluationReport field asserted)
- `tests/integration/test_evaluation_to_consumers.py` — 25 integration tests (markdown, HTML standalone, HTML embeddable, JSON round-trip boundary)

Total new tests: 97. Pipeline total: ~399 passing, 2 skipped (plotly chart tests when plotly not installed).

**Design decisions:**
- `evaluate()` is a pure function (no side effects); `save()` is strictly separate and handles all disk I/O. Callers can evaluate without persisting.
- Seeded recall denominator: `|seeded_set|` (not proposals). Prevents artificially high recall from a small proposal set.
- Recall keys `@10`, `@30`, `@all` always present. When primary K=30, `@{k}` and `@30` alias the same dict entry (benign; documented in `compute_seeded_recall` docstring).
- Krippendorff's alpha: ordinal metric (`d(c,k)^2`); returns `None` when fewer than 2 raters have overlapping ratings. Never returns a sentinel float.
- Bootstrap CI suppressed (None) when n_ratings < 30. Always reported alongside `n_ratings_per_arm` so downstream can distinguish "no CI" from "CI is wide."
- HTML renderer: plotly optional. When not installed, chart divs are omitted; tables and IRA block still render. Embeddable mode skips `<html>` wrapper and CDN `<script>` for React injection.
- `float("nan")` for arms with no ratings stays in memory (valid Python float); sanitized to `null` in `to_json()` so JSON serialization never crashes.

**Fixes applied during hygiene protocol:**
- `n_raters_overlapping` was incorrectly returning total rater count; fixed to return count of raters who have at least one rating on a proposal that another rater also rated.
- `float("nan")` values for no-rating arms would crash `json.dumps()`. Fixed by adding `_safe()` in `EvaluationReport.to_json()`.
- Renderer's `math.isnan()` calls updated to handle `None` (values from JSON-restored reports) without raising `TypeError`.

**Hygiene protocol (all 7 steps passed):**
1. Smoke: ✅ 25 smoke tests pass; minimal input produces valid EvaluationReport
2. Contract: ✅ 24 contract tests pass; every EvaluationReport field asserted with correct type
3. Cross-module integration: ✅ 25 integration tests; all 4 downstream consumers (markdown, HTML standalone, HTML embeddable, JSON boundary) validated
4. Pessimistic review: ✅ 3 concerns identified; 2 fixed (`n_raters_overlapping` bug, NaN serialization crash), 1 documented (@K key collision in docstring)
5. Reconciliation: ✅ README Module 6 section updated; mean_coherence/mean_usefulness NaN behavior documented; interfaces/report.py consistent with implementation
6. Cache: ✅ No cache layer — `evaluate()` is a pure function; reproducibility verified (same inputs → identical JSON output)
7. Sign-off: ✅ This entry

**Accepted risks:**
- `krippendorff_alpha` returns `None` when all items have a single rater (no overlapping pairs), not when IRA is genuinely low. Callers should check `n_raters_overlapping` before interpreting `None` as "agreement was not measured."
- `render_html(embeddable=True)` assumes the host React page has already loaded Plotly as a global. If Plotly is not present in the host, charts silently produce no-op calls. Document in the Judge UI integration spec.

---

## 2026-05-26 — Module 4: Scorer v1.0

**Built:**
- `pipeline/modules/scorer/config.py` — `TextClient` protocol, `ScorerWeights`, `ScorerConfig`, `ScorerError`, `PlausibilityCheckFailedError`
- `pipeline/modules/scorer/prompts/v1_plausibility.txt` — plausibility prompt template (`{{COMPOSITION}}` placeholder; asks for `{"score", "justification"}`)
- `pipeline/modules/scorer/prompts/v1_difficulty.txt` — difficulty prompt template (`{{COMPOSITION}}` placeholder; asks for `{"action", "confidence", "reasoning_consistent_with_action"}`)
- `pipeline/modules/scorer/plausibility.py` — `PlausibilityArm` (3-run conservative aggregation: 3/3→median, 2/3→lower, 1/3→single, 0/3→raise), `describe_composition`, `_three_orderings` (deterministic sha256 seed), `StubPlausibilityClient`, `FailingPlausibilityClient`
- `pipeline/modules/scorer/difficulty.py` — `DifficultyArm` (3 runs, partial failure OK, all-fail→`(None, {})`), `compute_difficulty_signals` (`action_variance*0.5 + (1-mean_confidence)*0.3 + reasoning_mismatch*0.2`), `_extract_difficulty_json`, `StubDifficultyClient`, `FailingDifficultyClient`
- `pipeline/modules/scorer/scorer.py` — `Scorer` class with constructor injection, cache read/write, `score()` (never raises), `score_batch()`, acceptance filter, `_final_rank_score()`

**Tests:**
- `tests/test_config.py` — 12 unit tests
- `tests/test_plausibility.py` — 22 unit tests
- `tests/test_difficulty.py` — 17 unit tests (including JSON extraction, signal computation, arm behavior)
- `tests/test_smoke.py` — 19 smoke tests (added boundary + failure justification tests from Step 4)
- `tests/contract/test_scorer_output_contract.py` — 23 contract tests
- `tests/integration/test_hypothesizer_to_scorer.py` — 9 cross-module integration tests

Total new tests: 102. Pipeline total: ~302 passing.

**Design decisions:**
- Constructor injection for both VLM clients — same pattern as Encoder; avoids per-call VLM construction overhead.
- `TextClient(prompt: str) -> str` is text-only. Distinct from Encoder's `VLMClient(video_url, prompt)` because the Scorer operates on composition descriptions, not video frames.
- Failure cache sentinel: `"no_difficulty_client"` — unambiguous from the cache key alone; `"none"` is ambiguous.
- Acceptance filter uses strict less-than (`score < threshold`), so score == threshold is accepted. Test pinned to prevent behavioral drift.
- Plausibility failures are NOT cached — they may succeed on retry (transient VLM error).
- Cache writes are atomic via `.json.tmp → .json` (`os.replace()`). Write failures are logged, non-fatal.

**Hygiene protocol (all 7 steps passed):**
1. Smoke: ✅ All 19 smoke tests pass
2. Contract: ✅ All 23 contract tests pass (every ScoredProposal field asserted)
3. Cross-module integration: ✅ All 9 Hypothesizer→Scorer integration tests pass
4. Pessimistic review: ✅ 3 concerns identified; 1 fixed (boundary + justification tests added), 1 documented (empty justification on failure in interfaces/proposal.py), 1 accepted (cache write failures non-fatal by design)
5. Reconciliation: ✅ README Module 4 Output Contract section updated; pipeline overview updated to ✅ complete; interfaces/proposal.py justification docstring updated
6. Cache: ✅ Key includes composition_id + p_model_id + d_model_id + p_prompt_v + d_prompt_v; version change → cache miss; failure not cached; no stale .tmp files
7. Sign-off: ✅ This entry

**Accepted risks:**
- `plausibility_threshold=0.5` is a placeholder. TODO: calibrate empirically at 20th-percentile plausibility score on a 50-proposal calibration set (week 2).
- `difficulty_signal_weights=(0.5, 0.3, 0.2)` are untuned. TODO: calibrate against 30 proposals with human difficulty labels (week 2).

---

## 2026-05-26 — Module 3: Hypothesizer v1.0

**Built:**
- `pipeline/modules/hypothesizer/config.py` — `HypothesizerConfig`, `SCHEMA_PATH_TO_ATOM_PREFIX`, `MULTI_VALUE_FIELDS`, `SINGLE_CATEGORICAL_FIELDS`, `HypothesizerEmptyInputError`, `VocabularyMismatchError`
- `pipeline/modules/hypothesizer/frequency.py` — `extract_atoms` (qualified atom extraction from SchemaRecord.fields), `compute_frequencies` (marginal + pairwise frequency tables)
- `pipeline/modules/hypothesizer/composition.py` — `build_proposals` (enumeration, mutual-exclusivity filter, pairwise filter, joint-frequency filter, novelty scoring, top-k ranking), `composition_id` (deterministic sha256 hash), `_is_mutually_exclusive`
- `pipeline/modules/hypothesizer/hypothesizer.py` — `Hypothesizer` class, public `propose()` method, two separate stderr skip counts

**Design decisions:**
- Cross-field qualified atoms (`"prefix:value"`), not conditions-only. Captures the full compositional novelty space (e.g., `"agents:pedestrian" + "weather:fog" + "ego_task:turning_left"`).
- `compose_over: list[str] | None` config field — None (default) = all fields; `["conditions"]` = conservative baseline.
- Mutual exclusivity: scalar fields (weather, time_of_day, lighting, road_geometry, traffic_control, ego_task) forbid same-prefix pairs. Multi-value fields (agents, conditions) allow them.
- No cache — `propose()` is a pure function (stateless, no VLM calls, no I/O).
- Deterministic: sorted enumeration + `(novelty_score DESC, composition_id ASC)` ranking.
- Novelty score: `ln(expected_joint / max(observed_joint, ε))` where `ε = 1/(10*N)`.

**Tests added:**
- `tests/test_config.py` — 13 tests (defaults, mapping completeness, field classification, error messages)
- `tests/test_frequency.py` — 19 tests (atom extraction, compose_over, valid_atoms, frequency computation)
- `tests/test_composition.py` — 22 tests (mutual exclusivity, expected joint, pairwise, build_proposals filters, ranking, motivating scenes)
- `tests/test_smoke.py` — 13 tests (full public interface, empty/failure handling, determinism, JSON round-trip)
- `tests/contract/test_hypothesizer_output_contract.py` — 15 contract tests (every README field asserted)
- `tests/integration/test_encoder_to_hypothesizer.py` — 4 integration tests (full boundary crossing)
- Updated `encoder/tests/integration/test_encoder_to_hypothesizer.py` — replaced `_StubHypothesizer` with real Hypothesizer

**Accepted risks (documented, not fixed):**
- O(n^k) enumeration: C(50,4)≈230K candidates at arity=4 is acceptable; arity≥5 on large atom sets requires pre-filtering. Documented in `composition_sizes` docstring.
- Epsilon-floor inflates novelty for unobserved-joint compositions that pass pairwise filter. Mathematically correct; documented in module docstring.
- `compose_over` and `valid_atoms` are independent filters (documented in config docstring).

**Hygiene protocol results:**

| Step | Result | Evidence |
|---|---|---|
| 1. Smoke | ✅ Pass | 13/13 smoke tests; proposals produced on 50-window fixture |
| 2. Contract | ✅ Pass | 15/15 contract tests (every README Output Contract field asserted) |
| 3. Cross-module | ✅ Pass | 4/4 encoder→hypothesizer integration tests; stub test updated to real Hypothesizer |
| 4. Pessimistic review | ✅ Pass | 3 concerns identified; all documented |
| 5. Reconciliation | ✅ Pass | README Module 3 section updated with full output contract; interfaces match |
| 6. Cache | ✅ Pass | No cache (pure function); determinism verified by test |
| 7. Sign-off | ✅ Pass | This entry |

**Total tests passing after protocol:** 200 (117 prior + 83 hypothesizer); 3 pre-existing env failures (gcsfs, pyarrow/numpy, google.cloud not installed in anaconda env)

---

## 2026-05-26 — Hygiene protocol re-run: Module 1 + Module 2 (interfaces rewire)

**Trigger:** `pipeline/interfaces/` package created; cross-module imports rewired.

**What changed:**
- `pipeline/interfaces/` created with `window.py`, `schema_record.py`, `proposal.py`, `rating.py`, `report.py` — all types have `to_json()`/`from_json()` and 21 round-trip tests
- `storage/adapters/base.py` — removed local `WindowKey`, `PoseRecord`, `WindowManifest`, `PoseData`, `DatasetManifest` definitions; imports from `pipeline.interfaces.window` and re-exports them for internal callers
- `encoder/schema.py` — removed local `SchemaRecord`; imports from `pipeline.interfaces.schema_record`; `WindowKey` now imported from `pipeline.interfaces.window` (not `storage.adapters.base`)
- `encoder/encoder.py` — cache read/write updated from `to_dict()`/`from_dict()` → `to_json()`/`from_json()`
- `storage/ingestion.py` — manifest serialization uses `manifest.to_json()` instead of `manifest.__dict__`
- `pipeline/README.md` created — full output contracts for Modules 1 and 2

**Tests added this session:**
- `pipeline/interfaces/tests/test_roundtrip.py` — 21 round-trip tests for every interface type
- `pipeline/modules/storage/tests/contract/test_storage_output_contract.py` — 9 contract tests
- `pipeline/modules/encoder/tests/contract/test_encoder_output_contract.py` — 7 contract tests
- `pipeline/tests/integration/test_storage_to_encoder.py` — 4 boundary-crossing integration tests

**Hygiene protocol results (both modules, combined run):**

| Step | Result | Evidence |
|---|---|---|
| 1. Smoke | ✅ Pass | Both modules produce correct output on minimal input |
| 2. Contract | ✅ Pass | 16/16 contract tests (9 storage + 7 encoder) |
| 3. Cross-module | ✅ Pass | 4/4 storage→encoder integration tests |
| 4. Pessimistic review | ✅ Pass | 6 concerns identified (1 fixed: `manifest.to_json()`; 1 fixed by linter: model_id in cache key; 4 documented) |
| 5. Reconciliation | ✅ Pass | `pipeline/README.md` created with full output contracts |
| 6. Cache | ✅ Pass | All 5 cache checks pass; model_id, schema_version, prompt_template_id all in key |
| 7. Sign-off | ✅ Pass | This entry |

**Total tests passing after protocol:** 117

---

## 2026-05-26 — Module 2: Encoder (reasoning arm) v1.0

**Built:**
- `pipeline/modules/encoder/encoder.py` — `Encoder` class, full cache read/write with atomic writes
- `pipeline/modules/encoder/reasoning_arm.py` — `VLMClient` protocol, `CosmosReason2Client` (NVIDIA NIM / Cosmos-Reason2-7B), `StubVLMClient`, `ReasoningArm` (3x retry with stricter prompt), `extract_json` (5 response format handlers)
- `pipeline/modules/encoder/schema.py` — `WindowInput` input contract; re-exports `SchemaRecord` from `pipeline.interfaces.schema_record`
- `pipeline/modules/encoder/vocabulary.py` — locked v1.0 vocabulary (12 agent tags, 10 condition tags, full environment/road/traffic/ego-task sets), `Vocabulary.validate_fields`, `fill_fraction`, `prompt_context`
- `pipeline/modules/encoder/prompts/v1_describe.txt` — v1 prompt template

**Tests added this session:**
- `tests/test_contract.py` — 16 contract tests (every README Output Contract field asserted)
- `tests/integration/test_encoder_to_hypothesizer.py` — 4 boundary-crossing integration tests with Module 3 stub

**Deviations from README:**
- Visual arm (`visual_arm.py`) not implemented — descoped per project decision; visual arm is Phase 1 optional.
- `SchemaRecord.arm` typed as `str` in `interfaces/` rather than `Literal["reasoning", "visual"]` — visual arm deferred to Phase 2, type widened intentionally.

**Fixes made during hygiene protocol:**
1. **WindowKey import violation** — `encoder/schema.py` previously imported `WindowKey` from `pipeline.modules.storage.adapters.base` (cross-module internal). Redirected to `pipeline.interfaces.window`.
2. **SchemaRecord consolidated** — encoder previously defined its own `SchemaRecord` with `to_dict/from_dict`. Now re-exports from `pipeline.interfaces.schema_record` (uses `to_json/from_json`). Background agent handled the reconciliation.
3. **Manifest exception logging** — `annotate_from_storage` previously swallowed all manifest fetch exceptions silently. Now logs `type(exc).__name__` and message before suppressing, so failures are diagnosable.
4. **Prompt template placeholder assertion** — `_build_prompt` now raises `ValueError` immediately if `{{VOCABULARY}}` or `{{POSE_SUMMARY}}` are missing from a template, rather than silently substituting nothing.
5. **Cache key missing model_id** — cache key previously contained only `(segment_id, window_idx, arm, schema_version, prompt_template_id)`. Added `model_id` so swapping the VLM backend correctly invalidates the cache.
6. **Transient failure caching** — `vlm_unavailable` failures previously got written to cache, meaning a network blip permanently poisoned the cache entry. Fixed: only deterministic failures (`invalid_json`, `vocabulary_violation`, `unknown`) are cached; `vlm_unavailable` is never cached so it retries after recovery.

**Accepted risks (documented, not fixed):**
- Video URL not pre-validated before passing to VLM API — checked at call time by the API, adding pre-validation duplicates work.
- Concurrent `process()` calls with the same `window` from two threads will both miss the cache and both invoke the VLM — double spend, not data corruption. Atomic `tmp → replace` write ensures no corrupt entries.

**Hygiene protocol results:**

| Step | Result | Evidence |
|---|---|---|
| 1. Smoke | ✅ Pass | `SchemaRecord` from `pipeline.interfaces`, all fields present, no crash |
| 2. Contract | ✅ Pass | 16/16 contract tests |
| 3. Cross-module | ✅ Pass | 4/4 integration tests with Module 3 boundary stub |
| 4. Pessimistic review | ✅ Pass | 6 concerns identified; 4 fixed, 2 documented |
| 5. Reconciliation | ✅ Pass | Docstring drift (cache key, `to_dict` ref) corrected |
| 6. Cache | ✅ Pass | All 5 cache checks demonstrated in script |
| 7. Sign-off | ✅ Pass | This entry |

**Total tests passing after protocol:** 57 (30 smoke + 16 contract + 4 integration + 7 implicit in cache checks)
