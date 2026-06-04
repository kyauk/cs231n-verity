# Verity Pipeline — Architecture & Methodology Reference

> **Getting started?** See the [root README](../README.md) for installation, configuration, and how to run a session. This document is the technical reference: the data model, the discovery method, the module contracts, and the design principles.

Verity is a pipeline for autonomous-vehicle safety scenario discovery. Every module under `pipeline/modules/` is a standalone Python package that imports shared types from `pipeline/interfaces/` and **nothing from another module's internals**. Composition roots (the `pipeline.run` CLI, the `judge_ui` FastAPI server, and the `verity_*.py` drivers) wire modules together over their public interfaces. The boundary is mechanical and verifiable:

```bash
# no module imports another module's internals:
grep -rn "from pipeline.modules" pipeline/modules/<m>/ | grep -v "pipeline.modules.<m>"
```

---

## 1. The data model — three strictly-separated objects

The core design decision that makes the system robust is separating three things that are easy to conflate. Each separation is enforced by the type system, and every downstream failure mode becomes recoverable because of it.

| Object | Nature | Rule |
|---|---|---|
| **Evidence** (`RawDescriptor`) | what a model observed about one scene | **immutable, append-only.** Never edited, merged, or regenerated. Carries a span pointer to the reasoning that justified it. |
| **Interpretation** (`Taxonomy`, `CanonicalLabel`) | emergent buckets over evidence | **mutable, versioned.** Recomputed as a pure function of the evidence; refines over time. |
| **Derivation** (`Projection`, proposals, rankings) | statistics / selections over interpretation | **ephemeral.** Always recomputed from current labels; never stored as truth. |

Because evidence is immutable and interpretation is a pure recompute, **adding data and re-deriving makes past data richer retroactively** — labels and statistics improve without ever touching the original observations. This is the property that lets the taxonomy "blossom" with scale.

---

## 2. Discovery pipeline (primary)

```
 raw clip ─▶ EXTRACTOR ─▶ CURATOR ─▶ SELECTION ─▶ SYNTHESIS ─▶ JUDGE UI
            evidence       emergent     rank by      novel        human
            (descriptors)  taxonomy     difficulty   scenario     rating
                                        + novelty     spec
```

### Stage 1 — Extraction (`pipeline/modules/extractor`)
A **reason-first, open-vocabulary** annotation pass, in two steps:
1. The vision-language model produces an **unconstrained** free-form analysis of the clip (no schema to fight — which is what made fixed-vocabulary annotation drop its own observations).
2. A structuring pass extracts **typed descriptors** from that text. Each `RawDescriptor` carries:
   - a **typed axis** (the fixed entity boundary: `agents`, `interactions`, `conditions`, `road`, `ego_maneuver`, `weather`, `time`),
   - a **salience** in `[0,1]` — the model's judgment of operational criticality (learned, not a hardcoded list — so it generalizes to unnamed edge cases),
   - a **span pointer** to the sentence that justified it (every structured atom is auditable to its source),
   - a **text embedding** of the descriptor phrase.

Clients (reason / structure / embed) are injected protocols with offline stubs — the reasoning model is one swap away from replacement.

### Stage 2 — Curation (`pipeline/modules/curator`)
Turns evidence into an **emergent, versioned taxonomy** via a pure, deterministic batch recompute (`canonicalize`). Per axis:
- match each descriptor to a carried-forward canonical label (nearest centroid within a threshold),
- cluster the unmatched, and **mint** a new label only past **two guards, both required**: *support* (the cluster recurs ≥ N times) and *cohesion* (the cluster is tight enough to be one concept). Support alone mints noise; cohesion alone mints rare flukes; requiring both rejects both.

Labels are minted *within* an axis only — the entity boundary is structural (weather is never merged into time; rain stays distinct from fog). `project()` re-maps any evidence onto any taxonomy version (a pure function), and `scene_atoms()` derives the per-scene atoms downstream stages consume. Four instrumentation metrics — drift, coverage, stability, re-projection conservation — are logged every run so silent degradation is detectable.

**Firewall:** the curator imports only `pipeline.interfaces` and has **no import path to the discovery stages**. It decides labels purely from evidence + cohesion; it can never see the scores its labels produce, so the system cannot grade its own homework (enforced by `tests/test_firewall.py`).

### Stage 3 — Selection (`pipeline/modules/selection`)
Ranks real scenes for review:

```
score = difficulty · w_difficulty + behavior_novelty · w_novelty      (w_difficulty ≫ w_novelty)
```

- **Difficulty leads.** A model-judged operational-hardness score from an **independent** viewing of the clip (separate from extraction — a confabulation cross-check: salience that disagrees with this independent difficulty is flagged). It is granularity-agnostic: it judges the whole scene, so it captures hard configurations without anyone pre-naming which attributes matter. The difficulty client is an injected seam (swappable for a stack-native reasoning signal).
- **Behavior-novelty refines.** Rarity of a scene's `{interactions, conditions, ego_maneuver}` signature across the corpus — computed over behavior/condition axes **only, never agent attributes**, so incidental attributes (e.g. vehicle colour) cannot drive the ranking.

`ranking.py` (pure), `difficulty.py` (injected client), `synthesis.py`, `config.py`. Firewall-tested like the curator.

### Stage 4 — Synthesis (`pipeline/modules/selection/synthesis.py`)
For each surfaced scene, generates a **novel, simulator-buildable scenario** from its composition's atoms (grounded by, but not copied from, the evidence). The output is a generation spec — a new scene that *embodies* the conditions/behaviors — not a caption of one observed clip. Pure text; no video re-watch.

### Stage 5 — Review (`pipeline/modules/judge_ui`)
A FastAPI server presenting a **blinded** ranked feed: reviewers play the source scene, read the synthesized scenario, and score coherence + usefulness. Ratings persist per-rater for calibration.

---

## 3. Compositional pipeline (alternative substrate)

The repository also contains the original **compositional symbolic** discovery path, useful as a baseline:

```
 window ─▶ ENCODER ─▶ HYPOTHESIZER ─▶ SCORER ─▶ JUDGE UI / EVALUATION
          SchemaRecord  CompositionProposal  ScoredProposal
```

- **Encoder** (`modules/encoder`) — annotates each window into categorical fields → `SchemaRecord`.
- **Hypothesizer** (`modules/hypothesizer`) — finds attribute combinations that are statistically under-represented (compositional novelty: `expected_joint` vs `observed_joint`) → `CompositionProposal`.
- **Scorer** (`modules/scorer`) — plausibility + frontier-difficulty scoring → `ScoredProposal`.
- **Evaluation** (`modules/evaluation`) — seeded recall@K, inter-rater agreement.

This substrate is driven by `python -m pipeline.run {ingest,analyze,report}`. It is symbolic and reproducible, but it ranks *attribute combinations* rather than scenes; the discovery pipeline (§2) selects *real scenes by operational difficulty*, which is the recommended path for surfacing edge cases.

An embedding-clustering + multi-agent-debate substrate also exists, as the `clustering` and `debate` modules. It predates the discovery pipeline above and is retained as a library-level baseline; its original FastAPI runner has been removed, so it is not actively driven and is not part of the active flow.

---

## 4. Module reference

| Module | Role | Key output types |
|---|---|---|
| `storage` | ingest raw driving data (Parquet/TFRecord) → windowed MP4s in a canonical bucket layout; `FlatMP4Storage` for bare-MP4 buckets | `WindowManifest`, `PoseData` |
| `extractor` | reason-first open-vocab annotation → immutable evidence | `RawDescriptor` |
| `curator` | emergent versioned taxonomy + pure re-projection + drift metrics | `Taxonomy`, `CanonicalLabel`, `Projection` |
| `selection` | difficulty + behavior-novelty ranking, difficulty cross-check, scenario synthesis | (rankings, scenario text) |
| `encoder` | fixed-schema annotation (compositional path) | `SchemaRecord` |
| `hypothesizer` | compositional-novelty proposals | `CompositionProposal` |
| `scorer` | plausibility + frontier-difficulty | `ScoredProposal` |
| `judge_ui` | blinded human-rating server | `Rating` |
| `evaluation` | seeded recall, inter-rater agreement | `EvaluationReport` |
| `clustering` | embedding-based scene clustering | — |
| `debate` | multi-agent proposal adjudication | — |
| `dev_dashboard` | operator-facing evaluation surface (gated) | — |

Each module ships its own `README.md`, tests, and offline stubs.

---

## 5. Shared interfaces (`pipeline/interfaces/`)

All cross-module types are frozen dataclasses with `to_json()` / `from_json()` and round-trip tests.

| File | Types |
|---|---|
| `taxonomy.py` | `RawDescriptor`, `CanonicalLabel`, `Taxonomy`, `Projection`, `DEFAULT_AXES` |
| `window.py` | `WindowKey`, `PoseRecord`, `PoseData`, `WindowManifest`, `WindowStorageBase` (Protocol) |
| `schema_record.py` | `SchemaRecord` |
| `proposal.py` | `CompositionProposal`, `ScoredProposal` |
| `rating.py` | `Rating` |
| `report.py` | `EvaluationReport`, `DifferentialExample` |
| `cluster.py`, `debate.py`, `dev_round.py`, `errors.py` | clustering / debate / dev-eval / error types |

---

## 6. Design principles

1. **Lego-block isolation** — modules import only `pipeline/interfaces`; composition roots wire them. Replace any module without touching the others.
2. **Evidence / interpretation / derivation** — immutable evidence at the bottom, recomputable interpretation in the middle, ephemeral derivations on top. Every failure is recoverable from the evidence.
3. **Open vocabulary, emergent structure** — annotation is unconstrained text; structure (the label taxonomy) emerges from the data via the locality rule (repetition + cohesion → a label), within fixed entity axes.
4. **Salience at extraction** — operational relevance is judged where the full scene context exists, as a learned per-descriptor signal, rather than reconstructed downstream from a fixed list.
5. **Difficulty-led selection** — what makes a scene worth surfacing is *how hard it is for the driver*, judged directly; rarity refines but never leads. Novelty is computed over behavior/condition axes only, never incidental attributes.
6. **Independent cross-checks** — the difficulty signal is a separate viewing from the salience signal; disagreement is surfaced, not hidden.
7. **Instrumented & firewalled** — drift/coverage/stability/conservation metrics every run; the interpretation layer cannot see the discovery scores it feeds.

---

## 7. Where the authoritative contracts live

Serialization contracts are not duplicated in prose — they live in code and are pinned by tests:

- **Field-level schema** — the frozen dataclasses in `pipeline/interfaces/*.py` and their `to_json()` / `from_json()`. These are the single source of truth for every cross-module type.
- **Round-trip guarantees** — `pipeline/interfaces/tests/` pins each type's serialization shape.
- **Per-module behavior** — each module's own `README.md` (e.g. `modules/curator/README.md`, `modules/selection/README.md`, `modules/storage/README.md`) documents its bucket layout, semantics, and failure modes.
- **Storage bucket layout** — `modules/storage/ingestion.py` (canonical `windows/{segment_id}/{idx:04d}/…` layout) and `modules/storage/flat_mp4.py` (bare-MP4 path).

To regenerate a type's exact JSON shape, read its dataclass + the matching round-trip test rather than relying on documentation drift.
