# Module: Curator — Emergent Taxonomy

Turns immutable scene **evidence** into a versioned, self-refining **taxonomy** of
labels, then projects evidence onto a version to produce the **atoms** the
Hypothesizer reasons over. The locality rule (repetition → label) and the typed
axes (entity boundaries) are the idea; this module makes them *honest*.

## The three objects (never conflate them)

| Object | Lives in | Nature | Rule |
|---|---|---|---|
| `RawDescriptor` | `interfaces/taxonomy.py` | **Evidence** | Immutable, append-only. Carries a `reasoning_span` back to the sentence that justified it. |
| `CanonicalLabel` / `Taxonomy` | `interfaces/taxonomy.py` | **Interpretation** | Mutable, **versioned**. Persists across runs; only refines. |
| `Projection` | `interfaces/taxonomy.py` | **Derivation** | Ephemeral. Recomputed from (evidence, taxonomy); never stored as truth. |

**Spine:** evidence is append-only at the bottom, interpretation is
deterministic-recomputable in the middle, derivations are ephemeral on top. Every
failure mode is recoverable because you can always re-derive labels and atoms from
the immutable evidence.

## Pipeline

```
RawDescriptor[]  ──canonicalize()──▶  Taxonomy(v+1)  ──project()──▶  Projection
   (evidence)        (interpretation)                  (derivation)
                                          │
                                  scene_atoms() ──▶ {scene: {axis:label}}  → Hypothesizer
```

- **`canonicalize(descriptors, base_taxonomy, config) -> Taxonomy`** — pure,
  deterministic batch recompute. Same `(descriptors, base, seed)` → same taxonomy.
  Per axis: match to carried-forward labels → cluster the unmatched → **mint** only
  what clears *both* guards → merge near-duplicates. Because it's a batch recompute,
  it is its own correctness oracle for any future online path.
- **`project(descriptors, taxonomy, config) -> Projection`** — pure descriptor→label
  assignment under one version; unmatched = orphan (surfaced, never dropped).
  Re-project *all* historical evidence on every taxonomy change so old data stays
  comparable and gets richer retroactively.
- **`scene_atoms(...)`** — the ephemeral composition input for the Hypothesizer.

## The mint guard (both required)

A cluster becomes a label only when it clears **support** (`recurs ≥ N`) **and**
**cohesion** (`cluster radius ≤ r`, i.e. it is *one* concept). Support alone mints
noise-labels from frequent phrasings; cohesion alone mints tight-but-rare flukes.
Requiring both rejects both. Axes are never merged across (weather ≠ time);
labels evolve only *within* an axis (rain ≠ fog).

## The firewall (structural, not disciplinary)

The curator imports **only** `pipeline.interfaces`. It has **no import path** to the
Hypothesizer and can never see novelty scores — labels are decided purely from
evidence + cohesion, so the system cannot grade its own homework. Enforced by
`tests/test_firewall.py` (no hypothesizer import; no reach into other modules).

## Instrumentation (log every run)

`metrics.py` — `drift` (minted/dropped/carried), `coverage` (assigned vs orphaned),
`stability` (re-canonicalize under two seeds; agreement of the co-assignment
relation — the best early warning that thresholds are unstable), and
`reprojection_sanity` (count + scene-id conservation).

## Persistence

`TaxonomyStore` — `evidence.jsonl` (append-only, idempotent by content id),
`taxonomy/v{N}.json` (immutable versions), `projections/v{N}.json` (cache,
reconstructible).

## Scoped now / deferred

**Built:** reason-agnostic core — interfaces, deterministic batch canonicalization
with the dual guard, pure re-projection, the four metrics, versioned store, tests.
Seed the taxonomy with the current vocab as coarse initial labels; it blossoms with
volume.

**Deferred (seams already in place):** online incremental canonicalization (batch
is the oracle), correlation-gain splitting (needs volume; `parent_id` reserved for
the general→local hierarchy), and the upstream **reason-first extractor** that emits
typed descriptors + span pointers + embeddings (the `RawDescriptor` producer).

## Tests

```bash
python -m pytest pipeline/modules/curator/tests/ -v
```

## Lego-block rule

Curator imports only `pipeline/interfaces/`. The Hypothesizer imports the curator's
output. There is no path back.
