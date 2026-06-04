# Architecture Proposal — Preventing Detail Collapse in Verity

**Status:** proposal / RFC. Not implemented. Written after a full first E2E run (16 Waymo segments → 48 windows → 27 accepted proposals) exposed where and why scene detail is lost.

**The question that prompted this:** *"The videos are being compressed too fast/too early into the structured JSON. How do we prevent the collapse of detail while staying deterministic?"*

This document argues that the honest answer is bigger than the question, and tries to separate (a) the small, correct fix you already identified, from (b) the structural change that actually matters, and (c) the thing the current design is missing entirely.

---

## 1. Where the collapse actually happens

The encoder VLM already produces rich output. On a busy window it returns, unprompted:

```json
"agents": [
  {"type":"pedestrian","description":"a person with a backpack crossing left-to-right at a marked crosswalk"},
  {"type":"construction_worker","description":"in high-vis near the work zone on the right"}
]
```

Then the **parse-to-contract step deletes all of it** except 8 categorical fields drawn from a locked vocabulary, so the whole window becomes:

```json
{"agents":["pedestrian","construction_worker"], "weather":"clear", "lighting":"well_lit",
 "road":{"geometry":"straight"}, "traffic_control":"traffic_light", "ego_task":"cruising"}
```

Everything else — descriptions, positions, counts, dynamics, the raw text — is discarded at step one, before the Hypothesizer / Scorer / UI ever see it. **The compression isn't the model's; it's the contract's,** and it's irreversible and immediate.

So the literal answer to the question is easy (Section 2). But it doesn't fix the real problem (Section 3).

---

## 2. The fix you already identified — necessary, not sufficient

**Retain, don't reconstruct.** Determinism only breaks when you *re-infer* detail (the LLM-expansion path that hallucinated "overcast"). If you instead **keep the model's original output verbatim**, it's deterministic by construction.

Concretely: split each record into two channels —

- **Statistical channel** (`fields`): the categorical tags. Required to be discrete so the Hypothesizer can *count* and compute frequency/novelty. Unchanged.
- **Evidence channel** (`evidence`): the model's free-text `scene_description`, the full `agents` objects, the raw response. Carried through untouched to the Scorer / Judge UI / `scored.json`.

The Hypothesizer reads only `fields`, so statistics stay identical and reproducible; the evidence rides alongside for humans and for scene generation. This also recovers the windows currently dropped for "agents-as-objects" (coerce `{type:X}`→`X` for the tag, keep the object in evidence).

**But here is the uncomfortable part:** preserving evidence changes *what you display*, not *what you discover*. The ranking still only sees 8 tags. You will get prettier evidence attached to the **same** limited — and sometimes nonsensical (`dim + day + clear`) — discoveries. **A richer JSON is still a symbolic collapse; the only question is how many symbols.** That is not the lever that matters.

---

## 3. What you're actually missing — the representation is the ceiling

The deeper issue, in the framing of the two people you asked me to channel:

**(Karpathy — it's an information-theoretic bottleneck.)** You are compressing a video — millions of pixels × time — into ~8 discrete symbols and then *searching for novelty in the symbol space*. The novel, safety-relevant content of driving lives in the **configuration and dynamics** of agents (a cyclist drifting into the lane as a door opens; a pedestrian occluded by a turning truck) — exactly the structure a fixed 8-field vocabulary cannot represent. The method can therefore only ever find novelty **the vocabulary was pre-designed to express.** The vocabulary is a hard ceiling on discoverable novelty, and you set it before seeing the data.

**(LeCun — a hand-designed symbolic prior is the wrong substrate for discovering the unknown.)** The product's premise is "find the edge cases you *haven't* thought to test." But a locked vocabulary requires you to **enumerate, in advance, the attribute axes along which edge cases can occur.** That is self-defeating: the unknown unknowns are precisely the ones not in your schema. The right primitive for "what's underrepresented" is **density in a learned, continuous representation**, not combinatorics over hand-chosen categories. Let the representation be learned; describe it symbolically *after*.

**Three more concrete weaknesses that follow from the same root:**

1. **Novelty ≠ risk, and the independence baseline is naive.** `novelty = ln(expected/observed)` under an independence assumption over marginals. Driving attributes are heavily *correlated* (night ⇒ dim; rain ⇒ overcast), so "rarer than independence predicts" mostly surfaces (a) **correlations the model treats as surprises** and (b) **contradictions** (`dim+day+clear` never co-occurs because it's contradictory, not because it's an untested gap). At small N it degenerates further: every accepted proposal here has `observed_joint = 0`, so novelty just re-ranks by marginal commonness. Statistical rarity in *your sample* is not the quantity you care about — **coverage risk for the AV stack** is.

2. **Plausibility is asking a tag-only LLM to be a physics engine.** Handing a 7B text model four tags and asking "can these co-occur?" with no grounding forces it to rationalize (it invented "overcast" to reconcile `dim`+`day`). Plausibility should come from **the data manifold** (on-manifold ⇒ physically realizable) and/or an explicit world/physics prior — not a lenient symbolic judge.

3. **Difficulty is a proxy of a proxy.** "Frontier difficulty" is estimated from reasoning/action variance of a text model over tags. The thing you actually want is **where your AV stack (or a target model) has high error/uncertainty.** That signal exists; it's just not wired in.

---

## 4. The proposed architecture — discover in continuous space, describe in symbolic space

Invert the current flow. Today: *discover in symbol space (collapses detail, capped by vocab) → describe.* Instead:

```
        ┌─────────────────────────────────────────────────────────────┐
        │ 1. REPRESENTATION (learned, continuous, deterministic)        │
        │    Embed each window with a strong video encoder              │
        │    (Cosmos-Embed1 — already in this repo — / VideoMAE /       │
        │    VLM hidden states). Detail lives here, losslessly-ish.     │
        └───────────────────────────┬─────────────────────────────────┘
                                     ▼
        ┌─────────────────────────────────────────────────────────────┐
        │ 2. DISCOVERY (density in embedding space)                     │
        │    Low density  = underrepresented (the real "gap" signal)    │
        │    On-manifold   = plausible  (no separate lenient LLM gate)  │
        │    × model-error = hard/frontier (cross with AV-stack loss)   │
        │    (UMAP + HDBSCAN are already in requirements.txt.)          │
        └───────────────────────────┬─────────────────────────────────┘
                                     ▼
        ┌─────────────────────────────────────────────────────────────┐
        │ 3. DESCRIPTION (post-hoc, grounded, symbolic)                 │
        │    For each discovered region, summarize its real member      │
        │    clips with the VLM → categorical tags + free text.         │
        │    The vocabulary becomes a LABEL for communication,          │
        │    not the substrate for discovery.                           │
        └───────────────────────────────────────────────────────────────┘
```

Why this is strictly better on the axes you care about:

- **Detail never collapses for discovery.** It collapses only at the *describe* step, after the gap is already found — and there it's grounded in real member clips, so the description is what was actually seen, deterministic given a fixed model + seed.
- **Plausibility is intrinsic.** Off-manifold = implausible; you stop relying on a 7B model to adjudicate contradictions.
- **Discovers unknown unknowns.** Density doesn't need the gap's category pre-enumerated.
- **Determinism is preserved throughout** (fixed encoder, fixed-seed UMAP/HDBSCAN, temp-0 description). Determinism was never the real constraint — *symbolic-vs-continuous* was. Don't optimize the wrong axis.

And it doesn't throw away the existing system: **the categorical pipeline becomes the *description + stratification + interpretability* layer** (it's auditable, communicable, and gives the Judge UI something concrete), while *discovery* moves to the manifold. Additive, not a teardown — and the repo's **Dev Dashboard discrimination test** is already the right harness to prove the new path beats the old (extend "Verity vs Random vs Naive-rare" with "embedding-density vs tag-novelty").

---

## 5. The hard tension nobody has named (the most important paragraph)

**"Underrepresented" and "plausible" are partially contradictory, and the current design hides this.**

The scenarios you most want are rare *in your data* — i.e. **off your data manifold by definition.** But a data-only manifold certifies plausibility by *on-manifold-ness*. So the rarest real scenarios are exactly the ones a data-derived plausibility prior will reject. You cannot get both "maximally novel" and "data-says-plausible" from the same distribution.

This is why a plausibility judge external to the data exists at all (the tag-LLM is a weak version of it). The honest resolution:

- Use **density for rarity** (where is my data thin?), and
- Use an **external plausibility prior for the tail** — a physics/kinematics check, a much broader world model, or a human — *not* the data manifold and *not* a lenient 7B rationalizer.

Naming this tension changes the eval too: rarity-in-sample is measurable, but **"is this an underrepresented *safety* gap" has no ground truth here.** The discrimination test measures human-rated rarity/relevance — a proxy. The real validation loop is missing: *does targeting these scenarios in sim/collection actually reduce downstream AV failures?* Until that loop exists, every novelty number is unanchored.

---

## 6. Pragmatic staging (make it work → measure → iterate)

- **v1 — ship now (this version).** Symbolic pipeline + **Section 2 evidence-retention**. Honest, working, interpretable, fully deterministic. Good baseline; fixes the agents-drop and gives grounded human-readable scenarios.
- **v2 — add the continuous channel.** Turn the **visual arm back on** (Cosmos-Embed1 is already wired), run **density-based discovery** in parallel, and use the **discrimination test** to measure whether embedding-density beats tag-novelty. Keep both; let data decide.
- **v3 — ground difficulty + close the loop.** Cross density with **target-model error/uncertainty** for true "rare AND hard," and stand up the **sim/collection → re-measure-failures** loop so novelty is anchored to outcomes.
- **External plausibility prior** (Section 5) lands whenever the tail starts mattering — a kinematic/physics sanity check is the cheapest first version.

---

## 7. TL;DR

- The detail collapse is real but it's a **contract choice**; fix it by **retaining** the model's output in an `evidence` channel (deterministic — you store, never re-infer).
- That fix improves *display*, not *discovery*. The thing you're missing: **you're discovering novelty inside a hand-designed symbol space, which caps novelty at what you pre-enumerated and conflates rarity with contradiction.**
- The right structure is **discover in a learned continuous representation (density), describe in symbols** — keeping the current pipeline as the description/interpretability layer.
- The deepest issue is a genuine tension: **the rarest real scenarios are off your data manifold, so a data-only plausibility prior can't certify them** — you need an external (physics/world-model/human) prior, and a validation loop tied to actual AV failures, before the novelty score means anything.

---

## 8. Implementation status — clustering merged as a module (2026-06-04)

The embedding arm is no longer a parallel `waymo_pipeline/` universe; its core is now a **first-class lego module** inside `pipeline/`, sharing Module 1's ingestion.

### ✅ Done (built + verified on this instance)
- **`pipeline/interfaces/cluster.py`** — `WindowEmbedding`, `ClusterAssignment`, `ClusterReport` (dataclass + `to_json`/`from_json`), pinned by a round-trip test. Same contract style as every other interface.
- **`pipeline/modules/clustering/`** — Module 8:
  - `config.py`: `EmbedClient` Protocol + `ClustererConfig(.from_env)` (UMAP/HDBSCAN/GLOSH knobs from `.env`) + errors.
  - `embed.py`: `NIMEmbedClient` (Cosmos-Embed, ported request) + `StubEmbedClient` (offline/CI).
  - `clusterer.py`: `Clusterer` — **reads windows via the `WindowStorageBase` Protocol** (the exact surface the encoder uses → shares Module 1 ingestion, zero coupling), runs the ported L2→UMAP(cosine)→HDBSCAN(GLOSH)→3D-viz math, returns `ClusterReport`. Degrades gracefully below 5 windows.
- **`pipeline/run.py cluster`** subcommand — peer to `analyze`; `ingest` → `analyze` (Judge) **and/or** `cluster`, off one ingest.
- **Verified:** 6 module tests + 36 interface round-trips pass; lego rule clean (imports only `pipeline.interfaces` + own package); all four subcommands parse (Judge path untouched); `cluster --stub` ran over the **same 48 ingested windows** the Judge run used → `clusters.json` (round-trips).

### ⏳ Remaining — phased retirement of `waymo_pipeline/`
The directory can't be deleted wholesale yet: it still hosts the `:8000` API (Ingest/Cluster/Analysis tabs) **and** the deferred `debate`/Analysis path. Phases:

1. **Unify ingest on canonical windows.** Point the batch at `pipeline.run ingest` (writes `windows/{seg}/{idx}/…`) so Judge *and* Cluster read the same artifact. Retires `waymo_video_pipeline.py` + `waymo_extract_scene_windows.py` (segment-MP4 + `SceneWindow` layout). *(NIM-free; instance-agnostic.)*
2. **Repoint the `:8000` API.** `waymo_runner` batch → `pipeline.run ingest` + `pipeline.run cluster`; adapt `/cluster-space` to read `clusters.json` (`ClusterReport` → `ClusterPoint[]`). Then **delete** `waymo_embed_scenes.py` + `waymo_cluster_embeddings.py` (superseded by Module 8).
3. **Point the Cluster Space tab** at the new output (proxy/route already single-origin).
4. **Real embeddings** — the only step needing a model: re-pull `cosmos-embed1` *here* (fits the L40S; frees by dropping unused `reason2-8b`) **or** use the unified **Cosmos-3** container on the CUDA-13 box. *Nothing else needs the new box — earlier "finish on Cosmos-3" was an overstatement; only meaningful-cluster verification needs an embed NIM.*
5. **Debate / Analysis** (`debate_actors`, `react_loop`, `proposal_builder`, `waymo_describe_and_debate`, `waymo_populate_pgvector`, `debate_*`) — out of scope now; later becomes its own module the same way. Until then `waymo_runner.py` + `store.py` stay.

**Net:** the algorithmic merge (the hard, design-bearing part) is done and lego-clean; what remains is mechanical repointing + one model-dependent verification, all instance-agnostic except the embed run itself.

### Progress update (2026-06-04, later) — HTTP layer de-duplicated
- ✅ **Phase 1 (unify ingest) + Phase 2 (repoint `:8000` batch):** `waymo_runner._run_batch_pipeline` now runs **`pipeline.run ingest`** (canonical Module 1) + **`pipeline.run cluster`** (Module 8) instead of its 4 bespoke scripts — *no separate ingestion universe*. **Verified:** a 2-segment batch ingested into the canonical `verity/windows/{seg}/` layout (the same windows Judge reads).
- ✅ **Ingestion is an isolated checkpoint:** `ingested=True` persists even when the cluster stage fails (verified live).
- ✅ **Per-subprocess credentials:** ingest = user ADC (reads Waymo source); cluster = signer key (reads output + signs).
- ✅ **Honest failure:** all-embeds-failed now raises (`embedded 0/N windows…`) → batch `failed` with a clear reason, instead of a hollow `completed` empty report. Locked by a unit test.
- ✅ **Phase 3 (`/cluster-space`)** reads the module's `clusters.json` (`ClusterReport`).
- ⏳ **Script deletion blocked:** `smoke_test.py` still invokes `waymo_cluster_embeddings`, and `waymo_extract_scene_windows`/`scene_window.py` import `waymo_video_pipeline`. Repoint `smoke_test.py` → `pipeline.run cluster`, then delete the 4 deprecated scripts.
- ⏳ **Judge-as-batch-selectable:** the batch currently runs ingest→cluster; wiring "Run Judge on this batch" (`pipeline.run analyze` → `scored.json` → judge_ui) is the remaining "pick judge **or** cluster" piece.
- ⏳ **Real embeddings:** fix `COSMOS_EMBED1_URL` (it points at `:8000`, the API, not an embed NIM) + a running embed NIM (`embed1` here or Cosmos-3 unified).
