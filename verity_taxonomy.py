"""End-to-end emergent-taxonomy judge pipeline (composition root).

    extract -> append evidence -> canonicalize -> project -> scene_atoms
            -> hypothesize (compositional novelty) -> score -> judge_scored.json

This is the ONE place that wires the firewalled pieces together: the extractor
produces immutable evidence, the curator turns it into a versioned taxonomy
(blind to discovery), and the hypothesizer/scorer consume the derived atoms.
Run --stub for an offline wiring check; drop --stub to run on real clips.

    set -a && . ./.env && set +a
    .venv/bin/python verity_taxonomy.py --stub                 # offline validation
    .venv/bin/python verity_taxonomy.py --limit 30             # real clips
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipeline.interfaces.taxonomy import EMPTY_TAXONOMY
from pipeline.interfaces.window import WindowKey
from pipeline.modules.curator import (
    TaxonomyStore, canonicalize, coverage, drift_metrics, project, scene_atoms, stability,
    CuratorConfig,
)
from pipeline.modules.extractor import (
    Extractor, ExtractorConfig,
    CosmosReasonClient, NIMStructureClient, NIMEmbedder,
    StubReasonClient, StubStructureClient, StubEmbedder,
)
# composition root may import modules — the FIREWALL is that the curator cannot.
from pipeline.modules.hypothesizer.composition import build_proposals
from pipeline.modules.hypothesizer.config import HypothesizerConfig
from pipeline.modules.hypothesizer.frequency import compute_frequencies

OUT = Path("outputs/waymo")
STORE = OUT / "taxonomy_store"

# Tuned canonicalization: collapse phrasings but keep distinct concepts apart.
CURATOR_CFG = CuratorConfig(cohesion_threshold=0.36, merge_threshold=0.20, support_threshold=2)

# Novelty is computed ONLY over the dynamic axes — what actually makes a scene
# rare. Weather / time / road are context, not novelty drivers, so a mundane
# "clear day on a multi-lane road" can never be surfaced as a proposal.
NOVELTY_AXES = {"interactions", "agents", "conditions", "ego_maneuver"}


def _build_extractor(stub: bool) -> Extractor:
    if stub:
        return Extractor(StubReasonClient(), StubStructureClient(), StubEmbedder(dim=32),
                         ExtractorConfig())
    return Extractor(CosmosReasonClient(), NIMStructureClient(), NIMEmbedder(),
                     ExtractorConfig())


def _build_scorer(stub: bool):
    from pipeline.modules.scorer import (
        NIMTextClient, Scorer, StubDifficultyClient, StubPlausibilityClient,
    )
    if stub:
        return Scorer(StubPlausibilityClient(), StubDifficultyClient())
    c = NIMTextClient()
    return Scorer(plausibility_client=c, difficulty_client=c)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20, help="0 or negative = all segments")
    ap.add_argument("--camera", default="FRONT")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stub", action="store_true", help="offline: stub clients, no GPU/network")
    ap.add_argument("--out", default=str(OUT / "judge_scored.json"))
    args = ap.parse_args()

    store = TaxonomyStore(STORE)
    extractor = _build_extractor(args.stub)
    # Skip clips that already produced evidence (append-only + idempotent): a
    # re-run only re-processes previously-FAILED or new clips, not the whole set.
    already = {d.scene_id for d in store.load_descriptors()}

    # --- Stage 1: extract -> immutable evidence (append-only) -----------------
    if args.stub:
        scenes = [(f"stub-scene-{i}", f"ref-{i}") for i in range(max(1, args.limit))]
    else:
        from verity_run import list_segments  # reuse the nested-layout lister
        limit = args.limit if args.limit and args.limit > 0 else 10_000
        scenes = [(seg, f"gs://nvidia-adr-waymo-segment-videos/segments/{seg}/{seg}_{args.camera}.mp4")
                  for seg, _ in list_segments(limit, args.camera)]

    pending = [(sid, ref) for sid, ref in scenes if sid not in already]
    print(f"[tax] {len(scenes)} scenes; {len(scenes) - len(pending)} already have evidence, "
          f"{len(pending)} to extract", file=sys.stderr)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    new_descs = []
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = {pool.submit(extractor.extract, sid, ref): sid for sid, ref in pending}
        for f in as_completed(futs):
            sid = futs[f]; done += 1
            try:
                ds = f.result(); new_descs.extend(ds)
                print(f"[tax] {done}/{len(pending)} {sid[:24]} -> {len(ds)} descriptors", file=sys.stderr)
            except Exception as exc:  # noqa: BLE001
                print(f"[tax] {done}/{len(pending)} {sid[:24]} FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
    added = store.append_descriptors(new_descs)
    descriptors = store.load_descriptors()
    print(f"[tax] appended {added} new; {len(descriptors)} total descriptors in evidence", file=sys.stderr)

    # --- Stage 2: canonicalize (evidence -> versioned taxonomy) ---------------
    base = store.load_taxonomy() or EMPTY_TAXONOMY
    cfg = CURATOR_CFG
    taxonomy = canonicalize(descriptors, base, cfg)
    store.save_taxonomy(taxonomy)
    proj = project(descriptors, taxonomy, cfg)
    store.save_projection(proj)

    # --- instrumentation (log every run) --------------------------------------
    drift = drift_metrics(base, taxonomy)
    cov = coverage(descriptors, proj)
    stab = stability(descriptors, base, cfg) if len(descriptors) <= 400 else {"stability": -1}
    print(f"[tax] taxonomy v{taxonomy.version}: {len(taxonomy.labels)} labels "
          f"(minted {drift['minted']}, dropped {drift['dropped']}) | "
          f"coverage {cov['coverage']:.0%} | stability {stab['stability']:.2f}", file=sys.stderr)

    # --- Stage 3: derive atoms + hypothesize (compositional novelty) ----------
    atoms = scene_atoms(descriptors, proj, taxonomy)
    atom_sets, keys = [], []
    for scene, a in atoms.items():
        # novelty over the dynamic axes only (weather/time/road are context)
        dyn = {atom for atom in a if atom.split(":", 1)[0] in NOVELTY_AXES}
        if dyn:
            atom_sets.append(frozenset(dyn)); keys.append(WindowKey(scene, 0))
    proposals = []
    if atom_sets:
        marg, pair = compute_frequencies(atom_sets)
        proposals = build_proposals(atom_sets=atom_sets, keys=keys, marginal=marg,
                                    pairwise=pair, config=HypothesizerConfig(), arm="reasoning")
    print(f"[tax] {len(proposals)} proposals over {len(atom_sets)} scenes", file=sys.stderr)

    # --- Stage 4: score -> Judge feed -----------------------------------------
    scored = _build_scorer(args.stub).score_batch(proposals) if proposals else []
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps([s.to_json() for s in scored], indent=2))

    # --- Engineer context: pair each proposal with its motivating scenes' rich
    # evidence (descriptor text + the reasoning span that justified it). Gives an
    # engineer the "why" behind a proposal and feeds detailed scene generation.
    by_scene: dict[str, list[dict]] = {}
    for d in descriptors:
        by_scene.setdefault(d.scene_id, []).append(
            {"axis": d.axis, "text": d.text, "span": d.reasoning_span})
    context = []
    for s in scored:
        sj = s.to_json()
        scenes = []
        for wk in sj.get("motivating_scene_ids", [])[:5]:
            sid = wk.get("segment_id") if isinstance(wk, dict) else str(wk)
            scenes.append({"scene_id": sid, "evidence": by_scene.get(sid, [])})
        context.append({
            "composition_id": sj["composition_id"],
            "constituents": sj["constituents"],
            "novelty_score": sj["novelty_score"],
            "final_rank_score": sj["final_rank_score"],
            "motivating_scenes": scenes,
        })
    (OUT / "judge_context.json").write_text(json.dumps(context, indent=2))

    n_acc = sum(1 for s in scored if s.accepted)
    print(f"\n[tax] DONE — v{taxonomy.version} taxonomy, {len(taxonomy.labels)} labels, "
          f"{len(scored)} scored ({n_acc} accepted) -> {args.out}")
    # show a few emergent labels per axis
    by_axis = taxonomy.labels_by_axis()
    for axis in sorted(by_axis):
        names = sorted(l.name for l in by_axis[axis] if l.support > 0)[:6]
        if names:
            print(f"   {axis}: {', '.join(names)}")


if __name__ == "__main__":
    main()
