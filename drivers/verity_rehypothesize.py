"""Re-run the original 202 through the hypothesizer -> judge, with the EVOLVED labels.

The question: do the blossomed canonical labels (101 from ~1000 segments) change
the compositional-novelty scenarios on the original 202 vs the raw/sparse
annotations? Key metric: the GHOST RATE (proposals with observed_joint==0 — combos
that never actually co-occur). Richer/consolidated labels should produce fewer
ghosts and more grounded, distinct scenarios.

Projects the 202 onto OLD (202-only) vs NEW (all-evidence) taxonomies, runs the
hypothesizer on each, compares, then scores the NEW proposals into the judge feed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipeline.interfaces.taxonomy import EMPTY_TAXONOMY
from pipeline.interfaces.window import WindowKey
from pipeline.modules.curator import (
    CuratorConfig, TaxonomyStore, canonicalize, project, scene_atoms,
)
from pipeline.modules.hypothesizer.composition import build_proposals
from pipeline.modules.hypothesizer.config import HypothesizerConfig
from pipeline.modules.hypothesizer.frequency import compute_frequencies

OUT = Path("outputs/waymo")
STORE = OUT / "taxonomy_store_salience"
CFG = CuratorConfig(cohesion_threshold=0.36, merge_threshold=0.20, support_threshold=2)
NOVELTY_AXES = {"interactions", "agents", "conditions", "ego_maneuver"}


def _scorer(stub: bool):
    from pipeline.modules.scorer import (
        NIMTextClient, Scorer, StubDifficultyClient, StubPlausibilityClient,
    )
    if stub:
        return Scorer(StubPlausibilityClient(), StubDifficultyClient())
    c = NIMTextClient()
    return Scorer(plausibility_client=c, difficulty_client=c)


def _hypothesize(descriptors, taxonomy):
    """Compositional-novelty proposals over the dynamic-axis canonical atoms."""
    proj = project(descriptors, taxonomy, CFG)
    atoms = scene_atoms(descriptors, proj, taxonomy)
    atom_sets, keys = [], []
    for sc, a in atoms.items():
        dyn = {x for x in a if x.split(":", 1)[0] in NOVELTY_AXES}
        if dyn:
            atom_sets.append(frozenset(dyn)); keys.append(WindowKey(sc, 0))
    if not atom_sets:
        return []
    marg, pair = compute_frequencies(atom_sets)
    return build_proposals(atom_sets=atom_sets, keys=keys, marginal=marg,
                           pairwise=pair, config=HypothesizerConfig(), arm="reasoning")


def _stats(props):
    ghosts = sum(1 for p in props if p.observed_joint == 0)
    backed = len(props) - ghosts
    return len(props), ghosts, backed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stub", action="store_true")
    args = ap.parse_args()

    store = TaxonomyStore(STORE)
    before = json.loads((OUT / "scale_before.json").read_text())
    bset = set(before["scenes"])
    allds = store.load_descriptors()
    ds202 = [d for d in allds if d.scene_id in bset]
    print(f"[rehypo] 202 set: {len(ds202)} descriptors over {len(bset)} scenes", file=sys.stderr)

    old_tax = canonicalize(ds202, EMPTY_TAXONOMY, CFG)      # labels from 202 only
    new_tax = canonicalize(allds, EMPTY_TAXONOMY, CFG)      # labels evolved over ~1000

    old_props = _hypothesize(ds202, old_tax)          # 202 labels, 202 frequencies
    new_props = _hypothesize(ds202, new_tax)          # evolved labels, 202 frequencies
    corpus_props = _hypothesize(allds, new_tax)       # evolved labels, FULL 983 frequencies

    on, og, ob = _stats(old_props)
    nn, ng, nb = _stats(new_props)
    cn, cg, cb = _stats(corpus_props)
    print("\n=== HYPOTHESIZER: effect of better labels AND more data ===")
    print(f"  OLD labels / 202 freq :  {on} proposals | ghosts: {og} | BACKED: {ob}")
    print(f"  NEW labels / 202 freq :  {nn} proposals | ghosts: {ng} | BACKED: {nb}")
    print(f"  NEW labels / 983 freq :  {cn} proposals | ghosts: {cg} | BACKED: {cb}")
    print("  sample BACKED proposals (real co-occurrence — not ghosts):")
    for p in sorted([p for p in corpus_props if p.observed_joint > 0],
                    key=lambda p: -p.novelty_score)[:8]:
        cons = ", ".join(c.split(":", 1)[1] for c in p.constituents)
        print(f"     obs={p.observed_joint:.3f} N={p.novelty_score:.2f} scenes={len(p.motivating_scene_ids)}  {cons[:55]}")

    # feed the BACKED corpus proposals (real, with motivating scenes) to the judge
    backed = [p for p in corpus_props if p.observed_joint > 0]
    scored = _scorer(args.stub).score_batch(backed) if backed else []
    (OUT / "judge_scored.json").write_text(json.dumps([s.to_json() for s in scored], indent=2))
    (OUT / "rehypothesize_report.json").write_text(json.dumps({
        "old_labels_202_freq": {"proposals": on, "ghosts": og, "backed": ob, "labels": len(old_tax.labels)},
        "new_labels_202_freq": {"proposals": nn, "ghosts": ng, "backed": nb, "labels": len(new_tax.labels)},
        "new_labels_983_freq": {"proposals": cn, "ghosts": cg, "backed": cb},
    }, indent=2))
    n_acc = sum(1 for s in scored if s.accepted)
    print(f"\n[rehypo] DONE — {len(scored)} BACKED proposals scored ({n_acc} accepted) -> judge feed")


if __name__ == "__main__":
    main()
