"""Hard + novel selection over the full 983-segment corpus.

The objective we converged on:
  * DIFFICULTY leads (model-judged operational hardness — granularity-agnostic;
    it settles "does this configuration matter" without us pre-deciding label
    granularity). Swappable client -> Alpamayo's reasoning later.
  * BEHAVIOR-NOVELTY refines (rarity of the scene's {interactions, conditions,
    ego_maneuver} signature across the corpus). Computed over behaviors/conditions
    ONLY — never agent attributes, so car colour can't drive novelty.
  * score = difficulty*W_DIFF + behavior_novelty*W_NOV   (difficulty heavy)

Cost-aware: cheap signals (max salience proxy + behavior-novelty) pre-rank all
983; the expensive VLM difficulty + vivid description run only on the top
candidates; final rank uses real difficulty. Writes the Judge feed directly.

    set -a && . ./.env && set +a
    .venv/bin/python verity_hardnovel.py --candidates 40 --topk 20
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pipeline.interfaces.taxonomy import EMPTY_TAXONOMY
from pipeline.modules.curator import CuratorConfig, TaxonomyStore, canonicalize, project
from pipeline.modules.extractor import CosmosReasonClient
from scenario_synth import synthesize_scenario

OUT = Path("outputs/waymo")
STORE = OUT / "taxonomy_store_salience"
BUCKET = "gs://nvidia-adr-waymo-segment-videos"
CFG = CuratorConfig(cohesion_threshold=0.36, merge_threshold=0.20, support_threshold=2)
BEHAVIOR_AXES = {"interactions", "conditions", "ego_maneuver"}   # NOT agents (granular)
W_DIFF, W_NOV = 0.7, 0.3

_DIFF_PROMPT = (
    "You are an AV safety analyst. On a scale of 0.0 to 1.0, how operationally "
    "DIFFICULT is this clip for an automated driver? 0.0-0.2 routine; 0.3-0.6 "
    "moderate (normal crossing, moderate traffic, reduced visibility from rain/fog/"
    "night/wet road); 0.7-1.0 genuinely hard (occluded or suddenly-emerging road "
    "user, conflict / near-miss, ambiguous right-of-way). Most clips are routine — "
    'be honest. Answer ONLY: {"difficulty": <0..1>, "reason": "<one short reason>"}'
)
_DESC_PROMPT = (
    "Write a COMPLETE, vivid scene specification for an AV simulator — detailed "
    "enough to RECREATE this scene: road type and lane layout, intersections/"
    "signals, weather, lighting, time of day, surrounding setting, and every notable "
    "agent (vehicles, pedestrians, cyclists) with position, motion, and intent. "
    "End with one sentence on what is operationally notable. 5-8 concrete sentences; "
    "describe only what is visible."
)


def _ref(sid: str) -> str:
    return f"{BUCKET}/segments/{sid}/{sid}_FRONT.mp4"


def _difficulty(reason: CosmosReasonClient, ref: str) -> tuple[float, str]:
    try:
        raw = reason.describe(ref, _DIFF_PROMPT)
        m = re.search(r"\{[\s\S]*\}", raw)
        obj = json.loads(m.group(0)) if m else {}
        return max(0.0, min(1.0, float(obj.get("difficulty", 0.0)))), str(obj.get("reason", "")).strip()
    except Exception as exc:  # noqa: BLE001
        return -1.0, f"difficulty check failed: {exc}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=int, default=40, help="VLM-verify this many")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    store = TaxonomyStore(STORE)
    ds = store.load_descriptors()
    tax = canonicalize(ds, EMPTY_TAXONOMY, CFG)
    name = {l.label_id: f"{l.axis}:{l.name}" for l in tax.labels}
    asg = project(ds, tax, CFG).as_dict()
    scenes = sorted({d.scene_id for d in ds})
    print(f"[hn] {len(ds)} descriptors over {len(scenes)} scenes; {len(tax.labels)} labels", file=sys.stderr)

    # per-scene: max salience (difficulty proxy), behavior-atom set, salient evidence
    by_scene = defaultdict(list)
    for d in ds:
        by_scene[d.scene_id].append(d)
    behav = {}            # scene -> set of behavior atoms (canonical)
    maxsal = {}
    for sc, dd in by_scene.items():
        maxsal[sc] = max((x.salience for x in dd), default=0.0)
        atoms = set()
        for x in dd:
            lid = asg.get(x.descriptor_id)
            if lid and x.axis in BEHAVIOR_AXES:
                atoms.add(name[lid])
        behav[sc] = atoms

    # behavior-novelty: rarity of a scene's behavior atoms across the corpus
    atom_scene_count = defaultdict(int)
    for sc, atoms in behav.items():
        for a in atoms:
            atom_scene_count[a] += 1
    N = len(scenes)
    def raw_nov(sc):
        atoms = behav[sc]
        if not atoms:
            return 0.0
        return sum(-math.log(atom_scene_count[a] / N) for a in atoms) / len(atoms)
    raws = {sc: raw_nov(sc) for sc in scenes}
    lo, hi = min(raws.values()), max(raws.values())
    nov = {sc: ((raws[sc] - lo) / (hi - lo) if hi > lo else 0.0) for sc in scenes}

    # cheap pre-rank (salience proxy for difficulty + behavior novelty) -> candidates
    prelim = sorted(scenes, key=lambda s: -(W_DIFF * maxsal[s] + W_NOV * nov[s]))
    cands = prelim[: args.candidates]
    print(f"[hn] VLM-verifying top {len(cands)} candidates (difficulty + description)...", file=sys.stderr)

    # expensive: real DIFFICULTY (VLM, video) on candidates only. No re-watch
    # description — descriptions are SYNTHESIZED from the composition below.
    reason = CosmosReasonClient(max_tokens=700)
    info = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        fd = {pool.submit(_difficulty, reason, _ref(s)): s for s in cands}
        for f in as_completed(fd):
            s = fd[f]; diff, why = f.result(); info.setdefault(s, {})["diff"] = diff; info[s]["why"] = why

    # final rank: difficulty (real) heavy + behavior novelty
    def final(s):
        d = info.get(s, {}).get("diff", 0.0); d = d if d >= 0 else maxsal[s]
        return W_DIFF * d + W_NOV * nov[s]
    ranked = sorted(cands, key=lambda s: -final(s))[: args.topk]

    # SYNTHESIZE a novel, generatable scenario from each ranked scene's COMPOSITION
    # (behavior atoms + a few grounding spans) — not a caption of the clip.
    def _ground(s):
        top = sorted(by_scene[s], key=lambda d: -d.salience)[:4]
        return [d.reasoning_span for d in top if d.reasoning_span]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        fsyn = {pool.submit(synthesize_scenario,
                            sorted(behav[s]) or [name.get(asg.get(by_scene[s][0].descriptor_id), "scene")],
                            _ground(s)): s for s in ranked}
        for f in as_completed(fsyn):
            info.setdefault(fsyn[f], {})["desc"] = f.result()

    # write Judge feed (real scenes, hard+novel ranked, with vivid descriptions)
    feed = []
    for s in ranked:
        i = info.get(s, {})
        d = i.get("diff", 0.0); d = d if d >= 0 else None
        feed.append({
            "composition_id": s,
            "constituents": sorted(behav[s]) or [f"conditions:{name.get(asg.get(by_scene[s][0].descriptor_id),'scene')}"],
            "marginal_frequencies": {}, "pairwise_frequencies": {},
            "expected_joint": 0.0, "observed_joint": 1.0, "novelty_score": 0.0,
            "motivating_scene_ids": [{"segment_id": s, "window_idx": 0}],
            "arm": "hard_novel",
            "plausibility_score": 1.0,
            "plausibility_justification": i.get("desc", i.get("why", "")),
            "frontier_difficulty_score": d,
            "frontier_difficulty_signals": {"behavior_novelty": round(nov[s], 2),
                                            "max_salience": round(maxsal[s], 2)},
            "final_rank_score": round(final(s) * 5, 1),
            "accepted": True, "rejection_reason": None,
        })
    (OUT / "judge_scored.json").write_text(json.dumps(feed, indent=2))
    (OUT / "hardnovel_ranking.json").write_text(json.dumps(
        [{"scene_id": s, "difficulty": info.get(s, {}).get("diff"),
          "behavior_novelty": round(nov[s], 2), "final": round(final(s), 3),
          "behaviors": sorted(behav[s])} for s in ranked], indent=2))

    print(f"\n[hn] DONE — top {len(feed)} hard+novel scenes -> judge feed")
    for s in ranked[:10]:
        d = info.get(s, {}).get("diff", 0.0)
        print(f"   D={d:.2f} nov={nov[s]:.2f} final={final(s):.2f}  {', '.join(sorted(behav[s]) and [b.split(':',1)[1] for b in sorted(behav[s])][:3]) or '(routine)'}")


if __name__ == "__main__":
    main()
