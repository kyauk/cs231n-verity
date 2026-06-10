"""Before/after top-20: the selection pipeline run over RETYPED atoms.

Mirrors verity_hardnovel.py but swaps in the label-surgery-corrected per-scene
behavior atoms (verity_retype.corrected_atoms). Difficulty is the same VLM call
(video-based, taxonomy-independent), so any top-20 change is attributable to the
typing fix's effect on novelty + candidate selection. Writes the corrected top-20
and prints a diff vs the stored old ranking.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pipeline.modules.extractor import CosmosReasonClient
from pipeline.modules.curator import TaxonomyStore
from pipeline.modules.selection import (
    SelectionConfig, behavior_novelty, combined_score, score_difficulty, synthesize_scenario,
)
from drivers.verity_retype import corrected_atoms

OUT = Path("outputs/waymo")
SRC = "outputs/waymo/taxonomy_store_salience"
BUCKET = "gs://nvidia-adr-waymo-segment-videos"
SEL = SelectionConfig()
BEH = SEL.novelty_axes  # interactions / conditions / ego_maneuver


def _ref(sid): return f"{BUCKET}/segments/{sid}/{sid}_FRONT.mp4"


def main():
    cand_n, topk, workers = 40, 20, 4
    ds = TaxonomyStore(SRC).load_descriptors()
    scene_atoms, _, _ = corrected_atoms(ds)

    # per-scene behavior atoms (corrected) + max salience proxy
    behav = {sc: {f"{ax}:{nm}" for (ax, nm) in atoms if ax in BEH}
             for sc, atoms in scene_atoms.items()}
    maxsal = defaultdict(float)
    for d in ds:
        maxsal[d.scene_id] = max(maxsal[d.scene_id], d.salience)
    scenes = sorted(behav)
    for s in scenes:
        behav.setdefault(s, set())

    nov = behavior_novelty(behav)
    prelim = sorted(scenes, key=lambda s: -combined_score(maxsal[s], nov[s], SEL))
    cands = prelim[:cand_n]
    print(f"[retyped] {len(scenes)} scenes; VLM-scoring top {len(cands)} candidates...", file=sys.stderr)

    reason = CosmosReasonClient(max_tokens=700)
    diff = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        fut = {pool.submit(score_difficulty, reason, _ref(s)): s for s in cands}
        for f in as_completed(fut):
            s = fut[f]; d, _ = f.result(); diff[s] = d

    def final(s):
        d = diff.get(s, 0.0); d = d if d >= 0 else maxsal[s]
        return combined_score(d, nov[s], SEL)
    ranked = sorted(cands, key=lambda s: -final(s))[:topk]

    new = [{"scene_id": s, "difficulty": diff.get(s), "behavior_novelty": round(nov[s], 2),
            "final": round(final(s), 3), "behaviors": sorted(behav[s])} for s in ranked]
    (OUT / "hardnovel_ranking_retyped.json").write_text(json.dumps(new, indent=2))

    if "--promote" in sys.argv:
        # SYNTHESIZE novel scenarios from corrected composition atoms + grounding spans,
        # then write the Judge feed (local NIM for synthesis; format pass is separate).
        by_scene = defaultdict(list)
        for d in ds:
            by_scene[d.scene_id].append(d)
        def ground(s):
            top = sorted(by_scene[s], key=lambda d: -d.salience)[:4]
            return [d.reasoning_span for d in top if d.reasoning_span]
        print("[retyped] synthesizing scenarios for top-20...", file=sys.stderr)
        desc = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            fut = {pool.submit(synthesize_scenario, sorted(behav[s]) or ["a routine scene"], ground(s)): s
                   for s in ranked}
            for f in as_completed(fut):
                desc[fut[f]] = f.result()
        feed = []
        for s in ranked:
            d = diff.get(s, 0.0); d = d if d >= 0 else None
            feed.append({
                "composition_id": s, "constituents": sorted(behav[s]) or ["conditions:routine"],
                "marginal_frequencies": {}, "pairwise_frequencies": {},
                "expected_joint": 0.0, "observed_joint": 1.0, "novelty_score": 0.0,
                "motivating_scene_ids": [{"segment_id": s, "window_idx": 0}],
                "arm": "hard_novel", "plausibility_score": 1.0,
                "plausibility_justification": desc.get(s, ""),
                "frontier_difficulty_score": d,
                "frontier_difficulty_signals": {"behavior_novelty": round(nov[s], 2),
                                                "max_salience": round(maxsal[s], 2)},
                "final_rank_score": round(final(s) * 5, 1),
                "accepted": True, "rejection_reason": None,
            })
        (OUT / "judge_scored.json").write_text(json.dumps(feed, indent=2))
        print(f"[retyped] PROMOTED {len(feed)} corrected proposals -> judge_scored.json", file=sys.stderr)

    # diff vs old
    old = json.loads((OUT / "hardnovel_ranking.json").read_text())
    old_ids = [o["scene_id"] for o in old]
    new_ids = [n["scene_id"] for n in new]
    print("\n=== CORRECTED top-20 (difficulty | novelty | final | behaviors) ===")
    for n in new:
        d = n["difficulty"]; d = f"{d:.2f}" if isinstance(d, (int, float)) else str(d)
        tag = "" if n["scene_id"] in old_ids else "  <-- NEW (was outside top-20)"
        print(f"  D={d} nov={n['behavior_novelty']:.2f} final={n['final']:.3f}  "
              f"{[b.split(':',1)[1] for b in n['behaviors']][:3]}{tag}")
    dropped = [o for o in old if o["scene_id"] not in new_ids]
    print(f"\n=== FELL OUT of top-20 ({len(dropped)}) ===")
    for o in dropped:
        print(f"  was final={o['final']}  D={o.get('difficulty')}  {o.get('behaviors')}")


if __name__ == "__main__":
    main()
