"""Salience-first discovery: rank REAL scenes by model-judged operational hardness.

The compositional-novelty objective surfaced statistical ghosts (mundane, no
evidence). This is the selection fix: extract with a salience-first prompt (each
descriptor tagged with learned criticality, "routine" a valid answer), then rank
scenes by how HARD they actually are — not how rare their attribute-combo is.

Every result is a real scene with its evidence + spans. As a guard against the
model confabulating danger to satisfy the prompt, we INDEPENDENTLY re-ask
difficulty (a separate call) and flag scenes where salience is high but that
independent difficulty is low.

    set -a && . ./.env && set +a
    .venv/bin/python verity_salience.py --limit 0 --topk 12
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pipeline.modules.extractor import (
    Extractor, ExtractorConfig, CosmosReasonClient, NIMStructureClient, NIMEmbedder,
)
from pipeline.modules.extractor.clients import _gcs_to_data_uri  # reuse clip->data-uri

OUT = Path("outputs/waymo")
STORE = OUT / "taxonomy_store_salience"

_DIFF_PROMPT = (
    "You are an AV safety analyst. On a scale of 0.0 to 1.0, how operationally "
    "DIFFICULT is this clip for an automated driver? Calibrate: 0.0-0.2 = routine "
    "(clear road, normal traffic, nothing emerging); 0.3-0.6 = moderate (a normal "
    "pedestrian crossing, moderate traffic, OR reduced visibility/traction from rain, "
    "fog, night, or wet road); 0.7-1.0 = genuinely hard (an occluded or suddenly-"
    "emerging road user, a conflict / near-miss, ambiguous right-of-way). Most clips "
    "are routine — be honest, neither inflate a calm scene nor dismiss real reduced-"
    'visibility difficulty. Answer ONLY: {"difficulty": <0..1>, "reason": "<one short reason>"}'
)


def _independent_difficulty(reason: CosmosReasonClient, ref: str) -> tuple[float, str]:
    """A SEPARATE difficulty judgment (not the per-descriptor salience) — the cross-check."""
    try:
        raw = reason.describe(ref, _DIFF_PROMPT)
        import re
        m = re.search(r"\{[\s\S]*\}", raw)
        obj = json.loads(m.group(0)) if m else {}
        return max(0.0, min(1.0, float(obj.get("difficulty", 0.0)))), str(obj.get("reason", "")).strip()
    except Exception:  # noqa: BLE001
        return -1.0, "difficulty check failed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--camera", default="FRONT")
    args = ap.parse_args()

    from pipeline.modules.curator import TaxonomyStore
    from verity_run import list_segments
    store = TaxonomyStore(STORE)
    extractor = Extractor(CosmosReasonClient(), NIMStructureClient(), NIMEmbedder(), ExtractorConfig())

    limit = args.limit if args.limit and args.limit > 0 else 10_000
    scenes = [(seg, f"gs://nvidia-adr-waymo-segment-videos/segments/{seg}/{seg}_{args.camera}.mp4")
              for seg, _ in list_segments(limit, args.camera)]
    already = {d.scene_id for d in store.load_descriptors()}
    pending = [(s, r) for s, r in scenes if s not in already]
    print(f"[sal] {len(scenes)} scenes; {len(pending)} to extract (salience-first)", file=sys.stderr)

    # --- extract with salience ------------------------------------------------
    new = []; done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(extractor.extract, s, r): s for s, r in pending}
        for f in as_completed(futs):
            done += 1
            try:
                new.extend(f.result())
            except Exception as exc:  # noqa: BLE001
                print(f"[sal] {done}/{len(pending)} {futs[f][:22]} FAILED: {exc}", file=sys.stderr)
            if done % 20 == 0:
                print(f"[sal] extracted {done}/{len(pending)}", file=sys.stderr)
    store.append_descriptors(new)
    ds = store.load_descriptors()
    print(f"[sal] {len(ds)} descriptors over {len({d.scene_id for d in ds})} scenes", file=sys.stderr)

    # --- rank scenes by salience ---------------------------------------------
    by_scene: dict[str, list] = {}
    for d in ds:
        by_scene.setdefault(d.scene_id, []).append(d)
    ranked = []
    for sid, descs in by_scene.items():
        sals = sorted((x.salience for x in descs), reverse=True)
        top = sals[0] if sals else 0.0
        n_hard = sum(1 for s in sals if s >= 0.6)
        ranked.append((sid, top, n_hard, descs))
    ranked.sort(key=lambda r: (r[1], r[2]), reverse=True)

    # --- independent difficulty cross-check on the top-K (confabulation guard) -
    ref_of = {s: r for s, r in scenes}
    reason = CosmosReasonClient()
    out = []
    for sid, top, n_hard, descs in ranked[: args.topk]:
        diff, why = _independent_difficulty(reason, ref_of.get(sid, ""))
        confab = (top >= 0.6 and 0 <= diff < 0.3)
        salient = sorted([d for d in descs if d.salience >= 0.5],
                         key=lambda d: -d.salience)[:6]
        out.append({
            "scene_id": sid, "max_salience": round(top, 2), "n_hard": n_hard,
            "independent_difficulty": round(diff, 2), "confabulation_flag": confab,
            "difficulty_reason": why,
            "salient_evidence": [{"axis": d.axis, "text": d.text,
                                  "salience": round(d.salience, 2), "span": d.reasoning_span}
                                 for d in salient],
        })
    (OUT / "salience_ranking.json").write_text(json.dumps(out, indent=2))

    print("\n=== TOP SCENES BY SALIENCE (independent difficulty cross-check) ===")
    for r in out:
        flag = "  ⚠ CONFABULATION?" if r["confabulation_flag"] else ""
        print(f"\n{r['scene_id'][:30]}  salience={r['max_salience']} "
              f"indep_difficulty={r['independent_difficulty']}{flag}")
        for e in r["salient_evidence"][:4]:
            print(f"    [{e['axis']}] {e['text']} (s={e['salience']})  <- \"{e['span'][:60]}\"")


if __name__ == "__main__":
    main()
