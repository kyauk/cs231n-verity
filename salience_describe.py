"""Scenario-description stage: rich, simulator-ready environment specs.

Selection (salience) finds the edge-case scenes; this DESCRIBES them. The
salience reasoning is deliberately terse about routine context (it optimizes for
finding what's hard), which makes for bland scenario text. A downstream
generation model needs the COMPLETE environment, so for each surfaced scene we
make one focused VLM pass that writes a vivid, concrete, regenerate-from-scratch
description — road/lane layout, signals, weather, lighting, setting, and every
agent with position + intent.

Reads/updates outputs/waymo/salience_ranking.json (adds `scenario_description`).
No re-extraction; only the top-K scenes are described.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pipeline.modules.extractor import CosmosReasonClient

RANKING = Path("outputs/waymo/salience_ranking.json")
BUCKET = "gs://nvidia-adr-waymo-segment-videos"

_SCENARIO_PROMPT = (
    "You are writing a scene specification for an autonomous-vehicle SIMULATOR. "
    "Watch this clip and write a COMPLETE, vivid, concrete description of the driving "
    "ENVIRONMENT — detailed enough that someone could faithfully RECREATE this scene "
    "with no other information. Cover specifically: the road type and lane layout "
    "(how many lanes, directions, markings); any intersections, crosswalks, traffic "
    "signals or signs; the weather, lighting, and time of day; the surrounding setting "
    "(urban / residential / highway, buildings, parked cars, vegetation); and EVERY "
    "notable agent (vehicles, pedestrians, cyclists) with its position, motion, and "
    "apparent intent. Write 5-8 concrete sentences — complete and specific, never terse.\n\n"
    "This scene was automatically flagged for these notable elements:\n{TAGS}\n"
    "GROUND-CHECK each one: in your description, explicitly confirm whether you can see "
    "it and WHERE — and if a flagged element is NOT actually visible in the clip, say so "
    "plainly (e.g. 'Note: no cyclist on a sidewalk is visible in this clip'). Be honest; "
    "describe only what is actually visible and do not force-confirm something that isn't there."
)


def _synth(evidence: list[dict]) -> str:
    """Synthesize a NOVEL generatable scenario from the scene's salient atoms +
    grounding spans — not a re-watch of the clip."""
    from pipeline.modules.selection import synthesize_scenario  # noqa: PLC0415
    atoms = [f"{e.get('axis', 'scene')}:{e['text']}" for e in evidence]
    grounding = [e.get("span", "") for e in sorted(evidence, key=lambda x: -x.get("salience", 0))[:4]]
    return synthesize_scenario(atoms, grounding)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=15)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--camera", default="FRONT")
    args = ap.parse_args()

    rows = json.loads(RANKING.read_text())
    targets = rows[: args.topk]
    reason = CosmosReasonClient(max_tokens=700)

    def ref(sid: str) -> str:
        return f"{BUCKET}/segments/{sid}/{sid}_{args.camera}.mp4"

    print(f"[describe] synthesizing novel scenarios for {len(targets)} scenes", file=sys.stderr)
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_synth, r["salient_evidence"]): i for i, r in enumerate(targets)}
        for f in as_completed(futs):
            i = futs[f]; done += 1
            targets[i]["scenario_description"] = f.result()
            print(f"[describe] {done}/{len(targets)} {targets[i]['scene_id'][:22]}", file=sys.stderr)

    RANKING.write_text(json.dumps(rows, indent=2))
    print(f"\n[describe] done. Example:\n{targets[0]['scenario_description'][:400]}")


if __name__ == "__main__":
    main()
