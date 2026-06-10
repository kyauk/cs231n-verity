"""Held-out incident recall — the metric reality grades.

Take real documented AV-relevant hard scenarios the pipeline NEVER saw, phrased
neutrally (so the embedding match favours no arm's writing style), and measure
whether each arm's discovered/proposed scenarios contain a semantically-matching
scenario at a higher rate than the others. This is external ground truth, not a
rater and not our own scorer.

Pre-registered prediction (locked before running):
  * Verity >= Comp-A on recall is the hoped result, BUT the ungrounded LLM has
    these famous incidents in its training data, so it may match well by
    regurgitation. The robust Verity edge is GROUNDING: 100% of Verity/Comp-B
    matches trace to a real fleet scene; 0% of Comp-A's do. We report both.

    set -a && . ./.env && set +a
    .venv/bin/python -m drivers.verity_eval_recall
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

from openai import OpenAI

EVAL = Path("outputs/waymo/eval")
EMBED_URL = os.environ.get("EMBED_TEXT_BASE_URL", "https://integrate.api.nvidia.com/v1")
EMBED_MODEL = os.environ.get("EMBED_TEXT_MODEL_ID", "nvidia/nv-embedqa-e5-v5")

# Documented real-world AV-relevant hard scenarios, held out from the corpus.
# Phrased as neutral factual descriptions. (Sources: NTSB/NHTSA SGO reports, public
# incident records — cite precisely before submission.)
INCIDENTS = [
    ("tempe_2018", "At night, a pedestrian walks a bicycle across a multi-lane road outside a crosswalk; she is detected late against the dark background and not yielded to."),
    ("stopped_firetruck", "On a highway at speed, a stationary emergency vehicle is parked in the travel lane ahead; the moving vehicle must detect the static obstacle in time to stop or change lanes."),
    ("cruise_secondary", "A pedestrian is thrown into the vehicle's path by a separate collision and ends up in the roadway directly ahead; the vehicle must handle a person who appears mid-lane after impact."),
    ("emergency_responder_hand", "A police officer or responder stands in the intersection directing traffic by hand, overriding the traffic signals, which still display their normal cycle."),
    ("cyclist_right_hook", "The vehicle turns right across a bike lane while a cyclist continues straight in that lane alongside it, creating a right-hook conflict."),
    ("school_bus_children", "A stopped school bus extends its stop arm with red lights flashing while children cross the road in front of it, partly hidden by the bus."),
    ("unprotected_left", "The vehicle makes an unprotected left turn across a stream of fast oncoming traffic at a permissive green as the signal ages toward yellow."),
    ("railroad_gates", "At a level railroad crossing the warning lights begin flashing and the gates start to lower as the vehicle approaches the tracks."),
    ("construction_flagger", "A construction flagger redirects traffic with a handheld sign, routing vehicles against the painted lane markings through a work zone."),
    ("dark_pedestrian_rain", "On a rainy night a pedestrian in dark clothing crosses an unlit road; glare and low contrast make detection difficult."),
    ("double_parked_oncoming", "A delivery vehicle is double-parked blocking the lane on a narrow two-way street, forcing the vehicle into the oncoming lane while an oncoming car approaches."),
    ("phantom_overpass", "Approaching an overpass, a strong shadow line across the road is mistaken for an obstacle, risking an unnecessary hard brake on a clear highway."),
    ("wrong_way_driver", "On a divided road a vehicle approaches head-on travelling the wrong way, closing rapidly at night."),
    ("highway_debris", "A large piece of tire debris lies in the travel lane on a highway, revealed at the last moment as the lead vehicle swerves around it."),
    ("occluded_ped_parked", "A pedestrian steps into the road from between two parked vehicles, hidden until they emerge directly in the vehicle's path."),
    ("black_ice_bridge", "On a freezing morning a bridge deck is glazed with near-invisible ice; the road looks merely wet but traction is sharply reduced."),
]

ARMS = [("verity", "verity.json"), ("ungrounded_llm", "ungrounded_llm.json"),
        ("compositional_rarity", "compositional_rarity.json")]


def _desc(e):
    return e.get("plausibility_justification") or e.get("description", "")


def _embed(client, texts, kind):
    r = client.embeddings.create(model=EMBED_MODEL, input=texts, extra_body={"input_type": kind})
    return [d.embedding for d in r.data]


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-9)


def main():
    client = OpenAI(api_key=os.environ["NVIDIA_API_KEY"], base_url=EMBED_URL)
    inc_emb = _embed(client, [d for _, d in INCIDENTS], "query")

    arm_props = {}      # arm -> list of (id, desc, grounded_scene_or_None)
    for arm, fn in ARMS:
        rows = json.loads((EVAL / fn).read_text())
        props = []
        for e in rows:
            sc = (e.get("motivating_scene_ids") or [{}])
            grounded = sc[0].get("segment_id") if sc and sc[0] else None
            props.append((e["composition_id"], _desc(e), grounded))
        arm_props[arm] = props

    arm_emb = {arm: _embed(client, [d for _, d, _ in props], "passage")
               for arm, props in arm_props.items()}

    # best match per incident per arm
    best = {arm: [] for arm, _ in ARMS}
    for i, (iid, _) in enumerate(INCIDENTS):
        for arm, _ in ARMS:
            sims = [(_cos(inc_emb[i], arm_emb[arm][j]), arm_props[arm][j])
                    for j in range(len(arm_props[arm]))]
            sims.sort(key=lambda t: -t[0])
            best[arm].append((iid, sims[0][0], sims[0][1]))

    print(f"{len(INCIDENTS)} held-out incidents x {sum(len(p) for p in arm_props.values())} proposals\n")
    # recall at thresholds
    print("=== recall: fraction of incidents with a match >= tau ===")
    print(f"{'tau':>5}  " + "  ".join(f"{a:>20}" for a, _ in ARMS))
    for tau in [0.55, 0.60, 0.65, 0.70, 0.75]:
        row = []
        for arm, _ in ARMS:
            cov = sum(1 for _, s, _ in best[arm] if s >= tau)
            row.append(f"{cov}/{len(INCIDENTS)} ({cov/len(INCIDENTS)*100:3.0f}%)")
        print(f"{tau:>5}  " + "  ".join(f"{c:>20}" for c in row))
    # mean best-match similarity
    print("\n=== mean best-match cosine (higher = arm's scenarios are closer to real incidents) ===")
    for arm, _ in ARMS:
        m = sum(s for _, s, _ in best[arm]) / len(INCIDENTS)
        print(f"  {arm:>22}: {m:.3f}")
    # grounding: of the best matches, how many trace to real footage
    print("\n=== grounding of best matches (traceable to a real fleet scene?) ===")
    for arm, _ in ARMS:
        g = sum(1 for _, _, p in best[arm] if p[2])
        print(f"  {arm:>22}: {g}/{len(INCIDENTS)} matches grounded in real footage")
    # per-incident winner
    print("\n=== per-incident best arm (tau=0.65) ===")
    for i, (iid, _) in enumerate(INCIDENTS):
        scores = {arm: best[arm][i][1] for arm, _ in ARMS}
        win = max(scores, key=scores.get)
        print(f"  {iid:<22} -> {win:<20} ({scores[win]:.2f})   "
              f"V={scores['verity']:.2f} L={scores['ungrounded_llm']:.2f} R={scores['compositional_rarity']:.2f}")

    json.dump({arm: [(iid, round(s, 3), p[0], p[2]) for iid, s, p in best[arm]] for arm, _ in ARMS},
              open(EVAL / "recall_result.json", "w"), indent=2)


if __name__ == "__main__":
    main()
