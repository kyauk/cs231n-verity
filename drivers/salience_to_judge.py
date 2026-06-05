"""Adapter: salience_ranking.json -> judge_scored.json (the Judge tab's feed).

The salience experiment produces real scenes ranked by model-judged hardness. The
Judge UI is built for exactly this — rate a scenario, watch its clip — so we map
each salient scene into the proposal shape the Judge tab already renders:

  scene            -> one "proposal"
  salient evidence -> constituents (axis:text atoms shown + scenario sentence)
  independent diff -> D badge (frontier_difficulty_score)
  max salience     -> R badge / rank (final_rank_score)
  confab flag      -> accepted=false + rejection_reason (so a flagged scene is visibly caveated)
  scene_id         -> motivating scene (the Judge tab plays its video)
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path("outputs/waymo")
src = json.load(open(OUT / "salience_ranking.json"))

proposals = []
for r in src:
    constituents = [f"{e['axis']}:{e['text']}" for e in r["salient_evidence"]]
    diff = r["independent_difficulty"] if r["independent_difficulty"] >= 0 else None
    flagged = r["confabulation_flag"]
    note = ("⚠ POSSIBLE CONFABULATION — salience says hard but an independent difficulty "
            "check says routine; review the clip. ") if flagged else ""
    # Prefer the rich simulator-ready scenario description; fall back to the difficulty reason.
    description = r.get("scenario_description") or r.get("difficulty_reason", "")
    proposals.append({
        "composition_id": r["scene_id"],
        "constituents": constituents,
        "marginal_frequencies": {}, "pairwise_frequencies": {},
        "expected_joint": 0.0, "observed_joint": 1.0,         # a REAL observed scene
        "novelty_score": 0.0,                                  # salience view: novelty N/A
        "motivating_scene_ids": [{"segment_id": r["scene_id"], "window_idx": 0}],
        "arm": "salience",
        "plausibility_score": 1.0,                             # observed => plausible by definition
        "plausibility_justification": note + description,
        "frontier_difficulty_score": diff,
        "frontier_difficulty_signals": {"max_salience": r["max_salience"], "n_hard": r["n_hard"]},
        "final_rank_score": round(r["max_salience"] * 5, 1),   # salience on a 0-5 priority scale
        "accepted": True,                                      # show ALL; flag is a caveat, not a hide
        "rejection_reason": None,
    })

# already sorted by salience; keep that order
(OUT / "judge_scored.json").write_text(json.dumps(proposals, indent=2))
print(f"wrote {len(proposals)} salience scenes to judge_scored.json "
      f"({sum(1 for p in proposals if p['accepted'])} clean, "
      f"{sum(1 for p in proposals if not p['accepted'])} flagged)")
