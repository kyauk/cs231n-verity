"""Combine the three eval arms into one blinded, seed-shuffled Judge feed.

Reads:
  outputs/waymo/eval/verity.json               (20 — difficulty-led, already generated)
  outputs/waymo/eval/ungrounded_llm.json       (20 — Comp-A, LLM-invented, no data)
  outputs/waymo/eval/compositional_rarity.json (20 — Comp-B, rarest corpus compositions)

Writes outputs/waymo/judge_scored.json with all 60, shuffled (seed=42) so arms are
interleaved, and final_rank_score assigned by shuffled position (the Judge sorts on
it). The Judge server blinds `arm` from raters and records it with each score.

    .venv/bin/python -m drivers.verity_eval_combine
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

EVAL = Path("outputs/waymo/eval")
FEED = Path("outputs/waymo/judge_scored.json")
SEED = 42


def main() -> None:
    arms = ["verity", "ungrounded_llm", "compositional_rarity"]
    combined: list[dict] = []
    for arm in arms:
        rows = json.loads((EVAL / f"{arm}.json").read_text())
        for e in rows:
            e["arm"] = arm
        combined.extend(rows)

    rng = random.Random(SEED)
    rng.shuffle(combined)
    for pos, e in enumerate(combined):
        e["final_rank_score"] = float(len(combined) - pos)  # shuffled display order
        e["accepted"] = True
        e["rejection_reason"] = None

    FEED.write_text(json.dumps(combined, indent=2))
    print(f"wrote {FEED} — {len(combined)} blinded proposals")
    print("arms:", dict(Counter(e["arm"] for e in combined)))


if __name__ == "__main__":
    main()
