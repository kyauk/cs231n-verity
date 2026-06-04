"""Independent difficulty cross-check — a SEPARATE VLM look at the clip.

This is the confabulation guard: salience is judged at extraction; difficulty is
re-judged here from a fresh viewing. Disagreement (high salience, low difficulty)
flags a possible over-report. The VLM client is injected (a `describe(video, prompt)`
seam) — swap Cosmos-Reason for Alpamayo's reasoning stack later with no other change.
"""

from __future__ import annotations

import json
import re
from typing import Protocol, runtime_checkable

from pipeline.modules.selection.config import DIFFICULTY_FACTORS, DIFFICULTY_PROMPT


@runtime_checkable
class DifficultyVLM(Protocol):
    def describe(self, video_ref: str, prompt: str) -> str: ...


def score_difficulty(client: DifficultyVLM, video_ref: str) -> tuple[float, str]:
    """Returns (difficulty in 0..1, one-line reason). difficulty == -1.0 on failure.

    The prompt asks for FOUR independent 0-25 factors (visibility / agents /
    maneuver / hazard) rather than a single 0-100 pick. We SUM them here — the
    model never sees a single scale to snap to, so the total interpolates (a sum
    of four independent judgments lands on 43, 67, … not on a round anchor). This
    is the de-anchoring fix: the old single-number prompt collapsed to {0.0, 0.3}
    (then {0.35, 0.5}) because the model treated the calibration anchors as a menu.

    Fallback: if the reply gives a single `difficulty` field instead of factors,
    we still parse it (values > 1 are treated as the 0-100 scale).
    """
    try:
        raw = client.describe(video_ref, DIFFICULTY_PROMPT)
        m = re.search(r"\{[\s\S]*\}", raw)
        obj = json.loads(m.group(0)) if m else {}
        present = [f for f in DIFFICULTY_FACTORS if f in obj]
        if present:
            # sum the four 0-25 factors -> 0-100 -> 0..1 (compute it ourselves;
            # small models do arithmetic poorly, so never trust a model-summed total)
            total = sum(max(0.0, min(25.0, float(obj.get(f, 0.0)))) for f in DIFFICULTY_FACTORS)
            val = total / 100.0
        else:
            val = float(obj.get("difficulty", 0.0))
            if val > 1.0:
                val = val / 100.0
        return max(0.0, min(1.0, val)), str(obj.get("reason", "")).strip()
    except Exception as exc:  # noqa: BLE001
        return -1.0, f"difficulty check failed: {exc}"
