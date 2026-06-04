"""Hard + novel scene ranking (pure functions).

The selection objective:
    score = difficulty * w_difficulty + behavior_novelty * w_novelty
DIFFICULTY leads (model-judged hardness — granularity-agnostic, catches the
unknown-unknowns). BEHAVIOR-NOVELTY refines, computed over interactions/
conditions/ego_maneuver ONLY — never agent attributes (car colour), which kept
polluting the ranking. Both are 0..1; difficulty is weighted heavily.

These are pure — they take already-projected per-scene behavior atoms (the driver
runs the curator projection) so this module imports only its own config.
"""

from __future__ import annotations

import math
from collections import defaultdict

from pipeline.modules.selection.config import SelectionConfig


def behavior_novelty(behavior_by_scene: dict[str, set[str]]) -> dict[str, float]:
    """Normalized rarity (0..1) of each scene's behavior-atom signature.

    A scene scores high when its behaviors/conditions are uncommon in the corpus;
    a scene with no behavior atoms scores 0.
    """
    n = len(behavior_by_scene) or 1
    count: dict[str, int] = defaultdict(int)
    for atoms in behavior_by_scene.values():
        for a in atoms:
            count[a] += 1
    raw = {
        sc: (sum(-math.log(count[a] / n) for a in atoms) / len(atoms)) if atoms else 0.0
        for sc, atoms in behavior_by_scene.items()
    }
    if not raw:
        return {}
    lo, hi = min(raw.values()), max(raw.values())
    return {sc: ((raw[sc] - lo) / (hi - lo) if hi > lo else 0.0) for sc in raw}


def combined_score(difficulty: float, novelty: float, config: SelectionConfig) -> float:
    """difficulty-heavy blend; a negative (failed) difficulty is treated as 0."""
    return config.w_difficulty * max(0.0, difficulty) + config.w_novelty * novelty
