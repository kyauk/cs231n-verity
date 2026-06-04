"""Module: Selection — the "expressive salience" layer.

Surfaces edge-case scenes from real evidence and turns them into novel,
generatable scenarios. Three lego-bricks, composable by a driver:

  ranking      — score = difficulty*w + behavior_novelty*w (pure; novelty over
                 interactions/conditions/ego_maneuver only, never agent attrs).
  difficulty   — independent VLM difficulty cross-check (confabulation guard;
                 client injected → Alpamayo-swappable).
  synthesis    — synthesize a NOVEL scenario from a composition's atoms (text;
                 not a re-watch of one clip).

Imports only pipeline.interfaces + external SDKs. Drivers (verity_hardnovel.py
etc.) compose these — same pattern as the curator/extractor modules.

    from pipeline.modules.selection import (
        SelectionConfig, behavior_novelty, combined_score,
        score_difficulty, synthesize_scenario,
    )
"""

from pipeline.modules.selection.config import (
    DIFFICULTY_PROMPT,
    SYNTHESIS_PROMPT,
    SelectionConfig,
)
from pipeline.modules.selection.difficulty import DifficultyVLM, score_difficulty
from pipeline.modules.selection.ranking import behavior_novelty, combined_score
from pipeline.modules.selection.synthesis import synthesize_scenario

__all__ = [
    "SelectionConfig",
    "DIFFICULTY_PROMPT",
    "SYNTHESIS_PROMPT",
    "behavior_novelty",
    "combined_score",
    "score_difficulty",
    "DifficultyVLM",
    "synthesize_scenario",
]
