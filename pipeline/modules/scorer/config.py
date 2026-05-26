"""Module 4: Scorer — configuration, client protocol, and error types.

The Scorer has two VLM arms (plausibility and difficulty) that both use
text-only prompts (no video URL). TextClient is the shared protocol for
both — distinct from the encoder's VLMClient which takes (video_url, prompt).

Cache key sentinel: "no_difficulty_client" when difficulty_client is None at
construction time. This is unambiguous — "none" is ambiguous between "client
returned None" and "no client was configured."
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# TextClient protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class TextClient(Protocol):
    """Text-only VLM client. Both plausibility and difficulty arms use this.

    Implementations: NIMTextClient (production), StubPlausibilityClient and
    StubDifficultyClient (tests/offline). Compatible with any
    OpenAI-chat-completion endpoint that accepts a text prompt.
    """
    model_id: str

    def complete(self, prompt: str) -> str:
        """Call the model and return its raw text output."""
        ...


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

@dataclass
class ScorerWeights:
    """Weights for the final_rank_score formula.

    final_rank_score = (
        novelty      * proposal.novelty_score              (log-scale, ~0–5)
        + plausibility * plausibility_score                (0–1)
        + difficulty   * (frontier_difficulty_score or 0.0) (0–1)
    )

    novelty_score is log-scale; a score of 3.0 is roughly equivalent to
    plausibility/difficulty of 1.0 when using equal weights. Calibrate
    knowing this scale difference — e.g., a smaller novelty weight relative
    to plausibility/difficulty de-emphasizes statistical novelty in favor of
    model-validated difficulty.
    """
    novelty: float = 0.4
    plausibility: float = 0.3
    difficulty: float = 0.3


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ScorerConfig:
    """Configuration for Module 4: Scorer.

    plausibility_threshold
        Hard acceptance filter: proposals with plausibility_score below this
        are marked accepted=False and do not reach the judge.

        # TODO (week 2 calibration): run plausibility on the first 30–50
        # proposals and examine the score distribution. VLMs typically cluster
        # scores in 0.6–0.9; a threshold of 0.5 may filter almost nothing.
        # Set the threshold at roughly the 20th percentile of observed scores
        # and document the chosen value and its empirical basis.

    difficulty_signal_weights
        Three-float tuple (action_variance_w, inverse_confidence_w, mismatch_w)
        controlling the difficulty score formula:
            score = (1 - mean_confidence) * w[1]
                  + action_variance       * w[0]
                  + reasoning_mismatch    * w[2]

        Defaults: action_variance leads (0.5) as the most reliable signal of
        genuine model uncertainty; inverse_confidence is secondary (0.3);
        reasoning_action_mismatch is noisiest so gets the smallest weight (0.2).

        # TODO (week 2 calibration): score 30 proposals with three weight
        # schemes (equal / action-heavy / confidence-heavy), compare which
        # ranking best aligns with expert intuition, lock and document.
    """
    plausibility_threshold: float = 0.5
    plausibility_runs: int = 3
    difficulty_runs: int = 3
    weights: ScorerWeights = field(default_factory=ScorerWeights)
    plausibility_prompt_version: str = "v1_plausibility"
    difficulty_prompt_version: str = "v1_difficulty"
    # (action_variance_weight, inverse_confidence_weight, reasoning_mismatch_weight)
    difficulty_signal_weights: tuple[float, float, float] = (0.5, 0.3, 0.2)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ScorerError(Exception):
    """Base class for Module 4: Scorer errors."""


class PlausibilityCheckFailedError(ScorerError):
    """All plausibility runs failed to produce a parseable score.

    Caught by Scorer.score() per-proposal. The batch continues.
    Resulting ScoredProposal: accepted=False,
    rejection_reason="plausibility_check_failed", plausibility_score=0.0.

    Not cached — transient parse/network failures should retry on next run.
    """
    def __init__(self, composition_id: str, detail: str) -> None:
        self.composition_id = composition_id
        self.detail = detail
        super().__init__(
            f"[Scorer] PlausibilityCheckFailedError for {composition_id!r}: {detail}"
        )
        print(str(self), file=sys.stderr)
