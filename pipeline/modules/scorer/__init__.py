"""Module 4: Scorer — plausibility and frontier-difficulty scoring.

Scores CompositionProposals (from the Hypothesizer) for plausibility and
frontier difficulty, applies the acceptance filter, and emits ScoredProposals.

Public surface (import from the package root):
    from pipeline.modules.scorer import Scorer, ScorerConfig

Both arms take an injected TextClient. Use the Stub* clients for offline runs
and tests; the Failing* clients deliberately error, for failure-path tests.
"""

from pipeline.modules.scorer.config import (
    PlausibilityCheckFailedError,
    ScorerConfig,
    ScorerError,
    ScorerWeights,
    TextClient,
)
from pipeline.modules.scorer.difficulty import (
    FailingDifficultyClient,
    StubDifficultyClient,
)
from pipeline.modules.scorer.plausibility import (
    FailingPlausibilityClient,
    StubPlausibilityClient,
)
from pipeline.modules.scorer.scorer import (
    REJECTION_BELOW_THRESHOLD,
    REJECTION_PLAUSIBILITY_FAILED,
    Scorer,
)

__all__ = [
    # Entry point + config
    "Scorer",
    "ScorerConfig",
    "ScorerWeights",
    # Client protocol + test doubles
    "TextClient",
    "StubPlausibilityClient",
    "FailingPlausibilityClient",
    "StubDifficultyClient",
    "FailingDifficultyClient",
    # Errors + rejection-reason constants
    "ScorerError",
    "PlausibilityCheckFailedError",
    "REJECTION_PLAUSIBILITY_FAILED",
    "REJECTION_BELOW_THRESHOLD",
]
