"""Module 4: Scorer — public interface.

Accepts CompositionProposal objects from Module 3, scores each for plausibility
and frontier difficulty, applies the acceptance filter, and returns ScoredProposal
objects to Module 5.

Cache strategy:
  Key = sha256(composition_id | p_model_id | d_model_id | p_prompt_v | d_prompt_v)
  where d_model_id = "no_difficulty_client" when difficulty_client is None.
  Location: {cache_root}/scorer/{sha256_key}.json
  Write: atomic (.json.tmp → .json)

Failure handling:
  Plausibility failure (all runs fail) → caught per-proposal → ScoredProposal
  emitted with accepted=False, rejection_reason="plausibility_check_failed".
  The batch continues. The failed proposal is not cached (it may succeed on retry).

  Difficulty failure (all runs fail OR client=None) → frontier_difficulty_score=None,
  frontier_difficulty_signals={}. The proposal is still accepted/rejected on
  plausibility alone. final_rank_score uses 0.0 for the difficulty term.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from pipeline.interfaces.proposal import CompositionProposal, ScoredProposal
from pipeline.modules.scorer.config import (
    PlausibilityCheckFailedError,
    ScorerConfig,
    TextClient,
)
from pipeline.modules.scorer.difficulty import DifficultyArm
from pipeline.modules.scorer.plausibility import PlausibilityArm

_NO_DIFFICULTY_SENTINEL = "no_difficulty_client"

REJECTION_PLAUSIBILITY_FAILED = "plausibility_check_failed"
REJECTION_BELOW_THRESHOLD = "plausibility_below_threshold"


class Scorer:
    """Scores CompositionProposals for plausibility and frontier difficulty.

    Parameters
    ----------
    plausibility_client
        TextClient for the plausibility arm. Required.
    difficulty_client
        TextClient for the difficulty arm. If None, frontier_difficulty_score
        is always None and the difficulty term is excluded from final_rank_score.
    config
        Scoring configuration (thresholds, weights, prompt versions).
    cache_root
        Directory for caching ScoredProposals. None = no caching.

    Thread safety
    -------------
    score() is re-entrant. Two calls on the same proposal concurrently may
    both miss cache and both invoke the VLM (double spend, not corruption).
    Atomic cache writes ensure no corrupt entries.
    """

    def __init__(
        self,
        plausibility_client: TextClient,
        difficulty_client: TextClient | None = None,
        config: ScorerConfig = ScorerConfig(),
        cache_root: Path | None = None,
    ) -> None:
        self._config = config
        self._cache_root = Path(cache_root) if cache_root else None

        self._plausibility_arm = PlausibilityArm(
            client=plausibility_client,
            prompt_version=config.plausibility_prompt_version,
        )
        self._difficulty_arm: DifficultyArm | None = (
            DifficultyArm(
                client=difficulty_client,
                prompt_version=config.difficulty_prompt_version,
                signal_weights=config.difficulty_signal_weights,
            )
            if difficulty_client is not None
            else None
        )

        self._p_model_id = plausibility_client.model_id
        self._d_model_id = (
            difficulty_client.model_id
            if difficulty_client is not None
            else _NO_DIFFICULTY_SENTINEL
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def score(self, proposal: CompositionProposal) -> ScoredProposal:
        """Score one proposal. Returns ScoredProposal always — never raises.

        Plausibility failure → accepted=False, rejection_reason set.
        Difficulty failure → frontier_difficulty_score=None.
        """
        cache_key = self._cache_key(proposal.composition_id)
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached

        result = self._score_uncached(proposal)

        # Only cache successful plausibility results
        if result.rejection_reason != REJECTION_PLAUSIBILITY_FAILED:
            self._write_cache(cache_key, result)

        return result

    def score_batch(self, proposals: list[CompositionProposal]) -> list[ScoredProposal]:
        """Score a list of proposals. Failures are caught per-proposal."""
        return [self.score(p) for p in proposals]

    # ------------------------------------------------------------------
    # Internal scoring logic
    # ------------------------------------------------------------------

    def _score_uncached(self, proposal: CompositionProposal) -> ScoredProposal:
        plausibility_score, plausibility_justification, plausibility_failed = (
            self._run_plausibility(proposal)
        )

        difficulty_score, difficulty_signals = self._run_difficulty(proposal)

        accepted, rejection_reason = self._apply_acceptance_filter(
            plausibility_score=plausibility_score,
            plausibility_failed=plausibility_failed,
            threshold=self._config.plausibility_threshold,
        )

        final_rank = self._final_rank_score(
            novelty_score=proposal.novelty_score,
            plausibility_score=plausibility_score,
            difficulty_score=difficulty_score,
        )

        return ScoredProposal(
            composition_id=proposal.composition_id,
            constituents=proposal.constituents,
            marginal_frequencies=proposal.marginal_frequencies,
            pairwise_frequencies=proposal.pairwise_frequencies,
            expected_joint=proposal.expected_joint,
            observed_joint=proposal.observed_joint,
            novelty_score=proposal.novelty_score,
            motivating_scene_ids=proposal.motivating_scene_ids,
            arm=proposal.arm,
            plausibility_score=plausibility_score,
            plausibility_justification=plausibility_justification,
            frontier_difficulty_score=difficulty_score,
            frontier_difficulty_signals=difficulty_signals,
            final_rank_score=final_rank,
            accepted=accepted,
            rejection_reason=rejection_reason,
        )

    def _run_plausibility(
        self, proposal: CompositionProposal
    ) -> tuple[float, str, bool]:
        """Returns (score, justification, failed).

        On failure: score=0.0, justification="", failed=True.
        """
        try:
            score, just = self._plausibility_arm.score(
                composition_id=proposal.composition_id,
                constituents=proposal.constituents,
                marginal_frequencies=proposal.marginal_frequencies,
                expected_joint=proposal.expected_joint,
                observed_joint=proposal.observed_joint,
            )
            return score, just, False
        except PlausibilityCheckFailedError:
            return 0.0, "", True

    def _run_difficulty(
        self, proposal: CompositionProposal
    ) -> tuple[float | None, dict[str, float]]:
        if self._difficulty_arm is None:
            return None, {}
        return self._difficulty_arm.score(
            composition_id=proposal.composition_id,
            constituents=proposal.constituents,
            marginal_frequencies=proposal.marginal_frequencies,
            expected_joint=proposal.expected_joint,
            observed_joint=proposal.observed_joint,
        )

    @staticmethod
    def _apply_acceptance_filter(
        plausibility_score: float,
        plausibility_failed: bool,
        threshold: float,
    ) -> tuple[bool, str | None]:
        if plausibility_failed:
            return False, REJECTION_PLAUSIBILITY_FAILED
        if plausibility_score < threshold:
            return False, REJECTION_BELOW_THRESHOLD
        return True, None

    def _final_rank_score(
        self,
        novelty_score: float,
        plausibility_score: float,
        difficulty_score: float | None,
    ) -> float:
        w = self._config.weights
        return (
            w.novelty * novelty_score
            + w.plausibility * plausibility_score
            + w.difficulty * (difficulty_score if difficulty_score is not None else 0.0)
        )

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _cache_key(self, composition_id: str) -> str:
        raw = "|".join([
            composition_id,
            self._p_model_id,
            self._d_model_id,
            self._config.plausibility_prompt_version,
            self._config.difficulty_prompt_version,
        ])
        return hashlib.sha256(raw.encode()).hexdigest()

    def _read_cache(self, key: str) -> ScoredProposal | None:
        if self._cache_root is None:
            return None
        path = self._cache_root / "scorer" / f"{key}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return ScoredProposal.from_json(data)
        except Exception as exc:
            print(
                f"[Scorer] Cache read failed for {key}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return None

    def _write_cache(self, key: str, result: ScoredProposal) -> None:
        if self._cache_root is None:
            return
        cache_dir = self._cache_root / "scorer"
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{key}.json"
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(result.to_json(), indent=2), encoding="utf-8")
            os.replace(tmp, path)
        except Exception as exc:
            print(
                f"[Scorer] Cache write failed for {key}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            if tmp.exists():
                tmp.unlink(missing_ok=True)
