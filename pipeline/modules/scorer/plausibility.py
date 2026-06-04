"""Module 4: Scorer — plausibility arm.

Runs the plausibility VLM prompt against a composition 3 times (with
constituents in different orderings) and aggregates the results.

Aggregation (conservative under partial failure):
  3/3 succeed → median score; justification from the run whose score is
                the median value (lower of the two middle values on ties)
  2/3 succeed → lower of the two scores (conservative read); that run's
                justification
  1/3 succeed → use the single successful score and justification
  0/3 succeed → raise PlausibilityCheckFailedError

Rationale for "lower on 2/3": if one VLM instance finds the scenario
implausible and another finds it plausible, default to the more cautious read.

Seed for constituent ordering shuffle uses hashlib.sha256 (not Python's
built-in hash()) so the seed is deterministic regardless of PYTHONHASHSEED.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import random
import re
import sys
from typing import Any

from pipeline.modules.scorer.config import PlausibilityCheckFailedError, TextClient

_PROMPT_DIR = pathlib.Path(__file__).parent / "prompts"


# ---------------------------------------------------------------------------
# Composition description builder
# ---------------------------------------------------------------------------

def describe_composition(
    composition_id: str,
    constituents: list[str],
    marginal_frequencies: dict[str, float],
    expected_joint: float,
    observed_joint: float,
    ordering_seed_offset: int = 0,
) -> str:
    """Build a natural-language description of the composition for the prompt.

    The ordering of constituents changes with ordering_seed_offset (0, 1, 2)
    to produce the three distinct prompt variations. Seed is derived from
    composition_id via SHA256 — deterministic regardless of PYTHONHASHSEED.
    """
    seed = int(hashlib.sha256(composition_id.encode()).hexdigest()[:8], 16)
    orderings = _three_orderings(constituents, seed)
    ordered = orderings[ordering_seed_offset % 3]

    # NB: deliberately omit dataset frequencies (marginal / observed-joint) from
    # the plausibility prompt. Plausibility must judge whether the conditions can
    # *physically* co-occur, independent of how often they appear in this dataset.
    # Rarity is captured separately by novelty_score; feeding "observed joint
    # frequency: 0.0000" here made the model equate rare-in-data with impossible
    # and reject every proposal (0/30 accepted).
    parts = [f"  - {atom}" for atom in ordered]

    return (
        f"Scenario conditions (count {len(ordered)}):\n"
        + "\n".join(parts)
        + "\n\nJudge only whether these conditions can physically and "
        "behaviorally co-occur in real-world driving. Do NOT consider how "
        "rare or common the combination is — statistical rarity is not the "
        "same as implausibility."
    )


def _three_orderings(constituents: list[str], seed: int) -> list[list[str]]:
    """Return 3 deterministic orderings of constituents."""
    sorted_order = sorted(constituents)
    reversed_order = list(reversed(sorted_order))
    shuffled = sorted_order.copy()
    random.Random(seed).shuffle(shuffled)
    return [sorted_order, reversed_order, shuffled]


# ---------------------------------------------------------------------------
# JSON extraction from VLM plausibility response
# ---------------------------------------------------------------------------

def _extract_plausibility_json(text: str) -> dict[str, Any]:
    """Extract {"score": float, "justification": str} from a VLM response.

    Tries direct JSON parse, then ```json``` fence, then bare {...}.
    Raises ValueError if no valid JSON with the required fields is found.
    """
    candidates = [text.strip()]

    fence = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
    if fence:
        candidates.append(fence.group(1).strip())

    bare_fence = re.search(r"```\s*([\s\S]*?)```", text, re.DOTALL)
    if bare_fence:
        candidates.append(bare_fence.group(1).strip())

    brace = re.search(r"\{[\s\S]*\}", text)
    if brace:
        candidates.append(brace.group(0))

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and "score" in obj and "justification" in obj:
                score = float(obj["score"])
                justification = str(obj["justification"])
                clamped = max(0.0, min(1.0, score))
                if clamped != score:
                    print(
                        f"[Scorer/Plausibility] Score {score} clamped to {clamped}.",
                        file=sys.stderr,
                    )
                return {"score": clamped, "justification": justification}
        except (json.JSONDecodeError, ValueError, TypeError):
            continue

    raise ValueError(
        f"No valid plausibility JSON found.\nResponse preview: {text[:300]!r}"
    )


# ---------------------------------------------------------------------------
# Plausibility arm
# ---------------------------------------------------------------------------

def _load_prompt_template(version: str) -> str:
    path = _PROMPT_DIR / f"{version}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"[Scorer/Plausibility] Prompt template {version!r} not found at {path}"
        )
    return path.read_text(encoding="utf-8")


class PlausibilityArm:
    """Scores one CompositionProposal for physical/behavioral plausibility.

    Runs 3 calls with different constituent orderings, aggregates conservatively.
    """

    def __init__(self, client: TextClient, prompt_version: str = "v1_plausibility") -> None:
        self._client = client
        self._template = _load_prompt_template(prompt_version)

        if "{{COMPOSITION}}" not in self._template:
            raise ValueError(
                f"[Scorer/Plausibility] Prompt template {prompt_version!r} is missing "
                f"{{{{COMPOSITION}}}} placeholder."
            )

    def score(
        self,
        composition_id: str,
        constituents: list[str],
        marginal_frequencies: dict[str, float],
        expected_joint: float,
        observed_joint: float,
    ) -> tuple[float, str]:
        """Run plausibility scoring. Returns (score: float, justification: str).

        Raises PlausibilityCheckFailedError if all runs fail.
        """
        results: list[tuple[float, str]] = []

        for i in range(3):
            description = describe_composition(
                composition_id=composition_id,
                constituents=constituents,
                marginal_frequencies=marginal_frequencies,
                expected_joint=expected_joint,
                observed_joint=observed_joint,
                ordering_seed_offset=i,
            )
            prompt = self._template.replace("{{COMPOSITION}}", description)

            try:
                raw = self._client.complete(prompt)
                parsed = _extract_plausibility_json(raw)
                results.append((parsed["score"], parsed["justification"]))
            except Exception as exc:
                print(
                    f"[Scorer/Plausibility] Run {i+1}/3 failed for {composition_id!r}: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )

        n = len(results)
        if n == 0:
            raise PlausibilityCheckFailedError(
                composition_id,
                "all 3 plausibility runs failed to produce a parseable score",
            )
        if n == 1:
            return results[0]
        if n == 2:
            # Conservative: use the lower score + its justification
            a, b = results
            return (a[0], a[1]) if a[0] <= b[0] else (b[0], b[1])
        # n == 3: median
        results.sort(key=lambda r: r[0])
        return results[1]  # middle value


# ---------------------------------------------------------------------------
# Stub client
# ---------------------------------------------------------------------------

class StubPlausibilityClient:
    """Deterministic stub for tests and offline runs.

    Returns a valid plausibility JSON response every time.
    """
    model_id: str = "stub/plausibility"

    def complete(self, prompt: str) -> str:
        return '{"score": 0.78, "justification": "Physically plausible combination for real-world driving."}'


class FailingPlausibilityClient:
    """Stub that always returns unparseable output — for testing failure paths."""
    model_id: str = "stub/plausibility-failing"

    def complete(self, prompt: str) -> str:
        return "I cannot assess this scenario."
