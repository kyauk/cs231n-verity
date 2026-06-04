"""Module 3: Hypothesizer — composition enumeration, filtering, and scoring.

A composition is a frozenset of qualified atoms ("prefix:value") of a given
arity. This module:
  1. Enumerates all k-combinations from the eligible atom pool.
  2. Filters out:
     - Atoms below min_marginal_frequency
     - Compositions where any pairwise co-occurrence is below min_pairwise_frequency
     - Compositions where observed_joint >= max_joint_frequency (already common)
     - Mutual-exclusivity violations: two atoms with the same prefix from a
       SINGLE_CATEGORICAL_FIELD (e.g., weather:fog + weather:rain).
  3. Scores each surviving composition by novelty.
  4. Finds motivating scenes: windows that contain all atoms of the composition.

Novelty score:
    ln(expected_joint / max(observed_joint, epsilon))
    where epsilon = 1 / (10 * N) and N is the number of windows.

Under independence, expected_joint = product of marginal frequencies.
A high score means the atoms are individually common but jointly rare —
the definition of compositional novelty.
"""

from __future__ import annotations

import hashlib
import math
from itertools import combinations
from typing import Any

from pipeline.interfaces.window import WindowKey
from pipeline.interfaces.proposal import CompositionProposal
from pipeline.modules.hypothesizer.config import (
    SINGLE_CATEGORICAL_FIELDS,
    HypothesizerConfig,
)


def atom_prefix(atom: str) -> str:
    """Return the prefix part of a qualified atom ("prefix:value" → "prefix")."""
    return atom.split(":", 1)[0]


def _is_mutually_exclusive(composition: frozenset[str]) -> bool:
    """Return True if the composition contains two atoms from the same
    SINGLE_CATEGORICAL_FIELD (which can never co-occur in one window)."""
    seen_prefixes: dict[str, str] = {}
    for atom in composition:
        prefix = atom_prefix(atom)
        if prefix in SINGLE_CATEGORICAL_FIELDS:
            if prefix in seen_prefixes:
                return True  # e.g. weather:fog + weather:rain
            seen_prefixes[prefix] = atom
    return False


def composition_id(constituents: list[str]) -> str:
    """Deterministic 16-char hex ID from a sorted constituent list."""
    key = "|".join(sorted(constituents))
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _expected_joint(atoms: frozenset[str], marginal: dict[str, float]) -> float:
    """Under independence assumption: product of marginal frequencies."""
    product = 1.0
    for atom in atoms:
        product *= marginal.get(atom, 0.0)
    return product


def _pairwise_key(a: str, b: str) -> str:
    if a > b:
        a, b = b, a
    return f"{a}|{b}"


def _min_pairwise(
    atoms: frozenset[str],
    pairwise: dict[str, float],
) -> float:
    """Minimum pairwise co-occurrence frequency across all atom pairs."""
    atoms_list = sorted(atoms)
    min_freq = math.inf
    for i, a in enumerate(atoms_list):
        for b in atoms_list[i + 1:]:
            freq = pairwise.get(_pairwise_key(a, b), 0.0)
            min_freq = min(min_freq, freq)
    return min_freq if min_freq != math.inf else 0.0


def build_proposals(
    atom_sets: list[frozenset[str]],
    keys: list[WindowKey],
    marginal: dict[str, float],
    pairwise: dict[str, float],
    config: HypothesizerConfig,
    arm: str,
) -> list[CompositionProposal]:
    """Enumerate, filter, score, and rank composition proposals.

    Parameters
    ----------
    atom_sets
        One frozenset per window (already filtered to succeeded records).
    keys
        WindowKey for each atom_set entry (parallel list).
    marginal
        Marginal frequency per atom.
    pairwise
        Pairwise co-occurrence frequency per atom pair.
    config
        Filter thresholds and ranking parameters.
    arm
        "reasoning" or "visual" — passed through to CompositionProposal.

    Returns
    -------
    list[CompositionProposal]
        Ranked by novelty_score DESC, composition_id ASC (tie-breaker).
        Length is at most config.top_k.
    """
    n = len(atom_sets)
    if n == 0:
        return []

    epsilon = 1.0 / (10 * n)

    # Eligible atoms: pass marginal frequency threshold
    eligible = frozenset(
        atom for atom, freq in marginal.items()
        if freq >= config.min_marginal_frequency
    )

    proposals: list[CompositionProposal] = []

    for size in config.composition_sizes:
        if size < 2:
            continue
        eligible_list = sorted(eligible)  # deterministic enumeration order

        for combo in combinations(eligible_list, size):
            composition = frozenset(combo)

            if _is_mutually_exclusive(composition):
                continue

            min_pair = _min_pairwise(composition, pairwise)
            if min_pair < config.min_pairwise_frequency:
                continue

            observed = _compute_observed_joint(composition, atom_sets, n)
            if observed >= config.max_joint_frequency:
                continue

            expected = _expected_joint(composition, marginal)
            score = math.log(expected / max(observed, epsilon))

            constituents = sorted(composition)
            cid = composition_id(constituents)

            motivating = [
                key for key, atom_set in zip(keys, atom_sets)
                if composition.issubset(atom_set)
            ]

            marg_freqs = {atom: marginal[atom] for atom in composition}
            pair_freqs = {
                _pairwise_key(a, b): pairwise.get(_pairwise_key(a, b), 0.0)
                for i, a in enumerate(constituents)
                for b in constituents[i + 1:]
            }

            proposals.append(CompositionProposal(
                composition_id=cid,
                constituents=constituents,
                marginal_frequencies=marg_freqs,
                pairwise_frequencies=pair_freqs,
                expected_joint=expected,
                observed_joint=observed,
                novelty_score=score,
                motivating_scene_ids=motivating,
                arm=arm,
            ))

    # Sort: novelty_score DESC, composition_id ASC (deterministic tie-breaker)
    proposals.sort(key=lambda p: (-p.novelty_score, p.composition_id))
    return proposals[: config.top_k]


def _compute_observed_joint(
    composition: frozenset[str],
    atom_sets: list[frozenset[str]],
    n: int,
) -> float:
    """Fraction of windows that contain all atoms in the composition."""
    count = sum(1 for atom_set in atom_sets if composition.issubset(atom_set))
    return count / n
