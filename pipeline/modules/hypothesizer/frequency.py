"""Module 3: Hypothesizer — atom extraction and frequency counting.

Extracts qualified atoms ("prefix:value") from SchemaRecord.fields dicts,
then computes marginal and pairwise atom frequencies across a record set.

Design notes:
- Atoms are always "prefix:value" strings. No bare values, no ambiguity.
- Extraction is strict: values must match valid_atoms exactly if provided.
  No normalization (no .lower(), no .strip()) — encoder output is authoritative.
- lane_count is skipped: numeric, not categorical.
- Missing or null fields produce zero atoms for that field — not an error.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from typing import Any

from pipeline.modules.hypothesizer.config import (
    SCHEMA_PATH_TO_ATOM_PREFIX,
    MULTI_VALUE_FIELDS,
    VocabularyMismatchError,
)


def extract_atoms(
    fields: dict[str, Any],
    compose_over: list[str] | None,
    valid_atoms: frozenset[str] | None,
    window_id: str,
) -> frozenset[str]:
    """Extract all qualified atoms from one SchemaRecord.fields dict.

    Parameters
    ----------
    fields
        The fields dict from a SchemaRecord (must have succeeded, i.e.
        failure_mode is None).
    compose_over
        Atom prefix whitelist. None = all prefixes (default).
    valid_atoms
        If provided, every extracted atom must appear here or
        VocabularyMismatchError is raised immediately.
    window_id
        Used only in error messages.

    Returns
    -------
    frozenset[str]
        Qualified atom strings for this window (e.g. {"agents:car",
        "weather:clear", "road_geometry:intersection"}).
    """
    atoms: list[str] = []

    for schema_path, prefix in SCHEMA_PATH_TO_ATOM_PREFIX.items():
        if compose_over is not None and prefix not in compose_over:
            continue

        value = _extract_field(fields, schema_path)
        if value is None:
            continue  # missing / null — skip silently (no atom emitted)

        if prefix in MULTI_VALUE_FIELDS:
            # list field: one atom per list element
            if not isinstance(value, list):
                continue
            for v in value:
                atom = f"{prefix}:{v}"
                _check_atom(atom, valid_atoms, window_id)
                atoms.append(atom)
        else:
            # scalar field: one atom
            if not isinstance(value, str):
                continue
            atom = f"{prefix}:{value}"
            _check_atom(atom, valid_atoms, window_id)
            atoms.append(atom)

    return frozenset(atoms)


def _extract_field(fields: dict[str, Any], path: str) -> Any:
    """Navigate a dot-notation path into the fields dict."""
    parts = path.split(".", 1)
    if len(parts) == 1:
        return fields.get(path)
    top, rest = parts
    sub = fields.get(top)
    if not isinstance(sub, dict):
        return None
    return sub.get(rest)


def _check_atom(atom: str, valid_atoms: frozenset[str] | None, window_id: str) -> None:
    if valid_atoms is not None and atom not in valid_atoms:
        raise VocabularyMismatchError(atom, window_id)


# ---------------------------------------------------------------------------
# Frequency tables
# ---------------------------------------------------------------------------

def compute_frequencies(
    atom_sets: list[frozenset[str]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute marginal and pairwise atom frequencies.

    Parameters
    ----------
    atom_sets
        One frozenset[str] per window (only windows that passed filtering).

    Returns
    -------
    marginal : dict[str, float]
        Fraction of windows containing each atom.
    pairwise : dict[str, float]
        Fraction of windows containing each ordered pair.
        Key format: "atom_a|atom_b" where atom_a < atom_b lexicographically.
    """
    n = len(atom_sets)
    if n == 0:
        return {}, {}

    marginal_counts: dict[str, int] = defaultdict(int)
    pairwise_counts: dict[str, int] = defaultdict(int)

    for atom_set in atom_sets:
        atoms_sorted = sorted(atom_set)

        for atom in atoms_sorted:
            marginal_counts[atom] += 1

        for i, a in enumerate(atoms_sorted):
            for b in atoms_sorted[i + 1:]:
                key = f"{a}|{b}"
                pairwise_counts[key] += 1

    marginal = {atom: count / n for atom, count in marginal_counts.items()}
    pairwise = {key: count / n for key, count in pairwise_counts.items()}
    return marginal, pairwise
