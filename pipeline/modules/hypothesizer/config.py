"""Module 3: Hypothesizer — configuration and error types.

SCHEMA_PATH_TO_ATOM_PREFIX maps each SchemaRecord field path (dot-notation)
to the atom prefix used in qualified atom strings ("prefix:value"). lane_count
is excluded — it's numeric, not categorical and can't form a discrete atom.

MULTI_VALUE_FIELDS are list fields where a window can carry multiple values
(e.g., ["car", "pedestrian"]). Same-prefix atom pairs within these fields
are allowed in compositions (two agents co-occurring is meaningful).

SINGLE_CATEGORICAL_FIELDS carry exactly one value per window (the VLM picks
one). Same-prefix pairs from these fields within one composition are forbidden
because a window can't have weather=fog AND weather=rain simultaneously.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Schema path → atom prefix mapping
# ---------------------------------------------------------------------------

SCHEMA_PATH_TO_ATOM_PREFIX: dict[str, str] = {
    "agents":                         "agents",
    "environment.weather":            "weather",
    "environment.time_of_day":        "time_of_day",
    "environment.lighting_condition": "lighting",
    "road.geometry":                  "road_geometry",
    "traffic_control":                "traffic_control",
    "ego_task":                       "ego_task",
    "conditions":                     "conditions",
}

# List fields: a window may have multiple atoms from the same prefix.
MULTI_VALUE_FIELDS: frozenset[str] = frozenset({"agents", "conditions"})

# Scalar fields: exactly one atom per prefix per window.
# A composition may not contain two atoms with the same prefix from these.
SINGLE_CATEGORICAL_FIELDS: frozenset[str] = frozenset({
    "weather",
    "time_of_day",
    "lighting",
    "road_geometry",
    "traffic_control",
    "ego_task",
})


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class HypothesizerError(Exception):
    """Base class for Module 3: Hypothesizer errors."""


class HypothesizerEmptyInputError(HypothesizerError):
    """Raised when propose() receives an empty or all-skipped record list."""


class VocabularyMismatchError(HypothesizerError):
    """Raised when a SchemaRecord atom value is not in valid_atoms.

    Strict equality — no normalization. Callers must pass clean SchemaRecords
    that passed encoder vocabulary validation. If this error appears in
    production, the encoder's vocabulary and the hypothesizer's valid_atoms
    are out of sync.
    """
    def __init__(self, atom: str, window_id: str) -> None:
        self.atom = atom
        self.window_id = window_id
        super().__init__(
            f"[Hypothesizer] VocabularyMismatchError: atom {atom!r} "
            f"not in valid_atoms (window {window_id}). "
            f"Ensure encoder vocabulary and hypothesizer valid_atoms are in sync."
        )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class HypothesizerConfig:
    """Configuration for Module 3: Hypothesizer.

    Parameters
    ----------
    min_marginal_frequency
        An atom must appear in at least this fraction of windows to be eligible
        as a composition constituent. Prevents long-tail singletons from
        dominating proposals.

    max_joint_frequency
        A composition must appear in fewer than this fraction of windows.
        Rejects already-common combinations — novelty requires joint rarity.

    min_pairwise_frequency
        Every pair of atoms within a composition must co-occur in at least this
        fraction of windows. Guards against physically implausible pairings
        (e.g., an agent that was never observed under a given condition).

    composition_sizes
        Arities to enumerate. [2, 3, 4] covers pairs, triples, and quads.
        Warning: enumeration is O(n^k) in atoms × arity. For k=4 and n~50
        atoms, this is ~125K candidates before filtering — acceptable. k≥5
        on large atom sets is inadvisable without pre-filtering.

    top_k
        Maximum proposals returned. Ranked by novelty_score DESC, with
        composition_id ASC as the deterministic tie-breaker.

    compose_over
        Atom prefixes to include in composition. None = all fields (default).
        Pass ["conditions"] for a conservative v0 run that replicates the
        conditions-only baseline.

    valid_atoms
        If provided, any qualified atom not in this set raises
        VocabularyMismatchError immediately (strict, no normalization).
        Build from DEFAULT_VOCABULARY for production enforcement.
        Note: compose_over and valid_atoms are independent filters. compose_over
        limits which fields produce atoms; valid_atoms validates the atom values.
        If compose_over=["conditions"] and valid_atoms covers all fields, only
        conditions atoms are extracted (and validated). This is intentional.
    """
    min_marginal_frequency: float = 0.05
    max_joint_frequency: float = 0.005
    min_pairwise_frequency: float = 0.01
    composition_sizes: list[int] = field(default_factory=lambda: [2, 3, 4])
    top_k: int = 30
    compose_over: list[str] | None = None       # None = all fields
    valid_atoms: frozenset[str] | None = None   # None = no validation
