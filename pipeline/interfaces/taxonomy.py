"""Shared types for the emergent taxonomy system.

Three strictly-separated objects — the spine of the whole design. Conflating any
two of them is a future bug, so the type system keeps them apart:

  RawDescriptor   IMMUTABLE EVIDENCE. What the model said about one scene
                  ("a pedestrian steps off the curb behind a stopped bus").
                  Append-only. Never edited, merged, or regenerated. Carries a
                  pointer back to the reasoning span that justified it, so every
                  structured atom is auditable to its source sentence.

  CanonicalLabel  MUTABLE INTERPRETATION. An emergent bucket that maps many raw
                  descriptors to one atom ("occluded_pedestrian_emergence").
                  Versioned. Persists across runs; only refines over time.

  Projection      EPHEMERAL DERIVATION. descriptor_id -> label_id under one
                  taxonomy version. Always recomputed from current labels,
                  never stored as truth. Stamped with the taxonomy version.

Dependency direction is one-way and enforced structurally:
    evidence  <-  curator output  <-  hypothesizer
Nothing in this file imports a module; modules import this. The curator has NO
import path to the hypothesizer — it decides labels purely from descriptor
evidence and can never see the novelty scores it produces.
"""

from __future__ import annotations

import datetime
import hashlib
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Typed axes — the fixed entity boundaries (the stable backbone)
# ---------------------------------------------------------------------------
# Labels evolve WITHIN an axis; the system never merges across axes (weather is
# never folded into time). Canonicalization always groups/matches by axis, so
# the boundary is structural, not a convention. The set may grow, but an axis
# is an entity type, not an emergent label.
DEFAULT_AXES: frozenset[str] = frozenset({
    "agents",          # objects: vehicle / pedestrian / cyclist ... (open within axis)
    "ego_maneuver",    # what the ego is doing
    "interactions",    # ego/agent/agent relations: "yielding_to_pedestrian", "cut_in"
    "conditions",      # safety-relevant scene conditions
    "road",            # geometry / layout
    "weather",         # from Waymo GT when available
    "time",            # time-of-day, from Waymo GT when available
})


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# RawDescriptor — immutable evidence
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RawDescriptor:
    """One typed observation about one scene. Immutable; append-only.

    descriptor_id is a deterministic content hash, so re-capturing identical
    evidence is idempotent (same id) and append is safe to retry.
    """
    scene_id: str
    axis: str
    text: str                       # the descriptor phrase ("yielding to pedestrian")
    reasoning_span: str             # the sentence/span that justified it (audit pointer)
    embedding: tuple[float, ...]    # text embedding of `text` (tuple => hashable/frozen)
    salience: float = 0.0           # model-judged operational criticality, 0..1 (routine=low).
                                    # LEARNED at extraction, not a hardcoded danger list — so it
                                    # generalizes to unnamed edge cases. NOT part of descriptor_id
                                    # (it's a judgment about the evidence, not its identity).
    descriptor_id: str = ""         # filled in __post_init__ if empty
    created_at: str = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        if not self.descriptor_id:
            digest = hashlib.sha256(
                f"{self.scene_id}|{self.axis}|{self.text}|{self.reasoning_span}".encode()
            ).hexdigest()[:16]
            object.__setattr__(self, "descriptor_id", digest)
        # normalize embedding to a tuple of floats (immutability + stable eq)
        object.__setattr__(self, "embedding", tuple(float(x) for x in self.embedding))

    def to_json(self) -> dict[str, Any]:
        return {
            "descriptor_id": self.descriptor_id,
            "scene_id": self.scene_id,
            "axis": self.axis,
            "text": self.text,
            "reasoning_span": self.reasoning_span,
            "embedding": list(self.embedding),
            "salience": self.salience,
            "created_at": self.created_at,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "RawDescriptor":
        return cls(
            scene_id=str(d["scene_id"]),
            axis=str(d["axis"]),
            text=str(d["text"]),
            reasoning_span=str(d.get("reasoning_span", "")),
            embedding=tuple(float(x) for x in d.get("embedding", [])),
            salience=float(d.get("salience", 0.0)),
            descriptor_id=str(d.get("descriptor_id", "")),
            created_at=str(d.get("created_at", "")),
        )


# ---------------------------------------------------------------------------
# CanonicalLabel — mutable, versioned interpretation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CanonicalLabel:
    """An emergent bucket over raw descriptors. Persists across taxonomy versions
    (stable label_id); its centroid/support are recomputed from evidence each run.

    parent_id is reserved for the general->local hierarchy (flat for now), so the
    store shape never has to change when hierarchy lands.
    """
    label_id: str
    axis: str
    name: str
    centroid: tuple[float, ...]
    support: int                       # number of member descriptors
    version_added: int
    parent_id: str | None = None       # reserved: general->local hierarchy

    @staticmethod
    def make_id(axis: str, name: str, version_added: int) -> str:
        return hashlib.sha256(f"{axis}|{name}|{version_added}".encode()).hexdigest()[:16]

    def to_json(self) -> dict[str, Any]:
        return {
            "label_id": self.label_id,
            "axis": self.axis,
            "name": self.name,
            "centroid": list(self.centroid),
            "support": self.support,
            "version_added": self.version_added,
            "parent_id": self.parent_id,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "CanonicalLabel":
        return cls(
            label_id=str(d["label_id"]),
            axis=str(d["axis"]),
            name=str(d["name"]),
            centroid=tuple(float(x) for x in d.get("centroid", [])),
            support=int(d.get("support", 0)),
            version_added=int(d.get("version_added", 0)),
            parent_id=(str(d["parent_id"]) if d.get("parent_id") else None),
        )


# ---------------------------------------------------------------------------
# Taxonomy — a numbered, immutable snapshot of the label set
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Taxonomy:
    """A versioned snapshot. Each canonicalization run emits a new version; old
    versions are kept so any past statistical picture can be reconstructed.
    """
    version: int
    labels: tuple[CanonicalLabel, ...]
    seed: int = 42
    created_at: str = field(default_factory=_utc_now)

    def labels_by_axis(self) -> dict[str, list[CanonicalLabel]]:
        out: dict[str, list[CanonicalLabel]] = {}
        for lab in self.labels:
            out.setdefault(lab.axis, []).append(lab)
        return out

    def label(self, label_id: str) -> CanonicalLabel | None:
        return next((lab for lab in self.labels if lab.label_id == label_id), None)

    def to_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "seed": self.seed,
            "created_at": self.created_at,
            "labels": [lab.to_json() for lab in self.labels],
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Taxonomy":
        return cls(
            version=int(d["version"]),
            seed=int(d.get("seed", 42)),
            created_at=str(d.get("created_at", "")),
            labels=tuple(CanonicalLabel.from_json(x) for x in d.get("labels", [])),
        )


EMPTY_TAXONOMY = Taxonomy(version=0, labels=())


# ---------------------------------------------------------------------------
# Projection — ephemeral descriptor->label assignment under one version
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Projection:
    """descriptor_id -> label_id (or None = orphan), under taxonomy_version.

    Never persisted as truth — recompute from (descriptors, taxonomy). Stored
    only as a cache/audit artifact, always reconstructible.
    """
    taxonomy_version: int
    assignments: tuple[tuple[str, str | None], ...]   # (descriptor_id, label_id|None)

    def as_dict(self) -> dict[str, str | None]:
        return {d: l for d, l in self.assignments}

    def to_json(self) -> dict[str, Any]:
        return {
            "taxonomy_version": self.taxonomy_version,
            "assignments": [[d, l] for d, l in self.assignments],
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Projection":
        return cls(
            taxonomy_version=int(d["taxonomy_version"]),
            assignments=tuple((str(a[0]), (str(a[1]) if a[1] else None))
                              for a in d.get("assignments", [])),
        )
