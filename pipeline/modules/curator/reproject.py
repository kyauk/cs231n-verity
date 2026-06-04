"""Stage 3: Re-projection — pure mapping of evidence onto a taxonomy version.

    project(descriptors, taxonomy, config) -> Projection
    scene_atoms(descriptors, projection, taxonomy) -> {scene_id: set[atom]}

Re-projection is what makes "the taxonomy evolves but old data stays comparable"
actually hold: when the taxonomy changes, you re-project ALL historical evidence
onto the new version before recomputing any statistics. Because it is pure and
the evidence is immutable, you can reconstruct the exact statistical picture at
any taxonomy version, and nothing is irreversibly mutated.

A descriptor that matches no label within match_threshold is an ORPHAN (label
None) — surfaced by the coverage metric, never silently dropped.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from pipeline.interfaces.taxonomy import (
    CanonicalLabel,
    Projection,
    RawDescriptor,
    Taxonomy,
)
from pipeline.modules.curator.canonicalize import _cos_dist
from pipeline.modules.curator.config import CuratorConfig


def project(
    descriptors: Sequence[RawDescriptor],
    taxonomy: Taxonomy,
    config: CuratorConfig = CuratorConfig(),
) -> Projection:
    """Assign each descriptor to its nearest label IN ITS AXIS, or orphan."""
    by_axis: dict[str, list[CanonicalLabel]] = taxonomy.labels_by_axis()
    cen_by_axis: dict[str, list[tuple[str, np.ndarray]]] = {
        axis: [(lab.label_id, np.asarray(lab.centroid, dtype=float)) for lab in labs]
        for axis, labs in by_axis.items()
    }

    assignments: list[tuple[str, str | None]] = []
    for d in descriptors:
        emb = np.asarray(d.embedding, dtype=float)
        best_id, best_dist = None, config.match_threshold
        for lid, cen in cen_by_axis.get(d.axis, []):
            if cen.shape[0] != emb.shape[0]:
                continue
            dist = _cos_dist(emb, cen)
            if dist <= best_dist:
                best_dist, best_id = dist, lid
        assignments.append((d.descriptor_id, best_id))

    # Deterministic order (by descriptor_id) so the Projection is comparable.
    assignments.sort(key=lambda a: a[0])
    return Projection(taxonomy_version=taxonomy.version, assignments=tuple(assignments))


def scene_atoms(
    descriptors: Sequence[RawDescriptor],
    projection: Projection,
    taxonomy: Taxonomy,
) -> dict[str, set[str]]:
    """The composition input: per scene, the set of qualified atoms
    ("axis:label_name") present. This is the ephemeral derivation the
    hypothesizer reasons over — recomputed, never stored as truth.
    """
    label_name = {lab.label_id: f"{lab.axis}:{lab.name}" for lab in taxonomy.labels}
    desc_scene = {d.descriptor_id: d.scene_id for d in descriptors}
    assigned = projection.as_dict()

    out: dict[str, set[str]] = {d.scene_id: set() for d in descriptors}
    for did, lid in assigned.items():
        if lid is None:
            continue
        scene = desc_scene.get(did)
        atom = label_name.get(lid)
        if scene is not None and atom is not None:
            out[scene].add(atom)
    return out
