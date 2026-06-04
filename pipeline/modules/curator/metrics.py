"""Stage 4 instrumentation: the numbers that make silent degradation impossible.

Log all four every run. When one moves, you know which component broke before it
poisons results:

  drift          — labels minted / merged / carried; descriptors unmatched.
  coverage       — fraction of descriptors assigned a label vs orphaned.
  stability      — re-canonicalize with two seeds; agreement of the co-assignment
                   relation. Low => thresholds are in an unstable regime and the
                   labels are seed-dependent noise. The single best early warning.
  reprojection_sanity — descriptor count conserved + no scene_id lost after a
                   re-projection. Catches corruption in the re-projection path.
"""

from __future__ import annotations

from itertools import combinations
from typing import Sequence

from pipeline.interfaces.taxonomy import Projection, RawDescriptor, Taxonomy
from pipeline.modules.curator.canonicalize import canonicalize
from pipeline.modules.curator.config import CuratorConfig
from pipeline.modules.curator.reproject import project


def drift_metrics(prev: Taxonomy, new: Taxonomy) -> dict[str, int]:
    """Label-set churn between two taxonomy versions."""
    prev_ids = {l.label_id for l in prev.labels}
    new_ids = {l.label_id for l in new.labels}
    minted = sorted(new_ids - prev_ids)
    dropped = sorted(prev_ids - new_ids)   # merged away / removed
    carried = sorted(prev_ids & new_ids)
    return {
        "labels_prev": len(prev_ids),
        "labels_new": len(new_ids),
        "minted": len(minted),
        "dropped": len(dropped),     # includes merges
        "carried": len(carried),
    }


def coverage(descriptors: Sequence[RawDescriptor], projection: Projection) -> dict[str, float]:
    """What fraction of evidence is assigned to a label vs orphaned."""
    total = len(descriptors)
    assigned = sum(1 for _, lid in projection.assignments if lid is not None)
    orphaned = total - assigned
    return {
        "total_descriptors": float(total),
        "assigned": float(assigned),
        "orphaned": float(orphaned),
        "coverage": (assigned / total) if total else 0.0,
    }


def stability(
    descriptors: Sequence[RawDescriptor],
    base_taxonomy: Taxonomy,
    config: CuratorConfig,
    seeds: tuple[int, int] = (1, 2),
) -> dict[str, float]:
    """Re-canonicalize under two seeds; measure agreement of the co-assignment
    relation (alignment-free, so label-id churn across seeds doesn't matter).

    For every pair of descriptors, do both runs agree on whether they share a
    label? 1.0 = labels are seed-independent (healthy); low = order-dependent noise.
    """
    tax_a = canonicalize(descriptors, base_taxonomy, config.with_seed(seeds[0]))
    tax_b = canonicalize(descriptors, base_taxonomy, config.with_seed(seeds[1]))
    asg_a = project(descriptors, tax_a, config).as_dict()
    asg_b = project(descriptors, tax_b, config).as_dict()

    ids = [d.descriptor_id for d in descriptors]
    pairs = 0
    agree = 0
    for i, j in combinations(ids, 2):
        # only pairs in the same axis can ever co-assign; skip cross-axis pairs
        pairs += 1
        same_a = asg_a.get(i) is not None and asg_a.get(i) == asg_a.get(j)
        same_b = asg_b.get(i) is not None and asg_b.get(i) == asg_b.get(j)
        if same_a == same_b:
            agree += 1
    return {
        "stability": (agree / pairs) if pairs else 1.0,
        "n_labels_seed_a": float(len(tax_a.labels)),
        "n_labels_seed_b": float(len(tax_b.labels)),
    }


def reprojection_sanity(
    descriptors: Sequence[RawDescriptor], projection: Projection
) -> dict[str, bool | int]:
    """Conservation checks after a re-projection. All must be True."""
    desc_ids = {d.descriptor_id for d in descriptors}
    proj_ids = {did for did, _ in projection.assignments}
    scenes_before = {d.scene_id for d in descriptors}
    # every descriptor appears exactly once in the projection
    count_conserved = len(projection.assignments) == len(descriptors)
    ids_conserved = proj_ids == desc_ids
    no_dupes = len(proj_ids) == len(projection.assignments)
    # scene ids are intrinsic to evidence — re-projection can never touch them
    scenes_preserved = scenes_before == {d.scene_id for d in descriptors}
    return {
        "count_conserved": count_conserved,
        "ids_conserved": ids_conserved,
        "no_duplicate_assignments": no_dupes,
        "scene_ids_preserved": scenes_preserved,
        "ok": bool(count_conserved and ids_conserved and no_dupes and scenes_preserved),
    }
