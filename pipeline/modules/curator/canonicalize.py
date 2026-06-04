"""Stage 2: Canonicalization — pure, deterministic batch recompute.

    canonicalize(descriptors, base_taxonomy, config) -> Taxonomy

is a pure function: same (descriptors, base taxonomy, seed) -> same taxonomy,
every time. No hidden state, no order dependence beyond the seed. Because it is a
batch recompute (not an online mutation), it is its own correctness oracle: an
eventual online path must produce the same labels as this on the same data.

Determinism note: the seed controls only the PROCESSING ORDER of unmatched
descriptors (greedy clustering is order-sensitive). Fixed seed -> fixed result
(the oracle). Different seeds -> the stability metric measures how much the
labels depend on order, which is the early-warning that thresholds are in an
unstable regime.

Firewall: this module imports ONLY pipeline.interfaces. It has no path to the
hypothesizer and cannot see novelty scores — it decides labels purely from
descriptor evidence + cohesion.
"""

from __future__ import annotations

import hashlib
import re
from typing import Sequence

import numpy as np

from pipeline.interfaces.taxonomy import (
    CanonicalLabel,
    RawDescriptor,
    Taxonomy,
)
from pipeline.modules.curator.config import CuratorConfig


# ---------------------------------------------------------------------------
# Distance helpers (cosine distance in [0, 2])
# ---------------------------------------------------------------------------

def _unit(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    return vec / n if n > 1e-12 else vec


def _cos_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(_unit(a), _unit(b)))


def _seeded_order(descriptors: Sequence[RawDescriptor], seed: int) -> list[RawDescriptor]:
    """Deterministic, seed-dependent processing order.

    Sort by a per-descriptor hash salted with the seed. Same seed -> same order
    (determinism); different seed -> a different but deterministic permutation
    (so the stability metric has something to measure).
    """
    def key(d: RawDescriptor) -> str:
        return hashlib.sha256(f"{seed}|{d.descriptor_id}".encode()).hexdigest()
    return sorted(descriptors, key=key)


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s[:48] or "unnamed"


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------

def canonicalize(
    descriptors: Sequence[RawDescriptor],
    base_taxonomy: Taxonomy,
    config: CuratorConfig = CuratorConfig(),
) -> Taxonomy:
    """Recompute the taxonomy from evidence. Returns a new version.

    Existing labels are carried forward (stable label_id) with centroid/support
    recomputed from current evidence; new concepts are minted only when they pass
    BOTH the support and cohesion guards. Labels are grouped strictly by axis.
    """
    new_version = base_taxonomy.version + 1
    by_axis: dict[str, list[RawDescriptor]] = {}
    for d in descriptors:
        by_axis.setdefault(d.axis, []).append(d)

    base_by_axis = base_taxonomy.labels_by_axis()
    out_labels: list[CanonicalLabel] = []

    for axis in sorted(by_axis.keys()):
        axis_descs = by_axis[axis]
        carried = base_by_axis.get(axis, [])
        out_labels.extend(
            _canonicalize_axis(axis, axis_descs, carried, config, new_version)
        )

    # Carry forward labels for axes that had base labels but no new descriptors
    # this run (their support is recomputed as 0 from current evidence — keep
    # them so the taxonomy is monotonic in concepts, not churny).
    for axis, carried in base_by_axis.items():
        if axis in by_axis:
            continue
        for lab in carried:
            out_labels.append(_with(lab, support=0))

    return Taxonomy(version=new_version, labels=tuple(out_labels), seed=config.seed)


def _canonicalize_axis(
    axis: str,
    descs: list[RawDescriptor],
    carried: list[CanonicalLabel],
    config: CuratorConfig,
    version: int,
) -> list[CanonicalLabel]:
    dim = len(descs[0].embedding) if descs else 0

    # --- 1. assign each descriptor to the nearest carried-forward label -------
    centroids: dict[str, np.ndarray] = {
        lab.label_id: np.asarray(lab.centroid, dtype=float) for lab in carried
    }
    members: dict[str, list[RawDescriptor]] = {lab.label_id: [] for lab in carried}
    unmatched: list[RawDescriptor] = []

    for d in _seeded_order(descs, config.seed):
        emb = np.asarray(d.embedding, dtype=float)
        best_id, best_dist = None, config.match_threshold
        for lid, cen in centroids.items():
            if cen.shape[0] != emb.shape[0]:
                continue
            dist = _cos_dist(emb, cen)
            if dist <= best_dist:  # <= so ties are resolved by iteration order (sorted ids below)
                best_dist, best_id = dist, lid
        if best_id is None:
            unmatched.append(d)
        else:
            members[best_id].append(d)

    # --- 2. cluster the unmatched (greedy, seeded order) ----------------------
    clusters: list[list[RawDescriptor]] = []
    cluster_centroids: list[np.ndarray] = []
    for d in unmatched:  # already in seeded order
        emb = np.asarray(d.embedding, dtype=float)
        best_i, best_dist = None, config.cohesion_threshold
        for i, cen in enumerate(cluster_centroids):
            dist = _cos_dist(emb, cen)
            if dist <= best_dist:
                best_dist, best_i = dist, i
        if best_i is None:
            clusters.append([d])
            cluster_centroids.append(emb.copy())
        else:
            clusters[best_i].append(d)
            cluster_centroids[best_i] = np.mean(
                [np.asarray(x.embedding, dtype=float) for x in clusters[best_i]], axis=0
            )

    # --- 3. mint clusters passing BOTH guards (support AND cohesion) ----------
    minted: list[tuple[str, list[RawDescriptor], np.ndarray]] = []
    for cl in clusters:
        if len(cl) < config.support_threshold:
            continue                              # support guard
        cen = np.mean([np.asarray(x.embedding, dtype=float) for x in cl], axis=0)
        radius = max(_cos_dist(np.asarray(x.embedding, dtype=float), cen) for x in cl)
        if radius > config.cohesion_threshold:
            continue                              # cohesion guard
        name = _mint_name(cl, cen)
        minted.append((name, cl, cen))

    # --- 4. assemble labels (carried + minted), recompute centroid/support ----
    result: list[CanonicalLabel] = []
    for lab in carried:
        mem = members[lab.label_id]
        cen = (np.mean([np.asarray(x.embedding, dtype=float) for x in mem], axis=0)
               if mem else np.asarray(lab.centroid, dtype=float))
        result.append(_with(lab, centroid=tuple(float(x) for x in cen), support=len(mem)))

    # minted ids are deterministic in name; sort by name for stable assembly
    for name, cl, cen in sorted(minted, key=lambda t: t[0]):
        lid = CanonicalLabel.make_id(axis, name, version)
        result.append(CanonicalLabel(
            label_id=lid, axis=axis, name=name,
            centroid=tuple(float(x) for x in cen), support=len(cl),
            version_added=version, parent_id=None,
        ))

    # --- 5. merge labels closer than merge_threshold (older version id wins) --
    return _merge_close(result, config, dim)


def _mint_name(cluster: list[RawDescriptor], centroid: np.ndarray) -> str:
    """Deterministic, canonical-leaning name for a cluster.

    Pick the MOST FREQUENT member phrasing (the wording the model used most often),
    tie-broken by SHORTEST (the most general form: "clear" beats "clear sunny day
    with good visibility"), then lexicographic for determinism. This avoids the
    noisy one-off medoid sentences. (LLM naming is the deferred upgrade.)
    """
    from collections import Counter  # noqa: PLC0415
    counts = Counter(d.text.strip().lower() for d in cluster if d.text.strip())
    if not counts:
        return "unnamed"
    best = min(counts, key=lambda t: (-counts[t], len(t), t))
    return _slug(best)


def _merge_close(labels: list[CanonicalLabel], config: CuratorConfig, dim: int) -> list[CanonicalLabel]:
    """Merge labels whose centroids are within merge_threshold. The label with the
    lower version_added (then lower id) survives, absorbing the other's support.
    Deterministic.
    """
    order = sorted(labels, key=lambda l: (l.version_added, l.label_id))
    survivors: list[CanonicalLabel] = []
    for lab in order:
        cen = np.asarray(lab.centroid, dtype=float)
        merged = False
        for i, s in enumerate(survivors):
            scen = np.asarray(s.centroid, dtype=float)
            if scen.shape[0] == cen.shape[0] and _cos_dist(cen, scen) <= config.merge_threshold:
                total = s.support + lab.support
                if total > 0:
                    blended = (scen * s.support + cen * lab.support) / total
                else:
                    blended = scen
                survivors[i] = _with(s, centroid=tuple(float(x) for x in blended), support=total)
                merged = True
                break
        if not merged:
            survivors.append(lab)
    return survivors


def _with(lab: CanonicalLabel, **changes) -> CanonicalLabel:
    return CanonicalLabel(
        label_id=changes.get("label_id", lab.label_id),
        axis=changes.get("axis", lab.axis),
        name=changes.get("name", lab.name),
        centroid=changes.get("centroid", lab.centroid),
        support=changes.get("support", lab.support),
        version_added=changes.get("version_added", lab.version_added),
        parent_id=changes.get("parent_id", lab.parent_id),
    )
