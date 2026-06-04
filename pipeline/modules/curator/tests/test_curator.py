"""Curator tests — pin the fool-proofing, not just the happy path."""

from __future__ import annotations

import math

from pipeline.interfaces.taxonomy import (
    EMPTY_TAXONOMY,
    CanonicalLabel,
    Projection,
    RawDescriptor,
    Taxonomy,
)
from pipeline.modules.curator import (
    CuratorConfig,
    canonicalize,
    coverage,
    drift_metrics,
    project,
    reprojection_sanity,
    scene_atoms,
    stability,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def desc(scene: str, axis: str, text: str, emb: list[float]) -> RawDescriptor:
    return RawDescriptor(scene_id=scene, axis=axis, text=text,
                         reasoning_span=f"...{text}...", embedding=tuple(emb))


def _dir(angle_deg: float) -> list[float]:
    r = math.radians(angle_deg)
    return [math.cos(r), math.sin(r)]


# ---------------------------------------------------------------------------
# interface round-trips
# ---------------------------------------------------------------------------

def test_interface_roundtrips():
    d = desc("s1", "interactions", "yielding to pedestrian", [1.0, 0.0])
    assert RawDescriptor.from_json(d.to_json()) == d
    lab = CanonicalLabel(label_id="l1", axis="interactions", name="yield",
                         centroid=(1.0, 0.0), support=3, version_added=1)
    assert CanonicalLabel.from_json(lab.to_json()) == lab
    tax = Taxonomy(version=1, labels=(lab,), seed=7)
    assert Taxonomy.from_json(tax.to_json()) == tax
    proj = Projection(taxonomy_version=1, assignments=(("a", "l1"), ("b", None)))
    assert Projection.from_json(proj.to_json()) == proj


def test_descriptor_id_is_deterministic_and_idempotent():
    a = desc("s1", "agents", "a red car", [1.0, 0.0])
    b = desc("s1", "agents", "a red car", [1.0, 0.0])
    assert a.descriptor_id == b.descriptor_id  # identical evidence -> same id


# ---------------------------------------------------------------------------
# determinism oracle: same (descriptors, base, seed) -> same taxonomy
# ---------------------------------------------------------------------------

def test_canonicalize_is_deterministic():
    ds = [desc(f"s{i}", "interactions", f"yield {i}", _dir(i)) for i in range(6)]
    cfg = CuratorConfig(seed=42)
    t1 = canonicalize(ds, EMPTY_TAXONOMY, cfg)
    t2 = canonicalize(ds, EMPTY_TAXONOMY, cfg)
    assert t1.to_json()["labels"] == t2.to_json()["labels"]


# ---------------------------------------------------------------------------
# the dual mint guard: support AND cohesion both required
# ---------------------------------------------------------------------------

def test_support_guard():
    # 4 tight descriptors near angle 0 -> one cohesive cluster.
    ds = [desc(f"s{i}", "interactions", f"yield {i}", _dir(i * 1.0)) for i in range(4)]
    minted_3 = canonicalize(ds, EMPTY_TAXONOMY, CuratorConfig(support_threshold=3)).labels
    minted_5 = canonicalize(ds, EMPTY_TAXONOMY, CuratorConfig(support_threshold=5)).labels
    assert len(minted_3) == 1   # support 4 >= 3 -> minted
    assert len(minted_5) == 0   # support 4 <  5 -> rejected (support guard)


def test_cohesion_guard_controls_granularity():
    # Two tight pairs ~53 deg apart (cosine distance ~0.4 between the pairs).
    ds = (
        [desc(f"a{i}", "interactions", f"slow {i}", _dir(0 + i)) for i in range(2)]
        + [desc(f"b{i}", "interactions", f"stop {i}", _dir(53 + i)) for i in range(2)]
    )
    # cohesion 0.30 < 0.4 gap -> two separate clusters of 2 -> each below support -> 0 minted.
    tight = canonicalize(ds, EMPTY_TAXONOMY, CuratorConfig(cohesion_threshold=0.30, support_threshold=3))
    # cohesion 0.55 > 0.4 gap -> one cluster of 4 -> >= support -> 1 minted.
    loose = canonicalize(ds, EMPTY_TAXONOMY, CuratorConfig(cohesion_threshold=0.55, support_threshold=3))
    assert len(tight.labels) == 0
    assert len(loose.labels) == 1


# ---------------------------------------------------------------------------
# axis boundary: never merge across entity types
# ---------------------------------------------------------------------------

def test_axis_boundary_is_structural():
    # Identical embeddings, different axes -> must never share a label.
    ds = [
        desc("s1", "weather", "rain", [1.0, 0.0]),
        desc("s2", "weather", "rain", [1.0, 0.0]),
        desc("s3", "weather", "rain", [1.0, 0.0]),
        desc("s4", "interactions", "yield", [1.0, 0.0]),
        desc("s5", "interactions", "yield", [1.0, 0.0]),
        desc("s6", "interactions", "yield", [1.0, 0.0]),
    ]
    tax = canonicalize(ds, EMPTY_TAXONOMY, CuratorConfig(support_threshold=3))
    axes = {lab.axis for lab in tax.labels}
    assert axes == {"weather", "interactions"}            # one label per axis, not merged
    proj = project(ds, tax)
    asg = proj.as_dict()
    # a weather descriptor and an interactions descriptor never share a label id
    assert asg[ds[0].descriptor_id] != asg[ds[3].descriptor_id]


# ---------------------------------------------------------------------------
# re-projection conservation
# ---------------------------------------------------------------------------

def test_reprojection_conserves_evidence():
    ds = [desc(f"s{i}", "agents", "car", _dir(i)) for i in range(5)]
    tax = canonicalize(ds, EMPTY_TAXONOMY, CuratorConfig(support_threshold=3))
    proj = project(ds, tax)
    sanity = reprojection_sanity(ds, proj)
    assert sanity["ok"] is True
    cov = coverage(ds, proj)
    assert cov["assigned"] + cov["orphaned"] == cov["total_descriptors"]


# ---------------------------------------------------------------------------
# stability: well-separated clusters are seed-independent
# ---------------------------------------------------------------------------

def test_stability_high_for_separated_clusters():
    # three orthogonal clusters of 3 in 3-D -> clearly separated.
    clusters = {
        "agents": [[1, 0, 0], [1, 0.03, 0], [0.99, 0, 0.02]],
        "interactions": [[0, 1, 0], [0, 1, 0.02], [0.02, 0.99, 0]],
        "conditions": [[0, 0, 1], [0, 0.02, 1], [0, 0, 0.99]],
    }
    ds = []
    for axis, embs in clusters.items():
        for i, e in enumerate(embs):
            ds.append(desc(f"{axis}{i}", axis, f"{axis}_{i}", e))
    s = stability(ds, EMPTY_TAXONOMY, CuratorConfig(support_threshold=3), seeds=(1, 2))
    assert s["stability"] == 1.0


# ---------------------------------------------------------------------------
# scene atoms + drift
# ---------------------------------------------------------------------------

def test_scene_atoms_and_drift():
    ds = [desc(f"s{i}", "interactions", "yield", _dir(i)) for i in range(4)]
    t0 = EMPTY_TAXONOMY
    t1 = canonicalize(ds, t0, CuratorConfig(support_threshold=3))
    proj = project(ds, t1)
    atoms = scene_atoms(ds, proj, t1)
    # every scene that had a descriptor in the minted cluster has exactly one atom
    assert all(len(a) == 1 for a in atoms.values())
    d = drift_metrics(t0, t1)
    assert d["minted"] == 1 and d["labels_new"] == 1
