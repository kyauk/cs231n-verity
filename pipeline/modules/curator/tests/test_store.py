"""Store invariants: append-only evidence, immutable taxonomy versions."""

from __future__ import annotations

import pytest

from pipeline.interfaces.taxonomy import CanonicalLabel, RawDescriptor, Taxonomy
from pipeline.modules.curator import TaxonomyStore


def _d(scene: str, text: str) -> RawDescriptor:
    return RawDescriptor(scene_id=scene, axis="agents", text=text,
                         reasoning_span="x", embedding=(1.0, 0.0))


def test_evidence_append_is_idempotent(tmp_path):
    store = TaxonomyStore(tmp_path)
    ds = [_d("s1", "car"), _d("s2", "bus")]
    assert store.append_descriptors(ds) == 2
    assert store.append_descriptors(ds) == 0          # identical evidence -> no dupes
    assert store.append_descriptors([_d("s3", "van")]) == 1
    assert len(store.load_descriptors()) == 3


def test_taxonomy_versions_are_immutable(tmp_path):
    store = TaxonomyStore(tmp_path)
    lab = CanonicalLabel(label_id="l1", axis="agents", name="car",
                         centroid=(1.0, 0.0), support=3, version_added=1)
    t1 = Taxonomy(version=1, labels=(lab,))
    store.save_taxonomy(t1)
    assert store.latest_version() == 1
    assert store.load_taxonomy() == t1
    with pytest.raises(FileExistsError):
        store.save_taxonomy(t1)                       # versions never overwritten


def test_load_missing_returns_none(tmp_path):
    store = TaxonomyStore(tmp_path)
    assert store.load_taxonomy() is None
    assert store.latest_version() is None
