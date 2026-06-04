"""Module: Curator — emergent, versioned taxonomy over immutable evidence.

The curator turns append-only RawDescriptor evidence into a versioned Taxonomy of
CanonicalLabels, then projects evidence onto a version to produce the ephemeral
atoms the hypothesizer reasons over.

THE FIREWALL: this module imports ONLY pipeline.interfaces. It has no import path
to the hypothesizer and can never see novelty scores — labels are decided purely
from descriptor evidence + cohesion, so the system cannot grade its own homework.
(Enforced by tests/test_firewall.py.)

Public surface:
    from pipeline.modules.curator import (
        canonicalize, project, scene_atoms, TaxonomyStore, CuratorConfig,
        drift_metrics, coverage, stability, reprojection_sanity,
    )
"""

from pipeline.modules.curator.canonicalize import canonicalize
from pipeline.modules.curator.config import CuratorConfig, CuratorError
from pipeline.modules.curator.metrics import (
    coverage,
    drift_metrics,
    reprojection_sanity,
    stability,
)
from pipeline.modules.curator.reproject import project, scene_atoms
from pipeline.modules.curator.store import TaxonomyStore

__all__ = [
    "canonicalize",
    "project",
    "scene_atoms",
    "TaxonomyStore",
    "CuratorConfig",
    "CuratorError",
    "drift_metrics",
    "coverage",
    "stability",
    "reprojection_sanity",
]
