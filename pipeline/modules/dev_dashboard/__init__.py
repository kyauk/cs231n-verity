"""Module 7: Dev Dashboard — private developer-facing evaluation surface.

Two tabs, two evaluations:
  - VLM Accuracy: per-field diff between a hand-labeled gold set and the
    encoder's schema_records.json output. Upload-only, no statistics in-UI.
  - Discrimination Test: blinded human ratings against three sample pools
    (Verity / Random / Naive-rare). Per-round filesystem layout. Export
    reveals the source pool of each rating for offline Mann-Whitney analysis.

Unlike judge_ui (customer-facing), this module is run only by the operator
and refuses to start unless `VERITY_DEV_MODE=1` is set in the environment.

Public surface (import from the package root):
    from pipeline.modules.dev_dashboard import sample_three_pools, compute_diff

Endpoints live in pipeline.modules.dev_dashboard.server (run via uvicorn).
"""

from pipeline.modules.dev_dashboard.accuracy import (
    ALL_FIELDS,
    AccuracyDiffError,
    AccuracyReport,
    FieldDiff,
    MissingEntry,
    WindowDiff,
    compute_diff,
    gold_template,
)
from pipeline.modules.dev_dashboard.sampling import (
    SampleResult,
    SamplingError,
    sample_three_pools,
)

__all__ = [
    # Sampling
    "SampleResult",
    "SamplingError",
    "sample_three_pools",
    # Accuracy
    "AccuracyReport",
    "AccuracyDiffError",
    "FieldDiff",
    "WindowDiff",
    "MissingEntry",
    "compute_diff",
    "gold_template",
    "ALL_FIELDS",
]
