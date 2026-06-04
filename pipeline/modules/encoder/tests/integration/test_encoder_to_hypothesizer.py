"""Integration test: Encoder (Module 2) → Hypothesizer (Module 3) boundary.

Tests the full boundary crossing: Encoder.process_batch → list[SchemaRecord]
→ Hypothesizer.propose → list[CompositionProposal].

Previously used a _StubHypothesizer. Module 3 is now built; tests use the
real Hypothesizer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from pipeline.interfaces.proposal import CompositionProposal
from pipeline.interfaces.schema_record import SchemaRecord
from pipeline.interfaces.window import WindowKey
from pipeline.modules.encoder.vocabulary import DEFAULT_VOCABULARY
from pipeline.modules.hypothesizer.config import HypothesizerConfig, HypothesizerEmptyInputError
from pipeline.modules.hypothesizer.hypothesizer import Hypothesizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_storage(pose_summary: str | None = "Ego vehicle at intersection.") -> Any:
    manifest = MagicMock()
    manifest.pose_summary = pose_summary
    storage = MagicMock()
    storage.get_window_video_url.return_value = "https://example.com/seg.mp4"
    storage.get_window_manifest.return_value = manifest
    return storage


def _make_encoder(tmp_path: Path):
    from pipeline.modules.encoder.encoder import Encoder
    from pipeline.modules.encoder.reasoning_arm import StubVLMClient
    return Encoder(vlm=StubVLMClient(), vocabulary=DEFAULT_VOCABULARY, cache_root=tmp_path)


def _make_window(segment_id: str, window_idx: int, storage: Any = None):
    from pipeline.modules.encoder.schema import WindowInput
    return WindowInput(
        segment_id=segment_id,
        window_idx=window_idx,
        storage=storage or _make_storage(),
    )


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

def test_encoder_output_consumed_by_hypothesizer(tmp_path: Path) -> None:
    """Encoder batch → Hypothesizer: full boundary crossing."""
    enc = _make_encoder(tmp_path)
    windows = [_make_window("seg_integ", i) for i in range(20)]
    records = enc.process_batch(windows)

    hyp = Hypothesizer(HypothesizerConfig(
        min_marginal_frequency=0.0,
        max_joint_frequency=1.1,
        min_pairwise_frequency=0.0,
        composition_sizes=[2],
        top_k=5,
    ))
    proposals = hyp.propose(records, arm="reasoning")
    assert isinstance(proposals, list)
    for p in proposals:
        assert isinstance(p, CompositionProposal)


def test_hypothesizer_skips_failed_records(tmp_path: Path) -> None:
    """Hypothesizer raises HypothesizerEmptyInputError for all-failed records."""
    from pipeline.modules.encoder.reasoning_arm import VLMUnavailableError
    from pipeline.modules.encoder.schema import FAILURE_VLM_UNAVAILABLE
    from pipeline.modules.encoder.encoder import Encoder

    class DeadVLM:
        model_id = "stub/dead"
        def complete(self, video_url: str, prompt: str) -> str:
            raise VLMUnavailableError("stub/dead", "refused")

    enc = Encoder(vlm=DeadVLM(), vocabulary=DEFAULT_VOCABULARY, cache_root=tmp_path)
    records = enc.process_batch([_make_window("seg_dead", i) for i in range(3)])

    assert all(r.failure_mode == FAILURE_VLM_UNAVAILABLE for r in records)
    hyp = Hypothesizer()
    with pytest.raises(HypothesizerEmptyInputError):
        hyp.propose(records, arm="reasoning")


def test_mixed_success_failure_batch(tmp_path: Path) -> None:
    """Batch with one failure: Hypothesizer processes only the 2 successes."""
    from pipeline.modules.encoder.encoder import Encoder
    from pipeline.modules.encoder.reasoning_arm import StubVLMClient, VLMUnavailableError

    call_n = 0

    class MostlyGoodVLM:
        model_id = "stub/mostly-good"
        def complete(self, video_url: str, prompt: str) -> str:
            nonlocal call_n
            call_n += 1
            if call_n == 2:
                raise VLMUnavailableError("stub/mostly-good", "blip")
            return StubVLMClient().complete(video_url, prompt)

    enc = Encoder(vlm=MostlyGoodVLM(), vocabulary=DEFAULT_VOCABULARY, cache_root=tmp_path)
    records = enc.process_batch([_make_window("seg_mixed", i) for i in range(3)])

    assert sum(1 for r in records if r.succeeded) == 2
    assert sum(1 for r in records if not r.succeeded) == 1

    # Only 2 good records — too few for composition with default thresholds,
    # but HypothesizerEmptyInputError should NOT be raised (records exist).
    hyp = Hypothesizer(HypothesizerConfig(
        min_marginal_frequency=0.0,
        max_joint_frequency=1.1,
        min_pairwise_frequency=0.0,
        composition_sizes=[2],
        top_k=5,
    ))
    # May return empty list (no compositions pass filters with 2 windows) but must not raise.
    proposals = hyp.propose(records, arm="reasoning")
    assert isinstance(proposals, list)


def test_window_id_survives_boundary_crossing(tmp_path: Path) -> None:
    """window_id identity is preserved through serialisation + boundary crossing."""
    enc = _make_encoder(tmp_path)
    window = _make_window("boundary_seg", 42)
    record = enc.process(window)[0]

    # Simulate what happens at a service boundary: to_json → from_json
    wire = record.to_json()
    restored = SchemaRecord.from_json(wire)

    assert isinstance(restored.window_id, WindowKey)
    assert str(restored.window_id) == "boundary_seg/0042"
