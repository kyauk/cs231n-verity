"""Cross-module integration test: FlatMP4Storage → Encoder.

Proves the boundary works: FlatMP4Storage is consumable by the same Encoder
that consumes the canonical WindowStorage, with no special-casing in the
Encoder. Uses StubVLMClient so no NIM network call.

This is the test the hygiene protocol describes as catching cross-module
drift at build time rather than week 3. If FlatMP4Storage ever produces
something the Encoder can't read, this test fails immediately.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from pipeline.interfaces.schema_record import SchemaRecord
from pipeline.modules.encoder import (
    DEFAULT_VOCABULARY,
    Encoder,
    StubVLMClient,
    WindowInput,
)
from pipeline.modules.storage import FlatMP4Storage


# ---------------------------------------------------------------------------
# Test fixture builder
# ---------------------------------------------------------------------------

def _build_storage_with_n_segments(n: int) -> FlatMP4Storage:
    """FlatMP4Storage with n synthetic MP4 blobs, GCS fully mocked."""
    s = FlatMP4Storage(bucket_uri="gs://it-test/clips", cameras=["FRONT"])
    blobs = []
    for i in range(n):
        b = MagicMock()
        b.name = f"clips/seg_{i:03d}.mp4"
        b.exists.return_value = True
        b.reload.return_value = None
        b.size = 100_000
        b.content_type = "video/mp4"
        b.generate_signed_url.return_value = f"https://signed/{i}.mp4"
        blobs.append(b)
    fake_bucket = MagicMock()
    fake_bucket.list_blobs.return_value = blobs
    fake_bucket.blob.side_effect = lambda name: next(
        (b for b in blobs if b.name == name), MagicMock(exists=lambda: False)
    )
    s._bucket_obj = fake_bucket
    return s


# ---------------------------------------------------------------------------
# Integration test: full Encoder → SchemaRecord pipeline
# ---------------------------------------------------------------------------

def test_encoder_consumes_flat_mp4_storage_end_to_end(tmp_path) -> None:
    """The Encoder processes FlatMP4Storage windows and emits valid SchemaRecords."""
    storage = _build_storage_with_n_segments(3)
    encoder = Encoder(
        vlm=StubVLMClient(),
        vocabulary=DEFAULT_VOCABULARY,
        cache_root=tmp_path,
        visual_arm=None,
    )

    inputs = [
        WindowInput(segment_id=w.segment_id, window_idx=w.window_idx, storage=storage)
        for w in storage.list_windows()
    ]
    records = encoder.process_batch(inputs)

    # Boundary assertion 1: one record per input
    assert len(records) == 3

    # Boundary assertion 2: every record is a SchemaRecord with the right shape
    assert all(isinstance(r, SchemaRecord) for r in records)
    assert all(r.arm == "reasoning" for r in records)
    assert all(r.schema_version for r in records)
    assert all(r.window_id.window_idx == 0 for r in records)
    assert {r.window_id.segment_id for r in records} == {"seg_000", "seg_001", "seg_002"}

    # Boundary assertion 3: stub returns a deterministic valid annotation
    for r in records:
        assert r.succeeded, f"unexpected failure_mode: {r.failure_mode}"


def test_encoder_reads_pose_summary_from_synthesized_manifest(tmp_path) -> None:
    """FlatMP4Storage synthesizes pose_summary=None.

    The encoder must handle that (the reasoning prompt should still build).
    """
    storage = _build_storage_with_n_segments(1)
    encoder = Encoder(
        vlm=StubVLMClient(),
        vocabulary=DEFAULT_VOCABULARY,
        cache_root=tmp_path,
        visual_arm=None,
    )
    inp = WindowInput(segment_id="seg_000", window_idx=0, storage=storage)
    rec = encoder.process(inp)[0]
    assert rec.succeeded  # pose_summary=None did not break the prompt build


def test_record_round_trips_through_interfaces(tmp_path) -> None:
    """Every SchemaRecord the Encoder produces from FlatMP4Storage must
    survive JSON serialization — this is what downstream modules
    (Hypothesizer, Scorer, Evaluation) consume."""
    storage = _build_storage_with_n_segments(1)
    encoder = Encoder(
        vlm=StubVLMClient(),
        vocabulary=DEFAULT_VOCABULARY,
        cache_root=tmp_path,
        visual_arm=None,
    )
    rec = encoder.process(
        WindowInput(segment_id="seg_000", window_idx=0, storage=storage)
    )[0]
    restored = SchemaRecord.from_json(rec.to_json())
    assert restored.window_id == rec.window_id
    assert restored.arm == rec.arm
    assert restored.fields == rec.fields
    assert restored.failure_mode == rec.failure_mode
