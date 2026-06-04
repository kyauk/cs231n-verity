"""Concurrency tests for Module 2: Encoder.

Verifies thread safety of process(), correctness of concurrent process_batch(),
and independence of the visual arm.

All tests run without real VLM or GCS connections.
"""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_storage() -> Any:
    manifest = MagicMock()
    manifest.pose_summary = "Vehicle traveled 10m."
    storage = MagicMock()
    storage.get_window_video_url.return_value = "https://example.com/video.mp4"
    storage.get_window_manifest.return_value = manifest
    return storage


def _make_encoder(tmp_path: Path, max_workers: int = 4):
    from pipeline.modules.encoder.encoder import Encoder
    from pipeline.modules.encoder.reasoning_arm import StubVLMClient
    from pipeline.modules.encoder.vocabulary import DEFAULT_VOCABULARY
    return Encoder(
        vlm=StubVLMClient(),
        vocabulary=DEFAULT_VOCABULARY,
        cache_root=tmp_path,
        max_workers=max_workers,
    )


def _make_window(segment_id: str = "seg_001", window_idx: int = 0):
    from pipeline.modules.encoder.schema import WindowInput
    return WindowInput(segment_id=segment_id, window_idx=window_idx, storage=_make_storage())


# ---------------------------------------------------------------------------
# process_batch concurrent == sequential results
# ---------------------------------------------------------------------------

def test_process_batch_parallel_gives_same_results_as_sequential(tmp_path: Path) -> None:
    """Concurrent process_batch must produce identical records to sequential."""
    from pipeline.modules.encoder.encoder import Encoder
    from pipeline.modules.encoder.reasoning_arm import StubVLMClient
    from pipeline.modules.encoder.vocabulary import DEFAULT_VOCABULARY

    windows = [_make_window(segment_id="seg_001", window_idx=i) for i in range(10)]

    enc_seq = Encoder(
        vlm=StubVLMClient(),
        vocabulary=DEFAULT_VOCABULARY,
        cache_root=tmp_path / "seq",
        max_workers=1,
    )
    enc_par = Encoder(
        vlm=StubVLMClient(),
        vocabulary=DEFAULT_VOCABULARY,
        cache_root=tmp_path / "par",
        max_workers=8,
    )

    seq_records = enc_seq.process_batch(windows)
    par_records = enc_par.process_batch(windows)

    assert len(seq_records) == len(par_records) == 10
    for i, (s, p) in enumerate(zip(seq_records, par_records)):
        assert s.window_id == p.window_id, f"Window {i}: window_id mismatch"
        assert s.fields == p.fields, f"Window {i}: fields mismatch"
        assert s.failure_mode == p.failure_mode, f"Window {i}: failure_mode mismatch"


# ---------------------------------------------------------------------------
# Partial failure does not truncate batch
# ---------------------------------------------------------------------------

def test_process_batch_partial_failure_does_not_drop_results(tmp_path: Path) -> None:
    """If one window fails, the other 9 records must still be returned."""
    from pipeline.modules.encoder.encoder import Encoder
    from pipeline.modules.encoder.reasoning_arm import StubVLMClient, VLMUnavailableError
    from pipeline.modules.encoder.vocabulary import DEFAULT_VOCABULARY
    from pipeline.modules.encoder.schema import WindowInput

    call_count = [0]

    class FailOnFirstClient(StubVLMClient):
        def complete(self, video_url: str, prompt: str) -> str:
            call_count[0] += 1
            if call_count[0] == 1:
                raise VLMUnavailableError("stub/fail", "injected failure")
            return super().complete(video_url, prompt)

    enc = Encoder(
        vlm=FailOnFirstClient(),
        vocabulary=DEFAULT_VOCABULARY,
        cache_root=tmp_path,
        max_workers=4,
    )
    windows = [_make_window(window_idx=i) for i in range(10)]
    records = enc.process_batch(windows)

    assert len(records) == 10, f"Expected 10 records, got {len(records)}"
    failed = [r for r in records if not r.succeeded]
    succeeded = [r for r in records if r.succeeded]
    assert len(failed) == 1
    assert len(succeeded) == 9


# ---------------------------------------------------------------------------
# Thread-safe cache writes: 20 threads on the same window key
# ---------------------------------------------------------------------------

def test_process_thread_safety_cache_writes(tmp_path: Path) -> None:
    """20 concurrent calls on the same window must produce exactly one valid cache file."""
    from pipeline.modules.encoder.encoder import Encoder
    from pipeline.modules.encoder.reasoning_arm import StubVLMClient
    from pipeline.modules.encoder.vocabulary import DEFAULT_VOCABULARY

    enc = Encoder(
        vlm=StubVLMClient(),
        vocabulary=DEFAULT_VOCABULARY,
        cache_root=tmp_path,
    )
    window = _make_window(segment_id="race_seg", window_idx=0)

    errors: list[Exception] = []

    def _call():
        try:
            enc.process(window)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=_call) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Exceptions during concurrent writes: {errors}"

    cache_dir = tmp_path / "encoder" / "reasoning"
    json_files = list(cache_dir.glob("*.json"))
    tmp_files = list(cache_dir.glob("*.json.tmp"))

    assert len(json_files) == 1, f"Expected 1 cache file, found {len(json_files)}"
    assert len(tmp_files) == 0, f"Found leftover .json.tmp files: {tmp_files}"

    import json as _json
    data = _json.loads(json_files[0].read_text())
    assert "window_id" in data
    assert "fields" in data
