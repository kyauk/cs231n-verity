"""Thread-safety tests for the lazy GCS-client initialization in
WindowStorage and FlatMP4Storage.

Before the double-checked locking fix, concurrent first calls from N threads
could construct the GCS client N times — functionally equivalent output, but
N× more API setup work and N× the network handshakes.

The fix: a per-instance threading.Lock guarded by a double-checked pattern.
The hot path (after init) is lock-free; only the first init contends.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

from pipeline.modules.storage import FlatMP4Storage, WindowStorage


def _hammer_get_bucket(storage_obj, n_threads: int) -> int:
    """Spawn n_threads that all call _get_bucket simultaneously.

    Returns the number of GCS storage.Client(...) calls that actually fired.
    Uses a Barrier so threads release as close to simultaneously as possible,
    maximizing the race window. Patches google.cloud.storage.Client so no
    network/auth happens.
    """
    barrier = threading.Barrier(n_threads)
    fake_client = MagicMock()
    fake_client.bucket.return_value = MagicMock()

    with patch("google.cloud.storage.Client", return_value=fake_client) as ctor:
        def worker() -> None:
            barrier.wait()  # release all threads at once
            storage_obj._get_bucket()

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return ctor.call_count


# ---------------------------------------------------------------------------
# WindowStorage
# ---------------------------------------------------------------------------

def test_window_storage_get_bucket_constructs_client_once_under_concurrency() -> None:
    storage = WindowStorage(bucket_uri="gs://test/prefix")
    n_constructions = _hammer_get_bucket(storage, n_threads=16)
    assert n_constructions == 1, (
        f"WindowStorage constructed the GCS client {n_constructions} times "
        f"with 16 concurrent first-callers — expected exactly 1 (lock held)."
    )


def test_window_storage_subsequent_calls_skip_the_lock() -> None:
    """After init, _get_bucket must take the fast path (no construction).

    Pre-warms the cache, then concurrent calls — none should construct.
    """
    storage = WindowStorage(bucket_uri="gs://test/prefix")
    storage._bucket_obj = MagicMock()  # pretend already initialized
    n = _hammer_get_bucket(storage, n_threads=8)
    assert n == 0


# ---------------------------------------------------------------------------
# FlatMP4Storage
# ---------------------------------------------------------------------------

def test_flat_mp4_storage_get_bucket_constructs_client_once_under_concurrency() -> None:
    storage = FlatMP4Storage(bucket_uri="gs://test/prefix", cameras=["FRONT"])
    n_constructions = _hammer_get_bucket(storage, n_threads=16)
    assert n_constructions == 1, (
        f"FlatMP4Storage constructed the GCS client {n_constructions} times "
        f"with 16 concurrent first-callers — expected exactly 1 (lock held)."
    )


def test_flat_mp4_storage_subsequent_calls_skip_the_lock() -> None:
    storage = FlatMP4Storage(bucket_uri="gs://test/prefix", cameras=["FRONT"])
    storage._bucket_obj = MagicMock()
    n = _hammer_get_bucket(storage, n_threads=8)
    assert n == 0
