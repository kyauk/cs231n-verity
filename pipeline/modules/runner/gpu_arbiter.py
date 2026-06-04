"""Single-GPU arbiter for the L40S that hosts the Cosmos NIMs.

The host has ONE L40S. ``cosmos-reason2`` (reason1, :8081) and ``cosmos-embed1``
(embed1, :8080) each reserve the whole GPU, so they cannot run concurrently
without OOM. reason1 is the resting tenant: it stays up and serves agentic
analysis. embed1 is borrowed only for the embed+cluster window of an ingest
batch, then reason1 is restored.

This module is the single owner of that mutex. It lives in the waymo_runner
process — the one place that both launches ingest batches and serves the
reason1-consuming ``/analysis/run-stream`` endpoint — so no cross-process
coordination is needed.

State machine (a stopped container releases its GPU reservation, so swapping is
just compose stop/up of the two services)::

    REASON_READY ──embed_window()──▶ DRAINING ──▶ SWAPPING_TO_EMBED ──▶ EMBED_READY
        ▲                                                                   │
        └──────────── SWAPPING_TO_REASON ◀──── last embed_window() exits ◀──┘

Guarantees:
  * reason1 callers are rejected (busy) the instant a swap is requested.
  * an in-flight analysis run is never killed mid-flight; the swap waits for it
    to drain (bounded by ``GPU_DRAIN_TIMEOUT_S``).
  * the resting state (reason1 up) is restored in a ``finally``, so any failure
    inside the embed window still returns the GPU to reason1.
  * overlapping ingest batches share one embed window via a refcount; the swap
    back happens only when the last one exits.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # runner -> modules -> pipeline -> repo root

# ---- config (override via env) --------------------------------------------
# compose service names (docker-compose.yml) and their host ports.
REASON_SERVICE = os.environ.get("GPU_REASON_SERVICE", "cosmos-reason2")
REASON_PORT = int(os.environ.get("GPU_REASON_PORT", "8081"))
EMBED_SERVICE = os.environ.get("GPU_EMBED_SERVICE", "cosmos-embed1")
EMBED_PORT = int(os.environ.get("GPU_EMBED_PORT", "8080"))
COMPOSE_PROFILE = os.environ.get("GPU_COMPOSE_PROFILE", "gpu")

# NIM cold-load can take minutes; readiness poll budgets.
READY_TIMEOUT_S = int(os.environ.get("GPU_READY_TIMEOUT_S", "420"))   # 7 min
READY_POLL_S = float(os.environ.get("GPU_READY_POLL_S", "3"))
# how long to wait for an in-flight analysis run to finish before swapping.
DRAIN_TIMEOUT_S = int(os.environ.get("GPU_DRAIN_TIMEOUT_S", "1200"))  # 20 min
DRAIN_POLL_S = float(os.environ.get("GPU_DRAIN_POLL_S", "1"))


class GpuState(str, Enum):
    REASON_READY = "reason_ready"        # resting: reason1 up, serving analysis
    DRAINING = "draining"                # swap requested; waiting for reason calls to finish
    SWAPPING_TO_EMBED = "swapping_to_embed"
    EMBED_READY = "embed_ready"          # embed1 up, serving clustering
    SWAPPING_TO_REASON = "swapping_to_reason"
    DEGRADED = "degraded"                # a swap failed; neither model is trustworthy


class GpuBusyError(RuntimeError):
    """Raised when a reason1 call is attempted while the GPU is not REASON_READY."""


def _compose(*args: str) -> subprocess.CompletedProcess[str]:
    """Run a ``docker compose`` command from the project root."""
    cmd = ["docker", "compose", "--profile", COMPOSE_PROFILE, *args]
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True,
                          text=True, check=False)


def _is_ready(port: int) -> bool:
    """True if the NIM on ``port`` reports ready (matches waymo_embed_scenes)."""
    try:
        r = requests.get(f"http://localhost:{port}/v1/health/ready", timeout=5)
        return r.status_code == 200
    except requests.RequestException:
        return False


class _GpuArbiter:
    """Threadsafe single-GPU mutex. One instance per process (see ``gpu`` below)."""

    def __init__(self) -> None:
        self._cv = threading.Condition(threading.RLock())
        # Initialize from reality: whichever NIM is already up is the live state.
        if _is_ready(REASON_PORT):
            self._state = GpuState.REASON_READY
        elif _is_ready(EMBED_PORT):
            self._state = GpuState.EMBED_READY
        else:
            self._state = GpuState.SWAPPING_TO_REASON  # unknown; reconciled on first use
        self._reason_inflight = 0   # active analysis runs holding the reason lease
        self._embed_refs = 0        # open embed windows (coalesced batches)
        self._last_error: str | None = None

    # -- introspection ------------------------------------------------------
    def status(self) -> dict[str, Any]:
        with self._cv:
            return {
                "state": self._state.value,
                "reasonInflight": self._reason_inflight,
                "embedWindows": self._embed_refs,
                "reasonReady": self._state is GpuState.REASON_READY,
                "embedReady": self._state is GpuState.EMBED_READY,
                "lastError": self._last_error,
            }

    def _set(self, state: GpuState, on_change: Callable[[str], None] | None = None) -> None:
        self._state = state
        if on_change is not None:
            on_change(state.value)
        self._cv.notify_all()

    # -- low-level swaps ----------------------------------------------------
    def _wait_ready(self, port: int) -> bool:
        deadline = time.monotonic() + READY_TIMEOUT_S
        while time.monotonic() < deadline:
            if _is_ready(port):
                return True
            time.sleep(READY_POLL_S)
        return False

    def _swap_to(self, up_service: str, up_port: int, down_service: str) -> None:
        """Stop ``down_service``, start ``up_service``, block until it is ready.

        Caller must hold the condition lock. Raises on failure (caller maps to
        DEGRADED / restores reason as appropriate).
        """
        # Stop the other tenant first so the GPU is free before the new one boots.
        _compose("stop", down_service)
        up = _compose("up", "-d", up_service)
        if up.returncode != 0:
            raise RuntimeError(
                f"`compose up {up_service}` failed: {(up.stderr or up.stdout)[-500:]}"
            )
        if not self._wait_ready(up_port):
            raise RuntimeError(
                f"{up_service} did not become ready on :{up_port} within {READY_TIMEOUT_S}s"
            )

    # -- reason lease (analysis endpoint) -----------------------------------
    @contextlib.contextmanager
    def reason_lease(self) -> Iterator[None]:
        """Hold the GPU for a reason1 call. Raises GpuBusyError if not resting.

        Acquiring increments the in-flight count so a concurrent embed_window()
        waits for this call to finish before swapping (no mid-flight kill).
        """
        with self._cv:
            if self._state is not GpuState.REASON_READY:
                raise GpuBusyError(
                    "GPU busy — embedding an ingest batch. Try again when it finishes."
                )
            self._reason_inflight += 1
        try:
            yield
        finally:
            with self._cv:
                self._reason_inflight -= 1
                self._cv.notify_all()

    # -- embed window (ingest cluster stage) --------------------------------
    @contextlib.contextmanager
    def embed_window(self, on_stage: Callable[[str], None] | None = None) -> Iterator[None]:
        """Borrow the GPU for embed1 for the duration of the block.

        Swaps reason1 -> embed1 on enter (draining in-flight analysis first) and
        embed1 -> reason1 on exit. Overlapping windows are coalesced via refcount;
        the swap-back runs only when the last window exits, and always runs (even
        on error) so reason1 is the guaranteed resting state.
        """
        self._acquire_embed(on_stage)
        try:
            yield
        finally:
            self._release_embed(on_stage)

    def _acquire_embed(self, on_stage: Callable[[str], None] | None) -> None:
        with self._cv:
            # Coalesce: a window is already open -> reuse it.
            if self._embed_refs > 0 and self._state is GpuState.EMBED_READY:
                self._embed_refs += 1
                return
            # If another thread is mid-swap, wait for it to settle, then re-decide.
            while self._state in (GpuState.DRAINING, GpuState.SWAPPING_TO_EMBED,
                                  GpuState.SWAPPING_TO_REASON):
                self._cv.wait()
            if self._embed_refs > 0 and self._state is GpuState.EMBED_READY:
                self._embed_refs += 1
                return

            # 1) Drain: reject new reason calls, wait for in-flight ones to finish.
            self._set(GpuState.DRAINING, on_stage)
            deadline = time.monotonic() + DRAIN_TIMEOUT_S
            while self._reason_inflight > 0:
                if time.monotonic() >= deadline:
                    break  # bounded: proceed rather than hang the batch forever
                self._cv.wait(timeout=DRAIN_POLL_S)

            # 2) Swap reason1 -> embed1.
            self._set(GpuState.SWAPPING_TO_EMBED, on_stage)
            try:
                self._swap_to(EMBED_SERVICE, EMBED_PORT, REASON_SERVICE)
            except Exception as exc:  # noqa: BLE001 — restore resting state on failure
                self._last_error = str(exc)
                self._restore_reason_locked(on_stage)
                raise
            self._embed_refs = 1
            self._set(GpuState.EMBED_READY, on_stage)

    def _release_embed(self, on_stage: Callable[[str], None] | None) -> None:
        with self._cv:
            self._embed_refs -= 1
            if self._embed_refs > 0:
                return  # another batch still using embed1; keep it up
            # Last window closed -> swap back to the resting tenant.
            self._set(GpuState.SWAPPING_TO_REASON, on_stage)
            try:
                self._swap_to(REASON_SERVICE, REASON_PORT, EMBED_SERVICE)
            except Exception as exc:  # noqa: BLE001
                self._last_error = str(exc)
                self._set(GpuState.DEGRADED, on_stage)
                raise
            self._set(GpuState.REASON_READY, on_stage)

    def _restore_reason_locked(self, on_stage: Callable[[str], None] | None) -> None:
        """Best-effort return to reason1 after a failed embed swap. Holds lock."""
        self._set(GpuState.SWAPPING_TO_REASON, on_stage)
        try:
            self._swap_to(REASON_SERVICE, REASON_PORT, EMBED_SERVICE)
            self._set(GpuState.REASON_READY, on_stage)
        except Exception as exc:  # noqa: BLE001 — give up loudly; operator must intervene
            self._last_error = f"{self._last_error}; restore failed: {exc}"
            self._set(GpuState.DEGRADED, on_stage)


# Process-wide singleton. The runner is a single uvicorn worker, so one arbiter
# instance owns the GPU for the whole app.
gpu = _GpuArbiter()
