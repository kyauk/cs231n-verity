"""Shared error types that cross module boundaries.

`WindowStorageError` is raised by Module 1's `WindowStorage` retrieval client
and caught by any module that reads windows (the Encoder, the Judge UI). Because
it travels across a module boundary, its definition lives here in the shared
interfaces layer rather than inside Module 1's internals — the same rule that
keeps `WindowKey` and friends in `pipeline.interfaces.window`.

Storage-internal errors (source-adapter failures, schema mismatches, fatal
ingestion aborts) stay inside Module 1; they never reach another module, so
they do not belong here. They subclass `StorageError` defined below.
"""

from __future__ import annotations


class StorageError(Exception):
    """Base class for all Module 1: Storage errors."""


class WindowStorageError(StorageError):
    """Raised when WindowStorage cannot fulfill a retrieval request.

    Carries the window/segment `key` that failed and a human-readable `detail`.
    Consumers catch this to record a storage failure rather than crashing the run.
    """

    def __init__(self, key: str, detail: str) -> None:
        self.key = key
        self.detail = detail
        super().__init__(
            f"[Storage] WindowStorage retrieval failed for {key!r}.\n"
            f"  Detail: {detail}"
        )
