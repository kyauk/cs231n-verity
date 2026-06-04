"""Lightweight JSON-file-backed store for runner state.

Keeps the runner self-contained and dependency-light. State lives under
``waymo-pipeline/store/`` and survives restarts. Each accessor reads/writes a
single JSON file under a process-level lock, which is sufficient for the
single-process uvicorn worker the frontend talks to.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

STORE_ROOT = Path(__file__).resolve().parent / "store"
STORE_ROOT.mkdir(parents=True, exist_ok=True)

_LOCK = threading.Lock()


def _path(name: str) -> Path:
    """Resolve a JSON file path inside the store directory."""
    return STORE_ROOT / f"{name}.json"


def read(name: str, default: Any) -> Any:
    """Read a JSON document from the store, returning ``default`` if absent."""
    target = _path(name)
    with _LOCK:
        if not target.exists():
            return default
        try:
            return json.loads(target.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return default


def write(name: str, value: Any) -> None:
    """Atomically write a JSON document to the store."""
    target = _path(name)
    tmp = target.with_suffix(".json.tmp")
    with _LOCK:
        tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
        tmp.replace(target)


def append_list(name: str, item: Any) -> list[Any]:
    """Append an item to a JSON list document and return the updated list."""
    with _LOCK:
        target = _path(name)
        current: list[Any] = []
        if target.exists():
            try:
                current = json.loads(target.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                current = []
        if not isinstance(current, list):
            current = []
        current.append(item)
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(current, indent=2), encoding="utf-8")
        tmp.replace(target)
        return current
