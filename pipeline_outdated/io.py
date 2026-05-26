"""Shared I/O primitives for the Waymo pipeline.

Every pipeline stage imports from here instead of re-implementing its own
JSONL reading/writing and progress-event emission. This keeps the wire
formats for those two concerns in one place.

Progress events
---------------
Pipeline stages emit structured progress by calling ``emit_progress``.
The runner parses lines that start with ``PIPELINE_PROGRESS:`` and
forwards them as SSE events.  The frontend routes on the ``step`` field
using ``PROGRESS_STEPS`` as the authoritative vocabulary — add new steps
here, not in the individual stage files.

JSONL helpers
-------------
``read_jsonl`` and ``write_jsonl`` are thin wrappers that encode the
error-handling policy once: skip blank lines, skip malformed rows with a
logged warning, never raise on a missing file.  Every stage that touches
JSONL calls these; none re-implement them.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Typed progress-event vocabulary
# ---------------------------------------------------------------------------

# Every step name the pipeline emits and the frontend expects lives here.
# Changing a name here is the one edit needed to rename a step everywhere.
ProgressStep = Literal[
    "start",
    "describe",
    "describe_done",
    "debate_round_proponent",
    "debate_round_critic",
    "debate_judge",
    "save",
]

PIPELINE_PROGRESS_PREFIX = "PIPELINE_PROGRESS:"


def emit_progress(step: ProgressStep, title: str, detail: str = "") -> None:
    """Print a structured progress line consumed by the SSE-streaming runner.

    The runner reads stdout from the subprocess and forwards any line that
    starts with ``PIPELINE_PROGRESS:`` as an SSE ``kind=progress`` event.
    """
    payload = json.dumps(
        {"step": step, "title": title, "detail": detail}, ensure_ascii=False
    )
    print(f"{PIPELINE_PROGRESS_PREFIX}{payload}", flush=True)


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts.

    Returns an empty list if the file does not exist.  Skips blank lines
    and logs a warning for any line that fails JSON parsing — never raises.
    """
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            log.warning("%s line %d: JSON parse error: %s", p.name, i, exc)
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
        else:
            log.warning("%s line %d: skipping non-object row", p.name, i)
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to a JSONL file, creating parent directories as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    """Append one row to a JSONL file (creates the file if absent)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
