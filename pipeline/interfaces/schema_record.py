"""Shared SchemaRecord type — produced by Module 2: Encoder, consumed by Modules 3–6."""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any

from pipeline.interfaces.window import WindowKey


# ---------------------------------------------------------------------------
# SchemaRecord — the annotated window contract
# ---------------------------------------------------------------------------

@dataclass
class SchemaRecord:
    """One annotated window. failure_mode=None means the record is usable.

    Downstream modules (Hypothesizer, Scorer, Evaluation) must filter on
    ``failure_mode is None`` before consuming ``fields``.
    """
    window_id: WindowKey
    arm: str                            # "reasoning" (visual arm Phase 2)
    schema_version: str
    prompt_template_id: str | None
    fields: dict[str, Any]             # conforms to locked schema vocabulary
    failure_mode: str | None           # None = success
    cached: bool = False
    created_at: str = field(
        default_factory=lambda: datetime.datetime.utcnow().isoformat() + "Z"
    )
    raw_vlm_response: str | None = field(default=None, repr=False)

    @property
    def succeeded(self) -> bool:
        return self.failure_mode is None

    def to_json(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id.to_json(),
            "arm": self.arm,
            "schema_version": self.schema_version,
            "prompt_template_id": self.prompt_template_id,
            "fields": self.fields,
            "failure_mode": self.failure_mode,
            "cached": self.cached,
            "created_at": self.created_at,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "SchemaRecord":
        raw_wid = d["window_id"]
        if isinstance(raw_wid, dict):
            wid = WindowKey.from_json(raw_wid)
        else:
            wid = WindowKey.from_str(str(raw_wid))
        return cls(
            window_id=wid,
            arm=str(d.get("arm", "reasoning")),
            schema_version=str(d.get("schema_version", "")),
            prompt_template_id=d.get("prompt_template_id"),
            fields=dict(d.get("fields", {})),
            failure_mode=d.get("failure_mode"),
            cached=bool(d.get("cached", False)),
            created_at=str(d.get("created_at", "")),
        )
