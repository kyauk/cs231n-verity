"""Module 2: Encoder — locked types.

SchemaRecord is the output contract (defined in pipeline.interfaces.schema_record).
WindowInput is the input contract (encoder-internal, not a cross-module type).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pipeline.interfaces.schema_record import SchemaRecord  # noqa: F401 — re-exported
from pipeline.interfaces.window import WindowKey  # noqa: F401 — used by WindowInput


CURRENT_SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Input contract (encoder-internal — not shared across modules)
# ---------------------------------------------------------------------------

@dataclass
class WindowInput:
    """Everything the Encoder needs to annotate one window.

    The Encoder fetches video and pose on demand via `storage`.
    It does not receive raw bytes directly.
    """
    segment_id: str
    window_idx: int
    storage: Any                         # WindowStorage — Any to avoid circular import
    schema_version: str = CURRENT_SCHEMA_VERSION
    prompt_template_id: str = "v1_describe"

    @property
    def window_key(self) -> WindowKey:
        return WindowKey(segment_id=self.segment_id, window_idx=self.window_idx)

    @property
    def window_id_str(self) -> str:
        return str(self.window_key)


# ---------------------------------------------------------------------------
# Failure mode constants
# ---------------------------------------------------------------------------

FAILURE_INVALID_JSON = "invalid_json"
FAILURE_VOCABULARY_VIOLATION = "vocabulary_violation"
FAILURE_VLM_UNAVAILABLE = "vlm_unavailable"
FAILURE_STORAGE_ERROR = "storage_error"
FAILURE_UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Schema field shape (v1.0) — used for null-filling on failure
# ---------------------------------------------------------------------------

NULL_FIELDS_V1: dict[str, Any] = {
    "agents": None,
    "environment": {
        "weather": None,
        "time_of_day": None,
        "lighting_condition": None,
    },
    "road": {
        "geometry": None,
        "lane_count": None,
    },
    "traffic_control": None,
    "ego_task": None,
    "conditions": None,
}
