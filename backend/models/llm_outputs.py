"""
Pydantic output models for LLM service boundaries.
"""

from typing import Literal, Optional

from pydantic import BaseModel


class TriageSummaryResult(BaseModel):
    """
    Structured triage summary returned by the LLM client.
    """

    summary: str
    scenario_type: Optional[str] = None
    failure_mode_hints: list[str]
    likely_subsystem: Optional[str] = None
    severity_cue: Literal["critical", "high", "medium", "low", "unknown"]
    tags: list[str]
