"""Module 5: Judge UI — configuration.

All tunables live here. Override via environment variables; nothing else
in this module needs to change when deploying to a different environment.
"""

from __future__ import annotations

import os
from pathlib import Path

_HERE = Path(__file__).parent

# Port the FastAPI server binds to.
# Separate from the legacy waymo runner on 8000.
JUDGE_PORT: int = int(os.environ.get("JUDGE_PORT", "8001"))

# Path to the JSON file containing list[ScoredProposal] from Module 4.
# Set JUDGE_PROPOSALS_PATH to point at a fixture file during tests.
JUDGE_PROPOSALS_PATH: Path = Path(
    os.environ.get("JUDGE_PROPOSALS_PATH", str(_HERE / "proposals.json"))
)

# Directory where Rating files are persisted.
# Structure: {JUDGE_RATINGS_DIR}/{rater_id}/{proposal_id}.json
JUDGE_RATINGS_DIR: Path = Path(
    os.environ.get("JUDGE_RATINGS_DIR", str(_HERE / "ratings"))
)

# Pre-signed URL TTL for GCS video blobs (seconds).
VIDEO_URL_TTL_SECONDS: int = int(os.environ.get("VIDEO_URL_TTL_SECONDS", "3600"))

# GCS bucket URI (gs://bucket/prefix) — passed to WindowStorage.
JUDGE_BUCKET_URI: str = os.environ.get("JUDGE_BUCKET_URI", "")

# Optional service-account email for local URL signing.
JUDGE_SIGN_AS: str | None = os.environ.get("JUDGE_SIGN_AS") or None
