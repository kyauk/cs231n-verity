"""Module 7: Dev Dashboard — configuration.

All tunables live here. Override via environment variables; nothing else
in this module needs to change when deploying to a different environment.

Hard gate: the server REFUSES to start unless VERITY_DEV_MODE=1. This module
is a private operator surface, not a customer-facing endpoint.
"""

from __future__ import annotations

import os
from pathlib import Path

_HERE = Path(__file__).parent

# Port the FastAPI server binds to.
DEV_DASHBOARD_PORT: int = int(os.environ.get("DEV_DASHBOARD_PORT", "8002"))

# Directory where round manifests + ratings live.
# Structure: {DEV_DASHBOARD_ROUNDS_DIR}/{round_id}/manifest.json
#            {DEV_DASHBOARD_ROUNDS_DIR}/{round_id}/ratings/{window_key_str}.json
DEV_DASHBOARD_ROUNDS_DIR: Path = Path(
    os.environ.get("DEV_DASHBOARD_ROUNDS_DIR", str(_HERE / "rounds"))
)

# GCS bucket URI for video URL generation (same shape used by judge_ui).
DEV_DASHBOARD_BUCKET_URI: str = os.environ.get("DEV_DASHBOARD_BUCKET_URI", "")

# Optional service-account email for v4 URL signing.
DEV_DASHBOARD_SIGN_AS: str | None = (
    os.environ.get("DEV_DASHBOARD_SIGN_AS") or None
)

# Pre-signed URL TTL.
VIDEO_URL_TTL_SECONDS: int = int(os.environ.get("VIDEO_URL_TTL_SECONDS", "3600"))

# Hard gate: server refuses to start unless this is set to "1".
VERITY_DEV_MODE_ENABLED: bool = os.environ.get("VERITY_DEV_MODE", "") == "1"


class DevModeNotEnabledError(RuntimeError):
    """Raised at server startup when VERITY_DEV_MODE != "1".

    The dev dashboard is a private operator surface. Refusing to start
    without an explicit opt-in keeps it off customer-facing deploys.
    """

    def __init__(self) -> None:
        super().__init__(
            "[DevDashboard] VERITY_DEV_MODE is not set to '1'.\n"
            "  This dashboard is a private developer evaluation surface and\n"
            "  refuses to start without explicit opt-in. To enable:\n"
            "    export VERITY_DEV_MODE=1\n"
            "  Or set it in your .env file."
        )
