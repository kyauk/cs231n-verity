"""Module 7: Dev Dashboard — FastAPI server.

Endpoints
---------
Discrimination test:
  POST /dev/rounds                                   create a round from uploaded
                                                     scored + records JSON
  GET  /dev/rounds                                   list rounds (newest first)
  GET  /dev/rounds/{round_id}                        round status + progress
  GET  /dev/rounds/{round_id}/next                   next blinded window for rater
  GET  /dev/rounds/{round_id}/video-url?segment_id=...&window_idx=...
                                                     pre-signed GCS URL
  POST /dev/rounds/{round_id}/ratings                persist Rating (source-pool
                                                     label is server-set)
  GET  /dev/rounds/{round_id}/export                 ratings + revealed source
                                                     labels for offline analysis

VLM accuracy:
  GET  /dev/accuracy/template                        copy-paste gold-set template
  POST /dev/accuracy/diff                            upload gold + records,
                                                     return diff JSON

Run with:
    VERITY_DEV_MODE=1 \\
    DEV_DASHBOARD_BUCKET_URI=gs://my-bucket/verity \\
    uvicorn pipeline.modules.dev_dashboard.server:app --port 8002 --reload

The server refuses to start unless VERITY_DEV_MODE=1 (see config.py).

Documented accepted risks (from hygiene protocol Step 4):
----------------------------------------------------------
1. CONCURRENT SAME-WINDOW RATINGS — Two simultaneous POSTs to
   /dev/rounds/{id}/ratings for the same (round_id, window) resolve via
   atomic rename. The loser's submission is silently dropped, no log
   entry. Mirrors judge_ui's pattern. Acceptable for single-rater dev use;
   if you ever wire this up for multi-rater scenarios, add an event log
   on top of the atomic-write filesystem.

2. NETWORK EXPOSURE — VERITY_DEV_MODE=1 is the only access gate. There is
   no authentication. **Bind to 127.0.0.1 only** (uvicorn's default).
   Running `uvicorn ... --host 0.0.0.0` would expose the dashboard to
   anyone on your network. If you ever need shared access, add a token
   gate first (see README "Option 1 — shared password" in the conversation
   that designed this module).
"""

from __future__ import annotations

import datetime
import hashlib
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from pipeline.interfaces.dev_round import DevRoundManifest
from pipeline.interfaces.proposal import ScoredProposal
from pipeline.interfaces.rating import Rating
from pipeline.interfaces.schema_record import SchemaRecord
from pipeline.interfaces.window import WindowKey
from pipeline.modules.dev_dashboard import config
from pipeline.modules.dev_dashboard.accuracy import (
    AccuracyDiffError,
    compute_diff,
    gold_template,
)
from pipeline.modules.dev_dashboard.sampling import (
    SamplingError,
    sample_three_pools,
)


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    if not config.VERITY_DEV_MODE_ENABLED:
        raise config.DevModeNotEnabledError()
    config.DEV_DASHBOARD_ROUNDS_DIR.mkdir(parents=True, exist_ok=True)
    _init_storage()
    yield


app = FastAPI(title="Verity Dev Dashboard", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Lazy WindowStorage instance — only built if a bucket URI is configured.
_storage: Any = None


def _init_storage() -> None:
    global _storage
    if not config.DEV_DASHBOARD_BUCKET_URI:
        print(
            "[DevDashboard] DEV_DASHBOARD_BUCKET_URI not set — "
            "/dev/rounds/{id}/video-url will return 503."
        )
        return
    try:
        from pipeline.modules.storage import WindowStorage  # noqa: PLC0415
        _storage = WindowStorage(
            bucket_uri=config.DEV_DASHBOARD_BUCKET_URI,
            sign_as=config.DEV_DASHBOARD_SIGN_AS,
        )
        print(
            f"[DevDashboard] WindowStorage initialized for "
            f"{config.DEV_DASHBOARD_BUCKET_URI}"
        )
    except Exception as exc:
        print(f"[DevDashboard] WARNING: storage init failed: {exc}")


# ---------------------------------------------------------------------------
# Filesystem helpers — round persistence
# ---------------------------------------------------------------------------

def _round_dir(round_id: str) -> Path:
    if "/" in round_id or "\\" in round_id or ".." in round_id:
        # Defensive: round_id appears in filesystem paths; refuse traversal.
        raise HTTPException(status_code=400, detail="invalid round_id")
    return config.DEV_DASHBOARD_ROUNDS_DIR / round_id


def _manifest_path(round_id: str) -> Path:
    return _round_dir(round_id) / "manifest.json"


def _ratings_dir(round_id: str) -> Path:
    return _round_dir(round_id) / "ratings"


def _rating_path(round_id: str, window_key: WindowKey) -> Path:
    safe = f"{window_key.segment_id}__{window_key.window_idx:04d}"
    return _ratings_dir(round_id) / f"{safe}.json"


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _load_manifest(round_id: str) -> DevRoundManifest:
    path = _manifest_path(round_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"round {round_id} not found")
    return DevRoundManifest.from_json(
        json.loads(path.read_text(encoding="utf-8"))
    )


def _load_ratings(round_id: str) -> list[Rating]:
    rdir = _ratings_dir(round_id)
    if not rdir.exists():
        return []
    return [
        Rating.from_json(json.loads(p.read_text(encoding="utf-8")))
        for p in sorted(rdir.glob("*.json"))
    ]


def _window_to_source_pool(
    manifest: DevRoundManifest, window: WindowKey,
) -> str | None:
    """Look up which source pool a window came from. None = not in this round."""
    for arm, windows in manifest.pools.items():
        if window in windows:
            return arm
    return None


def _round_is_complete(manifest: DevRoundManifest, ratings: list[Rating]) -> bool:
    """A round is complete when every unique window in shuffled_order has
    at least one rating. Pools may overlap (a window can be in Verity AND
    Random); ratings are deduplicated by window key on disk, so the right
    completion criterion is set-coverage of unique windows, not a count
    comparison against shuffled_order's length."""
    rated_keys = {r.proposal_id for r in ratings}
    return all(
        _window_key_str(w) in rated_keys for w in manifest.shuffled_order
    )


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------

class CreateRoundRequest(BaseModel):
    dataset_label: str
    pool_size: int = 30
    seed: int = 0
    top_k_rare_atoms: int = 5
    scored: list[dict]            # ScoredProposal-shaped dicts
    schema_records: list[dict]    # SchemaRecord-shaped dicts


class CreateRoundResponse(BaseModel):
    round_id: str
    total_windows: int            # length of shuffled_order
    naive_rare_atoms: list[str]


class WindowKeyDTO(BaseModel):
    segment_id: str
    window_idx: int


class RoundStatus(BaseModel):
    round_id: str
    created_at: str
    dataset_label: str
    pool_size: int
    total_windows: int
    rated_count: int
    complete: bool


class NextWindowResponse(BaseModel):
    """Blinded next-window payload for the rater. Source pool is NOT here."""
    complete: bool
    window: WindowKeyDTO | None
    progress_idx: int             # 1-based position in shuffled_order
    total_windows: int


class VideoUrlResponse(BaseModel):
    url: str
    generated_at: str


class SubmitRatingRequest(BaseModel):
    rater_id: str
    window: WindowKeyDTO
    safety_relevance: int         # 1-5
    perceived_rarity: int         # 1-5
    free_text_note: str | None = None

    @field_validator("safety_relevance", "perceived_rarity")
    @classmethod
    def _range(cls, v: int) -> int:
        if not 1 <= v <= 5:
            raise ValueError("score must be between 1 and 5")
        return v


class ExportRow(BaseModel):
    """One row of the discrimination-test export. Source pool revealed."""
    rater_id: str
    window: WindowKeyDTO
    source_pool: str              # "verity" | "random" | "naive_rare"
    safety_relevance: int
    perceived_rarity: int
    timestamp: str
    free_text_note: str | None


class ExportResponse(BaseModel):
    round_id: str
    dataset_label: str
    pool_size: int
    seed: int
    naive_rare_atoms: list[str]
    complete: bool
    ratings: list[ExportRow]


class RoundListEntry(BaseModel):
    round_id: str
    created_at: str
    dataset_label: str


class DiffRequest(BaseModel):
    gold: dict
    schema_records: list[dict]


# ---------------------------------------------------------------------------
# Endpoint: create round
# ---------------------------------------------------------------------------

@app.post("/dev/rounds", response_model=CreateRoundResponse)
def create_round(req: CreateRoundRequest) -> CreateRoundResponse:
    # Parse incoming JSON into interface types
    try:
        scored = [ScoredProposal.from_json(d) for d in req.scored]
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"malformed scored payload: {exc}"
        ) from exc
    try:
        records = [SchemaRecord.from_json(d) for d in req.schema_records]
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"malformed schema_records payload: {exc}"
        ) from exc

    try:
        sample = sample_three_pools(
            scored=scored, schema_records=records,
            pool_size=req.pool_size, seed=req.seed,
            top_k_rare_atoms=req.top_k_rare_atoms,
        )
    except SamplingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    import random as _random_mod
    # Deterministic shuffle keyed off seed so we can reproduce the
    # presentation order alongside the pools.
    all_windows = sample.verity + sample.random + sample.naive_rare
    rng = _random_mod.Random(req.seed + 2)
    shuffled = list(all_windows)
    rng.shuffle(shuffled)

    now = datetime.datetime.utcnow().isoformat() + "Z"
    # Round ID: timestamp + short hash of (dataset_label, seed) to keep
    # filesystem paths unique across re-runs at the same instant.
    h = hashlib.sha256(
        f"{req.dataset_label}|{req.seed}|{now}".encode()
    ).hexdigest()[:8]
    safe_ts = now.replace(":", "-").replace(".", "-")
    round_id = f"round_{safe_ts}_{h}"

    manifest = DevRoundManifest(
        round_id=round_id,
        created_at=now,
        dataset_label=req.dataset_label,
        pool_size=req.pool_size,
        seed=req.seed,
        pools={
            "verity": sample.verity,
            "random": sample.random,
            "naive_rare": sample.naive_rare,
        },
        shuffled_order=shuffled,
        naive_rare_atoms=sample.naive_rare_atoms,
    )
    _atomic_write_json(_manifest_path(round_id), manifest.to_json())

    return CreateRoundResponse(
        round_id=round_id,
        total_windows=len(shuffled),
        naive_rare_atoms=sample.naive_rare_atoms,
    )


# ---------------------------------------------------------------------------
# Endpoint: list rounds
# ---------------------------------------------------------------------------

@app.get("/dev/rounds", response_model=list[RoundListEntry])
def list_rounds() -> list[RoundListEntry]:
    base = config.DEV_DASHBOARD_ROUNDS_DIR
    if not base.exists():
        return []
    out: list[RoundListEntry] = []
    for d in base.iterdir():
        if not d.is_dir():
            continue
        try:
            manifest = _load_manifest(d.name)
        except HTTPException:
            continue  # skip missing manifests silently
        out.append(RoundListEntry(
            round_id=manifest.round_id,
            created_at=manifest.created_at,
            dataset_label=manifest.dataset_label,
        ))
    # Newest first
    out.sort(key=lambda r: r.created_at, reverse=True)
    return out


# ---------------------------------------------------------------------------
# Endpoint: round status
# ---------------------------------------------------------------------------

@app.get("/dev/rounds/{round_id}", response_model=RoundStatus)
def get_round_status(round_id: str) -> RoundStatus:
    manifest = _load_manifest(round_id)
    rated = _load_ratings(round_id)
    return RoundStatus(
        round_id=manifest.round_id,
        created_at=manifest.created_at,
        dataset_label=manifest.dataset_label,
        pool_size=manifest.pool_size,
        total_windows=len(manifest.shuffled_order),
        rated_count=len(rated),
        complete=_round_is_complete(manifest, rated),
    )


# ---------------------------------------------------------------------------
# Endpoint: next blinded window
# ---------------------------------------------------------------------------

@app.get("/dev/rounds/{round_id}/next", response_model=NextWindowResponse)
def get_next_window(round_id: str) -> NextWindowResponse:
    manifest = _load_manifest(round_id)
    rated_keys = {r.proposal_id for r in _load_ratings(round_id)}
    # Use proposal_id as the rating key (mapped from window_key str form)
    total = len(manifest.shuffled_order)
    for idx, win in enumerate(manifest.shuffled_order, start=1):
        if _window_key_str(win) not in rated_keys:
            return NextWindowResponse(
                complete=False,
                window=WindowKeyDTO(
                    segment_id=win.segment_id, window_idx=win.window_idx,
                ),
                progress_idx=idx,
                total_windows=total,
            )
    return NextWindowResponse(
        complete=True, window=None, progress_idx=total, total_windows=total,
    )


def _window_key_str(win: WindowKey) -> str:
    return f"{win.segment_id}/{win.window_idx:04d}"


# ---------------------------------------------------------------------------
# Endpoint: pre-signed video URL
# ---------------------------------------------------------------------------

@app.get("/dev/rounds/{round_id}/video-url", response_model=VideoUrlResponse)
def get_video_url(
    round_id: str,
    segment_id: str = Query(...),
    window_idx: int = Query(...),
    camera: str = Query("FRONT"),
) -> VideoUrlResponse:
    if _storage is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "video URLs disabled: DEV_DASHBOARD_BUCKET_URI not set. "
                "Set the env var and restart the server."
            ),
        )
    manifest = _load_manifest(round_id)
    target = WindowKey(segment_id=segment_id, window_idx=window_idx)
    if _window_to_source_pool(manifest, target) is None:
        raise HTTPException(
            status_code=400,
            detail=f"{_window_key_str(target)} is not in round {round_id}",
        )
    try:
        url = _storage.get_window_video_url(
            segment_id=segment_id, window_idx=window_idx,
            camera=camera, ttl_seconds=config.VIDEO_URL_TTL_SECONDS,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"signed-URL generation failed: {type(exc).__name__}: {exc}",
        ) from exc
    return VideoUrlResponse(
        url=url,
        generated_at=datetime.datetime.utcnow().isoformat() + "Z",
    )


# ---------------------------------------------------------------------------
# Endpoint: submit rating
# ---------------------------------------------------------------------------

@app.post("/dev/rounds/{round_id}/ratings")
def submit_rating(round_id: str, req: SubmitRatingRequest) -> dict[str, Any]:
    manifest = _load_manifest(round_id)
    target = WindowKey(
        segment_id=req.window.segment_id, window_idx=req.window.window_idx,
    )
    source_pool = _window_to_source_pool(manifest, target)
    if source_pool is None:
        raise HTTPException(
            status_code=400,
            detail=f"{_window_key_str(target)} is not in round {round_id}",
        )

    rating = Rating(
        rater_id=req.rater_id,
        proposal_id=_window_key_str(target),    # keyed by window str
        arm=source_pool,                         # server-set; rater never knows
        coherence_score=req.safety_relevance,    # reuse field for our axis 1
        usefulness_score=req.perceived_rarity,   # reuse field for our axis 2
        timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        free_text_note=req.free_text_note,
        seen_motivating_scenes=[],
    )
    _atomic_write_json(_rating_path(round_id, target), rating.to_json())
    return {"ok": True}


# ---------------------------------------------------------------------------
# Endpoint: export
# ---------------------------------------------------------------------------

@app.get("/dev/rounds/{round_id}/export", response_model=ExportResponse)
def export_round(round_id: str) -> ExportResponse:
    manifest = _load_manifest(round_id)
    ratings = _load_ratings(round_id)
    rows: list[ExportRow] = []
    for r in ratings:
        win = WindowKey.from_str(r.proposal_id)
        rows.append(ExportRow(
            rater_id=r.rater_id,
            window=WindowKeyDTO(
                segment_id=win.segment_id, window_idx=win.window_idx,
            ),
            source_pool=r.arm,
            safety_relevance=r.coherence_score,
            perceived_rarity=r.usefulness_score,
            timestamp=r.timestamp,
            free_text_note=r.free_text_note,
        ))
    return ExportResponse(
        round_id=manifest.round_id,
        dataset_label=manifest.dataset_label,
        pool_size=manifest.pool_size,
        seed=manifest.seed,
        naive_rare_atoms=manifest.naive_rare_atoms,
        complete=_round_is_complete(manifest, ratings),
        ratings=rows,
    )


# ---------------------------------------------------------------------------
# Endpoint: accuracy template
# ---------------------------------------------------------------------------

@app.get("/dev/accuracy/template")
def accuracy_template() -> dict[str, Any]:
    return gold_template()


# ---------------------------------------------------------------------------
# Endpoint: accuracy diff
# ---------------------------------------------------------------------------

@app.post("/dev/accuracy/diff")
def accuracy_diff(req: DiffRequest) -> dict[str, Any]:
    try:
        records = [SchemaRecord.from_json(d) for d in req.schema_records]
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"malformed schema_records: {exc}"
        ) from exc
    try:
        report = compute_diff(req.gold, records)
    except AccuracyDiffError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return report.to_json()
