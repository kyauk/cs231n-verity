"""Module 5: Judge UI — FastAPI server.

Endpoints
---------
GET  /judge/proposals                  list[ProposalRow]   (arm blinded)
GET  /judge/proposals/{proposal_id}    ProposalDetail       (arm blinded)
GET  /judge/video-url                  {url, generated_at}  pre-signed GCS URL
POST /judge/ratings                    {ok: true}           persist Rating
GET  /judge/session/{rater_id}         {rated_ids, summary} resumability + stats
GET  /judge/ratings/export             list[Rating]         for Module 6

Run with:
    JUDGE_BUCKET_URI=gs://my-bucket/verity \\
    uvicorn pipeline.modules.judge_ui.server:app --port 8001 --reload

Known assumptions and accepted risks (from hygiene protocol Step 4):
----------------------------------------------------------------------
1. STALE PROPOSALS — Proposals are loaded once at startup. If Module 4
   regenerates proposals.json mid-session, the server continues serving
   the old snapshot. Restart the server to pick up new proposals.
   Accepted risk for Phase 1; acceptable because rating sessions are short
   and Module 4 won't re-run during an active session in practice.

2. CONCURRENT SAME-RATER WRITES — Two simultaneous POST /judge/ratings from
   the same (rater_id, proposal_id) are last-writer-wins via atomic rename.
   No log entry is written for the race loser. Accepted risk: the UI is
   single-page and doesn't allow concurrent submission, so double-submits
   are only possible via direct API calls.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from pipeline.interfaces.proposal import ScoredProposal
from pipeline.interfaces.rating import Rating
from pipeline.interfaces.window import WindowKey
from pipeline.modules.judge_ui import config

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    _load_proposals()
    _init_storage()
    yield


app = FastAPI(title="Verity Judge UI", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Server-side state (loaded once at startup)
# ---------------------------------------------------------------------------

# proposal_id → ScoredProposal (includes arm — never sent to rater)
_proposals: dict[str, ScoredProposal] = {}

# Ordered list of accepted proposal IDs for stable list ordering
_accepted_ids: list[str] = []

# WindowStorage instance (lazy-initialized when bucket URI is set)
_storage: Any = None


def _load_proposals() -> None:
    path = config.JUDGE_PROPOSALS_PATH
    if not path.exists():
        raise RuntimeError(
            f"[JudgeUI] proposals file not found: {path}\n"
            "Generate it by running Module 4 (Scorer) or provide a fixture file.\n"
            f"Override path with JUDGE_PROPOSALS_PATH env var."
        )

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise RuntimeError(f"[JudgeUI] proposals file must contain a JSON array, got {type(raw)}")

    loaded = 0
    for item in raw:
        try:
            proposal = ScoredProposal.from_json(item)
        except Exception as exc:
            print(f"[JudgeUI] WARNING: skipping malformed proposal entry: {exc}")
            continue
        _proposals[proposal.composition_id] = proposal
        if proposal.accepted:
            _accepted_ids.append(proposal.composition_id)
        loaded += 1

    # Sort accepted proposals by final_rank_score descending (highest first)
    _accepted_ids.sort(
        key=lambda pid: _proposals[pid].final_rank_score,
        reverse=True,
    )

    accepted_count = len(_accepted_ids)
    print(
        f"[JudgeUI] Loaded {loaded} proposals ({accepted_count} accepted) from {path}"
    )
    if accepted_count == 0:
        print("[JudgeUI] WARNING: no accepted proposals — raters will see an empty queue")


def _init_storage() -> None:
    global _storage
    bucket_uri = config.JUDGE_BUCKET_URI
    if not bucket_uri:
        print(
            "[JudgeUI] JUDGE_BUCKET_URI not set — video URL endpoint will return 503. "
            "Set it to enable video playback."
        )
        return
    try:
        from pipeline.modules.storage.client import WindowStorage  # noqa: PLC0415
        _storage = WindowStorage(
            bucket_uri=bucket_uri,
            sign_as=config.JUDGE_SIGN_AS,
        )
        print(f"[JudgeUI] WindowStorage initialized for {bucket_uri}")
    except Exception as exc:
        print(f"[JudgeUI] WARNING: WindowStorage init failed: {exc}")


# ---------------------------------------------------------------------------
# Pydantic response/request models
# ---------------------------------------------------------------------------

class ScoreBadges(BaseModel):
    novelty_score: float
    plausibility_score: float
    frontier_difficulty_score: float | None
    final_rank_score: float


class MotivatingScene(BaseModel):
    segment_id: str
    window_idx: int


class ProposalRow(BaseModel):
    """Blinded proposal summary for the list view. arm is NOT present."""
    proposal_id: str
    constituents: list[str]
    scores: ScoreBadges
    motivating_scene_count: int


class ProposalDetail(BaseModel):
    """Blinded proposal detail for the review screen. arm is NOT present."""
    proposal_id: str
    constituents: list[str]
    scores: ScoreBadges
    plausibility_justification: str
    motivating_scenes: list[MotivatingScene]
    rejection_reason: str | None


class VideoUrlResponse(BaseModel):
    url: str
    generated_at: str  # ISO-8601 — used by frontend to detect staleness


class RatingSubmission(BaseModel):
    rater_id: str
    proposal_id: str
    coherence_score: int
    usefulness_score: int
    free_text_note: str | None = None
    seen_motivating_scenes: list[MotivatingScene] = []

    @field_validator("coherence_score", "usefulness_score")
    @classmethod
    def _score_range(cls, v: int) -> int:
        if not 1 <= v <= 5:
            raise ValueError("score must be between 1 and 5")
        return v


class SessionSummary(BaseModel):
    rater_id: str
    rated_proposal_ids: list[str]
    total_accepted: int
    coherence_distribution: dict[int, int]   # score → count
    usefulness_distribution: dict[int, int]
    mean_coherence: float | None
    mean_usefulness: float | None


# ---------------------------------------------------------------------------
# Blinding helper
# ---------------------------------------------------------------------------

def _blind_row(proposal: ScoredProposal) -> ProposalRow:
    """Strip arm identity from a proposal before sending to a rater."""
    row = ProposalRow(
        proposal_id=proposal.composition_id,
        constituents=proposal.constituents,
        scores=ScoreBadges(
            novelty_score=proposal.novelty_score,
            plausibility_score=proposal.plausibility_score,
            frontier_difficulty_score=proposal.frontier_difficulty_score,
            final_rank_score=proposal.final_rank_score,
        ),
        motivating_scene_count=len(proposal.motivating_scene_ids),
    )
    # Critical: arm must never leave this server in a response to the rater.
    # If ProposalRow ever gains an arm field by accident, this assertion fires.
    assert "arm" not in row.model_dump(), (
        "BUG: arm leaked into ProposalRow — blinded eval is invalid. Fix immediately."
    )
    return row


def _blind_detail(proposal: ScoredProposal) -> ProposalDetail:
    """Strip arm identity from a proposal for the detail view."""
    detail = ProposalDetail(
        proposal_id=proposal.composition_id,
        constituents=proposal.constituents,
        scores=ScoreBadges(
            novelty_score=proposal.novelty_score,
            plausibility_score=proposal.plausibility_score,
            frontier_difficulty_score=proposal.frontier_difficulty_score,
            final_rank_score=proposal.final_rank_score,
        ),
        plausibility_justification=proposal.plausibility_justification,
        motivating_scenes=[
            MotivatingScene(segment_id=k.segment_id, window_idx=k.window_idx)
            for k in proposal.motivating_scene_ids
        ],
        rejection_reason=proposal.rejection_reason,
    )
    assert "arm" not in detail.model_dump(), (
        "BUG: arm leaked into ProposalDetail — blinded eval is invalid. Fix immediately."
    )
    return detail


# ---------------------------------------------------------------------------
# Ratings persistence helpers
# ---------------------------------------------------------------------------

def _ratings_dir(rater_id: str) -> Path:
    d = config.JUDGE_RATINGS_DIR / rater_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _rating_path(rater_id: str, proposal_id: str) -> Path:
    # Guard against path traversal: composition_id is a hex hash so '/' should
    # never appear, but check explicitly to catch malformed upstream data early.
    if "/" in proposal_id or "\\" in proposal_id:
        raise ValueError(
            f"proposal_id contains path separator: {proposal_id!r}. "
            "composition_id must be a flat hex string."
        )
    return _ratings_dir(rater_id) / f"{proposal_id}.json"


def _persist_rating(rating: Rating) -> None:
    """Atomic write: write to .tmp then rename."""
    path = _rating_path(rating.rater_id, rating.proposal_id)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rating.to_json(), indent=2), encoding="utf-8")
    tmp.replace(path)


def _load_ratings_for_rater(rater_id: str) -> list[Rating]:
    d = config.JUDGE_RATINGS_DIR / rater_id
    if not d.exists():
        return []
    ratings = []
    for f in d.glob("*.json"):
        try:
            ratings.append(Rating.from_json(json.loads(f.read_text(encoding="utf-8"))))
        except Exception as exc:
            print(f"[JudgeUI] WARNING: could not parse rating file {f}: {exc}")
    return ratings


def _load_all_ratings() -> list[Rating]:
    if not config.JUDGE_RATINGS_DIR.exists():
        return []
    ratings = []
    for rater_dir in config.JUDGE_RATINGS_DIR.iterdir():
        if not rater_dir.is_dir():
            continue
        for f in rater_dir.glob("*.json"):
            try:
                ratings.append(Rating.from_json(json.loads(f.read_text(encoding="utf-8"))))
            except Exception as exc:
                print(f"[JudgeUI] WARNING: could not parse rating file {f}: {exc}")
    return ratings


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/judge/proposals", response_model=list[ProposalRow])
def list_proposals() -> list[ProposalRow]:
    """Return all accepted proposals in rank order, arm blinded."""
    return [_blind_row(_proposals[pid]) for pid in _accepted_ids]


@app.get("/judge/proposals/{proposal_id}", response_model=ProposalDetail)
def get_proposal(proposal_id: str) -> ProposalDetail:
    """Return full detail for one accepted proposal, arm blinded."""
    proposal = _proposals.get(proposal_id)
    if proposal is None:
        raise HTTPException(status_code=404, detail=f"Proposal {proposal_id!r} not found")
    if not proposal.accepted:
        raise HTTPException(status_code=404, detail=f"Proposal {proposal_id!r} was not accepted")
    return _blind_detail(proposal)


@app.get("/judge/video-url", response_model=VideoUrlResponse)
def get_video_url(
    segment_id: str = Query(...),
    window_idx: int = Query(...),
    camera: str = Query(default="FRONT"),
) -> VideoUrlResponse:
    """Return a fresh pre-signed GCS URL for a motivating-scene video.

    The frontend calls this on initial load and again on video.onerror
    (capped at 2 retries) to handle expired URLs.
    """
    if _storage is None:
        raise HTTPException(
            status_code=503,
            detail="Video storage not configured. Set JUDGE_BUCKET_URI env var.",
        )
    try:
        url = _storage.get_window_video_url(
            segment_id=segment_id,
            window_idx=window_idx,
            camera=camera,
            ttl_seconds=config.VIDEO_URL_TTL_SECONDS,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Storage error: {exc}",
        ) from exc

    return VideoUrlResponse(
        url=url,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/judge/ratings")
def submit_rating(submission: RatingSubmission) -> dict[str, bool]:
    """Persist a rater's evaluation of one proposal.

    arm is injected server-side from the proposal store — the rater never
    provides it. Duplicate submissions overwrite the previous rating (idempotent).
    Both coherence_score and usefulness_score are required (validated by Pydantic).
    """
    proposal = _proposals.get(submission.proposal_id)
    if proposal is None:
        raise HTTPException(
            status_code=404,
            detail=f"Proposal {submission.proposal_id!r} not found",
        )
    if not proposal.accepted:
        raise HTTPException(
            status_code=422,
            detail=f"Proposal {submission.proposal_id!r} is not in the accepted queue",
        )

    rating = Rating(
        rater_id=submission.rater_id,
        proposal_id=submission.proposal_id,
        arm=proposal.arm,  # server-side; never from rater input
        coherence_score=submission.coherence_score,
        usefulness_score=submission.usefulness_score,
        timestamp=datetime.now(timezone.utc).isoformat(),
        free_text_note=submission.free_text_note,
        seen_motivating_scenes=[
            WindowKey(segment_id=s.segment_id, window_idx=s.window_idx)
            for s in submission.seen_motivating_scenes
        ],
    )

    _persist_rating(rating)
    return {"ok": True}


@app.get("/judge/session/{rater_id}", response_model=SessionSummary)
def get_session(rater_id: str) -> SessionSummary:
    """Return this rater's progress: which proposals they've rated + score distributions.

    The frontend uses rated_proposal_ids to mark already-rated rows in the
    ProposalList (Scenario A resumability — rows are marked, not skipped).
    """
    ratings = _load_ratings_for_rater(rater_id)

    coh_dist: dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    use_dist: dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    rated_ids: list[str] = []

    for r in ratings:
        rated_ids.append(r.proposal_id)
        coh_dist[r.coherence_score] = coh_dist.get(r.coherence_score, 0) + 1
        use_dist[r.usefulness_score] = use_dist.get(r.usefulness_score, 0) + 1

    n = len(ratings)
    mean_coh = (
        sum(r.coherence_score for r in ratings) / n if n > 0 else None
    )
    mean_use = (
        sum(r.usefulness_score for r in ratings) / n if n > 0 else None
    )

    return SessionSummary(
        rater_id=rater_id,
        rated_proposal_ids=rated_ids,
        total_accepted=len(_accepted_ids),
        coherence_distribution=coh_dist,
        usefulness_distribution=use_dist,
        mean_coherence=mean_coh,
        mean_usefulness=mean_use,
    )


@app.get("/judge/ratings/export")
def export_ratings() -> list[dict]:
    """Export all ratings across all raters as JSON — consumed by Module 6 (Evaluation).

    Module 6 calls this endpoint rather than reading the ratings directory
    directly, preserving the lego-block boundary.
    """
    return [r.to_json() for r in _load_all_ratings()]
