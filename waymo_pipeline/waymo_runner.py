"""FastAPI runner for the Waymo discovery pipeline.

Exposes the HTTP interface the Verity frontend consumes. Follows the
conventions established in
``relevant_video_debate_files/pipeline/remote_gpu_runner.py``:
permissive CORS, a /health probe, JSON responses, and an SSE-streaming
endpoint that forwards ``PIPELINE_PROGRESS:`` lines emitted by the pipeline
subprocess.

Endpoints
  GET  /health                          -- connectivity probe
  GET  /batches                         -- list embedding batch jobs
  POST /batches                         -- launch a new batch job
  GET  /cluster-space                   -- 3D points + per-cluster stats
  GET  /scenes/{scene_id}               -- one scene's detail + video URL
  GET  /scenarios                       -- flagged scenarios for the dashboard
  POST /analysis/run-stream             -- SSE: agentic analysis for a scene

Run:
  cd waymo-pipeline
  uvicorn waymo_runner:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from waymo_pipeline import store

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = Path(__file__).resolve().parent
OUTPUTS_ROOT = PROJECT_ROOT / "outputs" / "waymo"
PIPELINE_PROGRESS_PREFIX = "PIPELINE_PROGRESS:"

load_dotenv(PROJECT_ROOT / ".env", override=True)

# Waymo camera rig -- FRONT is the primary view shown in the scene modal.
CAMERA_NAMES = ("FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT")

DEFAULT_REGRESSION_SUITE = [
    "Night-time right turn at signalized intersection.",
    "Pedestrian crossing in rain with limited visibility.",
    "Unprotected left turn with cross traffic.",
    "Vehicle emerging from occluded driveway.",
]

app = FastAPI(title="Waymo Discovery Runner")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dict rows; empty list when missing."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def _now_iso() -> str:
    """Current UTC timestamp in ISO-8601 form."""
    return datetime.now(timezone.utc).isoformat()


def _segment_index() -> dict[str, dict[str, str]]:
    """Load the Waymo segment_index.json (segment_id -> {camera: signed_url}).

    Looks under outputs/waymo/ then the project root. Returns {} if absent --
    the runner stays functional even before the MP4 stage has been run.
    """
    for candidate in (OUTPUTS_ROOT / "segment_index.json", PROJECT_ROOT / "segment_index.json"):
        if candidate.exists():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, OSError):
                pass
    return {}


def _video_url_for(log_id: str, camera: str = "FRONT") -> str:
    """Resolve a playable video URL for a segment/camera from the segment index."""
    index = _segment_index()
    cameras = index.get(log_id, {})
    return cameras.get(camera) or cameras.get("FRONT") or ""


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    """Lightweight health probe for the frontend connectivity check."""
    return {"status": "ok", "dataset": "waymo"}


# ---------------------------------------------------------------------------
# Batches  (Ingest tab)
# ---------------------------------------------------------------------------

class LaunchBatchRequest(BaseModel):
    """Payload for launching a new embedding batch."""

    dataSourceUri: str
    label: str
    region: str
    maxSegments: int = 5


@app.get("/batches")
def list_batches() -> JSONResponse:
    """Return all embedding batch jobs, newest first."""
    batches = store.read("batches", [])
    batches_sorted = sorted(batches, key=lambda b: b.get("startedAt", ""), reverse=True)
    return JSONResponse({"batches": batches_sorted})


def _run_batch_pipeline(batch_id: str, data_source_uri: str, max_segments: int) -> None:
    """Run extraction + embedding + clustering for one batch (background task).

    Updates the batch record's status/progress as stages complete. On any
    failure the batch is marked ``failed`` so the UI reflects it.
    """
    def _update(**fields: Any) -> None:
        batches = store.read("batches", [])
        for b in batches:
            if b.get("id") == batch_id:
                b.update(fields)
        store.write("batches", batches)

    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(PROJECT_ROOT)}
    OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    mp4_dir = OUTPUTS_ROOT / "mp4s"
    mp4_dir.mkdir(parents=True, exist_ok=True)
    index_path = OUTPUTS_ROOT / "segment_index.json"
    scene_jsonl = OUTPUTS_ROOT / "waymo_scene_windows.jsonl"
    embed_jsonl = OUTPUTS_ROOT / "waymo_window_embeddings.jsonl"
    npz_path = OUTPUTS_ROOT / "waymo_clusters.npz"
    clusters_jsonl = OUTPUTS_ROOT / "waymo_clusters.jsonl"
    flagged_jsonl = OUTPUTS_ROOT / "flagged_windows.jsonl"

    stages = [
        # Stage 1: Parquet -> MP4 -> GCS + signed-URL index
        ([sys.executable, "-u", "-m", "waymo_pipeline.waymo_video_pipeline",
          "--num-segments", str(max_segments),
          "--out-dir", str(mp4_dir),
          "--index-out", str(index_path),
          *(["--data-source-uri", data_source_uri] if data_source_uri else [])],
         "mp4"),
        # Stage 2: GCS frame URIs -> SceneWindow JSONL
        ([sys.executable, "-u", "-m", "waymo_pipeline.waymo_extract_scene_windows",
          "--output-jsonl", str(scene_jsonl), "--max-segments", str(max_segments),
          "--data-source-uri", data_source_uri], "extract"),
        # Stage 3: SceneWindow JSONL -> Cosmos Embed1 -> embedding JSONL
        ([sys.executable, "-u", "-m", "waymo_pipeline.waymo_embed_scenes",
          "--input-jsonl", str(scene_jsonl), "--output-jsonl", str(embed_jsonl)], "embed"),
        # Stage 4: embeddings -> UMAP(50d) -> HDBSCAN -> 3D UMAP -> cluster JSONL
        ([sys.executable, "-u", "-m", "waymo_pipeline.waymo_cluster_embeddings",
          "--input-jsonl", str(embed_jsonl), "--output-npz", str(npz_path),
          "--output-jsonl", str(clusters_jsonl), "--flagged-jsonl", str(flagged_jsonl)], "cluster"),
    ]

    try:
        for args, stage_name in stages:
            _update(stage=stage_name)
            completed = subprocess.run(
                args, cwd=str(PROJECT_ROOT), env=env,
                capture_output=True, text=True, check=False,
            )
            if completed.returncode != 0:
                _update(
                    status="failed", stage=stage_name,
                    error=(completed.stderr or completed.stdout)[-2000:],
                    completedAt=_now_iso(),
                )
                return

        clusters = _read_jsonl(clusters_jsonl)
        _update(
            status="completed",
            stage="done",
            scenesProcessed=len(clusters),
            totalScenes=len(clusters),
            completedAt=_now_iso(),
        )
    except Exception as error:  # noqa: BLE001
        _update(status="failed", error=str(error), completedAt=_now_iso())


def _probe_gcs_path(uri: str) -> tuple[int, str]:
    """Return (count, error_detail). count=-1 on failure."""
    try:
        import gcsfs  # noqa: PLC0415
        fs = gcsfs.GCSFileSystem(token="google_default")
        path = uri.removeprefix("gs://")
        files = fs.ls(path)
        count = sum(1 for f in files if str(f).endswith(".parquet"))
        return count, ""
    except Exception as e:  # noqa: BLE001
        return -1, str(e)


@app.get("/probe-path")
async def probe_path(uri: str) -> JSONResponse:
    """Count Parquet files at a GCS URI without launching a batch.

    Returns {valid: bool, segmentCount: int, detail: str}.
    Used by the frontend to populate the segment slider after URI entry.
    """
    uri = uri.strip()
    if not uri.startswith("gs://") or len(uri) <= len("gs://"):
        return JSONResponse({"valid": False, "segmentCount": 0, "detail": "Not a valid gs:// path."})
    count, err = _probe_gcs_path(uri)
    if count < 0:
        return JSONResponse({"valid": False, "segmentCount": 0, "detail": f"Could not access path: {err}"})
    if count == 0:
        return JSONResponse({"valid": False, "segmentCount": 0, "detail": "No Parquet scene files found at this path."})
    return JSONResponse({"valid": True, "segmentCount": count, "detail": f"{count} segments available."})


@app.post("/batches")
async def launch_batch(payload: LaunchBatchRequest) -> JSONResponse:
    """Launch a new Waymo embedding batch and return the created record."""
    uri = payload.dataSourceUri.strip()

    # Basic format check
    if not uri.startswith("gs://") or len(uri) <= len("gs://"):
        return JSONResponse(
            {"detail": f"'{uri}' is not a valid GCS path. Must start with gs:// followed by a bucket and prefix."},
            status_code=400,
        )

    # Probe the path for Parquet files
    file_count, probe_err = _probe_gcs_path(uri)
    if file_count == 0:
        return JSONResponse(
            {"detail": f"No Parquet scene files found at '{uri}'. Check the path and make sure the bucket is accessible."},
            status_code=400,
        )

    batch_id = f"batch-{uuid.uuid4().hex[:8]}"
    record = {
        "id": batch_id,
        "label": payload.label,
        "dataSourceUri": payload.dataSourceUri,
        "region": payload.region,
        "status": "running",
        "stage": "queued",
        "scenesProcessed": 0,
        "totalScenes": None,
        "startedAt": _now_iso(),
        "completedAt": None,
        "error": None,
    }
    store.append_list("batches", record)

    # Run the pipeline off the request thread so the UI gets an immediate ack.
    asyncio.get_event_loop().run_in_executor(
        None, _run_batch_pipeline, batch_id, payload.dataSourceUri, payload.maxSegments
    )
    return JSONResponse({"batch": record}, status_code=201)


# ---------------------------------------------------------------------------
# Cluster space  (Cluster Space tab)
# ---------------------------------------------------------------------------

@app.get("/cluster-space")
def cluster_space() -> JSONResponse:
    """Return 3D cluster points and per-cluster statistics.

    Built from waymo_clusters.jsonl produced by the clustering stage. Each row
    has window_id, cluster_label, glosh_score, and coord_3d.
    """
    clusters = _read_jsonl(OUTPUTS_ROOT / "waymo_clusters.jsonl")

    points: list[dict[str, Any]] = []
    per_cluster: dict[int, list[float]] = {}
    for row in clusters:
        coord = row.get("coord_3d") or [0.0, 0.0, 0.0]
        cluster_label = int(row.get("cluster_label", -1))
        is_noise = cluster_label == -1
        points.append({
            "id": str(row.get("window_id", "")),
            "x": float(coord[0]),
            "y": float(coord[1]),
            "z": float(coord[2]),
            "clusterId": cluster_label,
            "sceneId": str(row.get("window_id", "")),
            "isNoise": is_noise,
        })
        if not is_noise:
            per_cluster.setdefault(cluster_label, []).append(
                float(row.get("cluster_probability", 0.0))
            )

    cluster_stats = [
        {
            "id": cid,
            "sceneCount": len(probs),
            "density": round(sum(probs) / len(probs), 4) if probs else 0.0,
        }
        for cid, probs in sorted(per_cluster.items())
    ]
    return JSONResponse({"points": points, "clusterStats": cluster_stats})


# ---------------------------------------------------------------------------
# Scene detail  (Cluster Space + Analysis tabs)
# ---------------------------------------------------------------------------

@app.get("/scenes/{scene_id}")
def get_scene(scene_id: str) -> JSONResponse:
    """Return one scene's detail: video URL, thumbnail, and annotations.

    Annotations are sourced from the latest reasoning output for the window
    when available; otherwise from anomaly scenario tags. The video URL is
    resolved from the Waymo segment index.
    """
    flagged = _read_jsonl(OUTPUTS_ROOT / "flagged_windows.jsonl")
    flagged_by_window = {r.get("window_id"): r for r in flagged}
    row = flagged_by_window.get(scene_id, {})
    log_id = str(row.get("log_id", scene_id.split("_")[0]))

    descriptions = _read_jsonl(OUTPUTS_ROOT / "reasoning" / "description_outputs.jsonl")
    latest_desc = next(
        (d for d in reversed(descriptions) if d.get("window_id") == scene_id), None
    )

    annotations: dict[str, Any]
    if latest_desc:
        annotations = {
            "weather": "Derived",
            "timeOfDay": "Derived",
            "roadType": "See description",
            "actors": [],
            "events": row.get("scenario_tags", []) or [],
        }
        scene_description = latest_desc.get("scene_description", "")
    else:
        annotations = {
            "weather": "Unknown",
            "timeOfDay": "Unknown",
            "roadType": "Unknown",
            "actors": [],
            "events": row.get("scenario_tags", []) or [],
        }
        scene_description = ""

    video_url = _video_url_for(log_id, "FRONT")
    return JSONResponse({
        "id": scene_id,
        "logId": log_id,
        "videoUrl": video_url,
        "thumbnail": "",
        "sceneDescription": scene_description,
        "annotations": annotations,
        "cameraUrls": _segment_index().get(log_id, {}),
    })


# ---------------------------------------------------------------------------
# Flagged scenarios  (Dashboard tab)
# ---------------------------------------------------------------------------

def _risk_to_score(priority_score: float, risk_level: str) -> int:
    """Convert a 0..1 priority score / risk level into the UI's 0..100 score."""
    base = int(round(priority_score * 100))
    floor_by_risk = {"critical": 90, "high": 80, "medium": 70, "low": 0}
    return max(base, floor_by_risk.get(risk_level, 0))


@app.get("/scenarios")
def list_scenarios() -> JSONResponse:
    """Return flagged scenarios assembled from regression-case proposals."""
    proposals = _read_jsonl(OUTPUTS_ROOT / "reasoning" / "proposals.jsonl")
    flagged = _read_jsonl(OUTPUTS_ROOT / "flagged_windows.jsonl")
    flagged_by_window = {r.get("window_id"): r for r in flagged}
    batches = store.read("batches", [])
    default_region = batches[-1]["region"] if batches else "US-West"

    scenarios: list[dict[str, Any]] = []
    for proposal in proposals:
        window_id = proposal.get("window_id", "")
        anomaly = flagged_by_window.get(window_id, {})
        risk_level = str(proposal.get("risk_level", "low"))
        scenarios.append({
            "id": str(proposal.get("case_id", window_id)),
            "scenarioName": str(proposal.get("failure_mode") or "Unnamed Scenario"),
            "clusterId": int(anomaly.get("cluster_label", -1)),
            "priorityScore": _risk_to_score(
                float(proposal.get("confidence", 0.0)), risk_level
            ),
            "definingConditions": str(proposal.get("evidence_summary", "")),
            "hasSimulationSpec": proposal.get("decision") == "add_to_suite",
            "region": default_region,
        })
    scenarios.sort(key=lambda s: s["priorityScore"], reverse=True)
    return JSONResponse({"scenarios": scenarios})


# ---------------------------------------------------------------------------
# Agentic analysis  (Analysis tab) -- SSE stream
# ---------------------------------------------------------------------------

class RunAnalysisRequest(BaseModel):
    """Payload for launching agentic analysis on a scene."""

    sceneId: str
    debateRounds: int = 2


def _ensure_regression_suite() -> Path:
    """Create the default regression suite JSON if it does not exist."""
    OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    suite_path = OUTPUTS_ROOT / "regression_suite.json"
    if not suite_path.exists():
        suite_path.write_text(json.dumps(DEFAULT_REGRESSION_SUITE, indent=2), encoding="utf-8")
    return suite_path


def _ensure_visual_manifest(scene_id: str) -> Path:
    """Build a media manifest mapping the scene window to its FRONT-camera clip.

    Tries to resolve a local MP4 clip in this order:
      1. Cached clip at outputs/waymo/clips/{log_id}_FRONT.mp4
      2. Local MP4 written by the video pipeline stage under outputs/waymo/mp4s/
      3. Signed GCS URL from segment_index.json — downloaded and cached locally
    """
    OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUTS_ROOT / "flagged_visuals" / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    flagged = _read_jsonl(OUTPUTS_ROOT / "flagged_windows.jsonl")
    row = next((r for r in flagged if r.get("window_id") == scene_id), {})
    log_id = str(row.get("log_id", scene_id.split("_")[0]))

    clip_dir = OUTPUTS_ROOT / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    local_clip = clip_dir / f"{log_id}_FRONT.mp4"

    # 1. Already cached
    if local_clip.exists():
        mp4_path = str(local_clip)
    else:
        # 2. Written locally by the video pipeline stage
        pipeline_clip = OUTPUTS_ROOT / "mp4s" / log_id / f"{log_id}_FRONT.mp4"
        if pipeline_clip.exists():
            mp4_path = str(pipeline_clip)
        else:
            # 3. Download from signed GCS URL and cache
            index = _segment_index()
            signed_url = index.get(log_id, {}).get("FRONT", "")
            if signed_url:
                try:
                    import urllib.request
                    urllib.request.urlretrieve(signed_url, str(local_clip))
                    mp4_path = str(local_clip)
                except Exception:  # noqa: BLE001
                    mp4_path = ""
            else:
                mp4_path = ""

    manifest_path.write_text(
        json.dumps({"window_id": scene_id, "grid_path": "", "mp4_path": mp4_path}) + "\n",
        encoding="utf-8",
    )
    return manifest_path


@app.post("/analysis/run-stream")
async def run_analysis_stream(payload: RunAnalysisRequest) -> StreamingResponse:
    """Run agentic analysis for one scene, streaming progress as SSE events.

    Event kinds (matching the reference contract):
      {"kind":"progress","payload":{"step","title","detail"}}
      {"kind":"complete", ...analysis result...}
      {"kind":"error","detail": ...}
    """
    suite_path = _ensure_regression_suite()
    manifest_path = _ensure_visual_manifest(payload.sceneId)
    flagged_jsonl = OUTPUTS_ROOT / "flagged_windows.jsonl"
    reasoning_dir = OUTPUTS_ROOT / "reasoning"

    args = [
        sys.executable, "-u", "-m", "waymo_pipeline.waymo_describe_and_debate",
        "--flagged-jsonl", str(flagged_jsonl),
        "--visual-manifest-jsonl", str(manifest_path),
        "--regression-suite-json", str(suite_path),
        "--output-dir", str(reasoning_dir),
        "--top-k", "1",
        "--debate-rounds", str(payload.debateRounds),
    ]
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(PROJECT_ROOT)}

    async def event_gen() -> Any:
        if not flagged_jsonl.exists():
            yield f"data: {json.dumps({'kind': 'error', 'detail': 'No clustered windows yet. Launch a batch first.'})}\n\n"
            return

        proc = await asyncio.create_subprocess_exec(
            *args, cwd=str(PROJECT_ROOT), env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        assert proc.stdout is not None
        log_lines: list[str] = []
        while True:
            raw = await proc.stdout.readline()
            if not raw:
                break
            text = raw.decode("utf-8", errors="replace").rstrip("\n\r")
            log_lines.append(text)
            if text.startswith(PIPELINE_PROGRESS_PREFIX):
                body = text[len(PIPELINE_PROGRESS_PREFIX):]
                try:
                    progress_payload = json.loads(body)
                except json.JSONDecodeError:
                    continue
                yield f"data: {json.dumps({'kind': 'progress', 'payload': progress_payload})}\n\n"

        code = await proc.wait()
        combined = "\n".join(log_lines)
        if code != 0:
            yield f"data: {json.dumps({'kind': 'error', 'code': code, 'detail': 'Analysis run failed.', 'logTail': combined[-8000:]})}\n\n"
            return

        # Assemble the final analysis result from the pipeline outputs.
        debate_rows = _read_jsonl(reasoning_dir / "debate_outputs.jsonl")
        description_rows = _read_jsonl(reasoning_dir / "description_outputs.jsonl")
        proposal_rows = _read_jsonl(reasoning_dir / "proposals.jsonl")

        latest_debate = next(
            (r for r in reversed(debate_rows) if r.get("window_id") == payload.sceneId), None
        )
        latest_description = next(
            (r for r in reversed(description_rows) if r.get("window_id") == payload.sceneId), None
        )
        latest_proposal = next(
            (r for r in reversed(proposal_rows) if r.get("window_id") == payload.sceneId), None
        )

        if not latest_debate or not latest_proposal:
            yield f"data: {json.dumps({'kind': 'error', 'detail': 'Analysis completed but produced no debate output.'})}\n\n"
            return

        debate_history = (latest_debate.get("metadata", {}) or {}).get("debate_history", [])
        proponent = "\n\n".join(h for h in debate_history if "Proponent" in h)
        critic = "\n\n".join(h for h in debate_history if "Critic" in h)
        judge = (latest_debate.get("metadata", {}) or {}).get("judge_raw_output", "")

        priority_score = _risk_to_score(
            float(latest_debate.get("priority_score", 0.0)),
            str(latest_proposal.get("risk_level", "low")),
        )
        verdict = (
            "PRIORITY SCENARIO"
            if latest_debate.get("decision") == "yes"
            else "NOT PRIORITY"
        )

        done = {
            "kind": "complete",
            "ok": True,
            "sceneId": payload.sceneId,
            "agentOutputs": {
                "proposer": proponent or "No proponent argument produced.",
                "critic": critic or "No critic argument produced.",
                "judge": judge or latest_debate.get("rationale", ""),
            },
            "conclusion": {
                "sceneId": payload.sceneId,
                "verdict": verdict,
                "priorityScore": priority_score,
                "simulationSpec": latest_proposal.get("recommended_test_spec", ""),
            },
            "sceneDescription": (latest_description or {}).get("scene_description", ""),
        }
        yield f"data: {json.dumps(done)}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
