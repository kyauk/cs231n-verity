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
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from waymo_pipeline import store
from waymo_pipeline.gpu_arbiter import GpuBusyError, gpu

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = Path(__file__).resolve().parent
OUTPUTS_ROOT = PROJECT_ROOT / "outputs" / "waymo"
PIPELINE_PROGRESS_PREFIX = "PIPELINE_PROGRESS:"

# The browser reaches /video through the single Next.js origin (which proxies to
# this runner), so the URL must be RELATIVE — never an absolute host, which would
# break under port-forwarding. Override only for an out-of-band deployment.
_VIDEO_API_BASE = os.environ.get("VERITY_PUBLIC_API_URL", "").rstrip("/")
_VIDEO_BUCKET_URI = os.environ.get("VERITY_BUCKET", "gs://nvidia-adr-waymo-segment-videos/verity")

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


_STATS_CACHE: dict[str, dict[str, str]] = {}
_WAYMO_SOURCE_BUCKET = os.environ.get("WAYMO_SOURCE_BUCKET", "waymo_open_dataset_v_2_0_1")


def _waymo_stats(segment_id: str) -> dict[str, str]:
    """Weather / time-of-day / location from the Waymo `stats` component (GT).

    These are dataset ground-truth labels — no reasoning model needed. Cached
    per segment; returns {} if the stats parquet can't be read.
    """
    if segment_id in _STATS_CACHE:
        return _STATS_CACHE[segment_id]
    result: dict[str, str] = {}
    try:
        import io  # noqa: PLC0415
        import sys  # noqa: PLC0415
        import pyarrow.parquet as pq  # noqa: PLC0415
        from google.cloud import storage  # noqa: PLC0415

        project = os.environ.get("GCS_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
        client = storage.Client(project=project)
        for split in ("validation", "training", "testing"):
            blob = client.bucket(_WAYMO_SOURCE_BUCKET).blob(f"{split}/stats/{segment_id}.parquet")
            if not blob.exists():
                continue
            df = pq.read_table(io.BytesIO(blob.download_as_bytes())).to_pandas()

            def _first(col: str) -> str:
                if col not in df.columns:
                    return ""
                vals = df[col].dropna()
                return str(vals.iloc[0]) if len(vals) else ""

            weather = _first("[StatsComponent].weather")
            tod = _first("[StatsComponent].time_of_day")
            loc = _first("[StatsComponent].location")
            if weather:
                result["weather"] = weather.title()           # "sunny" -> "Sunny"
            if tod:
                result["timeOfDay"] = tod                      # "Day" / "Night" / ...
            if loc:
                result["location"] = loc.replace("location_", "").upper()  # phx -> PHX
            break
    except Exception as exc:  # noqa: BLE001
        import sys  # noqa: PLC0415
        print(f"[runner] Waymo stats lookup failed for {segment_id}: {exc}", file=sys.stderr)
    _STATS_CACHE[segment_id] = result
    return result


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    """Lightweight health probe for the frontend connectivity check."""
    return {"status": "ok", "dataset": "waymo"}


@app.get("/gpu")
def gpu_status() -> JSONResponse:
    """Current L40S arbiter state.

    reason1 and embed1 share one GPU and cannot co-run. The frontend uses this
    to show GPU status and disable Analyze while an ingest batch is embedding.
    """
    return JSONResponse(gpu.status())


# ---------------------------------------------------------------------------
# Batches  (Ingest tab)
# ---------------------------------------------------------------------------

class LaunchBatchRequest(BaseModel):
    """Payload for launching a new embedding batch."""

    dataSourceUri: str
    label: str
    region: str
    maxSegments: int = 5
    # Which stages to run on the single GPU:
    #   "cluster" -> ingest + embed/cluster (Embed1)
    #   "reason"  -> ingest + discovery analyze -> Judge proposals (Reason1)
    #   "both"    -> ingest + cluster, then auto-swap to Reason1 and analyze
    mode: str = "both"


@app.get("/batches")
def list_batches() -> JSONResponse:
    """Return all embedding batch jobs, newest first."""
    batches = store.read("batches", [])
    batches_sorted = sorted(batches, key=lambda b: b.get("startedAt", ""), reverse=True)
    return JSONResponse({"batches": batches_sorted})


def _list_segment_ids(data_source_uri: str, n: int) -> list[str]:
    """List up to `n` segment IDs (parquet stems) under the source URI. n<=0 = all."""
    import gcsfs  # noqa: PLC0415
    fs = gcsfs.GCSFileSystem(token="google_default")
    path = data_source_uri.removeprefix("gs://").rstrip("/")
    ids: list[str] = []
    for f in fs.ls(path):
        name = str(f).split("/")[-1]
        if name.endswith(".parquet"):
            ids.append(name[: -len(".parquet")])
        if n and len(ids) >= n:
            break
    return ids


def _run_batch_pipeline(batch_id: str, data_source_uri: str, max_segments: int,
                        mode: str = "both") -> None:
    """Ingest, then run the GPU stage(s) selected by `mode`.

    mode == "cluster": ingest + embed/cluster (Embed1).
    mode == "reason":  ingest + discovery analyze -> Judge proposals (Reason1).
    mode == "both":    ingest + cluster, then the embed window closes (GPU auto-
                       swaps back to Reason1) and analyze runs — no manual swap.

    Ingest (shared Module 1) then cluster (Module 8) via the unified
    `pipeline.run` composition root — no separate waymo ingestion universe.

    Ingestion is an isolated checkpoint: if clustering fails (e.g. the embed NIM
    is down) the ingested windows persist and the batch is still usable by the
    Judge path. Per-subprocess credentials: ingest reads the Waymo SOURCE bucket
    with user ADC; cluster reads the OUTPUT bucket + signs URLs with the signer
    key (a user token can't sign; the signer SA can't read the public source).
    """
    def _update(**fields: Any) -> None:
        batches = store.read("batches", [])
        for b in batches:
            if b.get("id") == batch_id:
                b.update(fields)
        store.write("batches", batches)

    base_env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(PROJECT_ROOT)}
    bucket = os.environ.get("VERITY_BUCKET", "gs://nvidia-adr-waymo-segment-videos/verity")
    signer_key = os.environ.get("VERITY_SIGNER_KEY", "")
    session_dir = OUTPUTS_ROOT / "sessions" / batch_id
    session_dir.mkdir(parents=True, exist_ok=True)

    def _run(args: list[str], stage: str, env: dict) -> Any:
        _update(stage=stage)
        return subprocess.run(args, cwd=str(PROJECT_ROOT), env=env,
                              capture_output=True, text=True, check=False)

    try:
        seg_ids = _list_segment_ids(data_source_uri, max_segments or 0)
        if not seg_ids:
            _update(status="failed", stage="ingest", completedAt=_now_iso(),
                    error=f"No .parquet segments found under {data_source_uri}.")
            return

        # --- Stage 1: INGEST via canonical Module 1 (user ADC reads source) ---
        ingest_env = {**base_env, "GOOGLE_APPLICATION_CREDENTIALS": ""}
        ingest = _run([sys.executable, "-u", "-m", "pipeline.run", "ingest",
                       "--source-format", "waymo_parquet",
                       "--source-root", data_source_uri,
                       "--bucket", bucket,
                       "--segments", ",".join(seg_ids)], "ingest", ingest_env)
        if ingest.returncode != 0:
            _update(status="failed", stage="ingest", completedAt=_now_iso(),
                    error=(ingest.stderr or ingest.stdout)[-2000:])
            return
        # Ingestion checkpoint — persists independently of any downstream analysis.
        _update(stage="ingested", ingested=True, bucket=bucket, segments=seg_ids,
                scenesProcessed=len(seg_ids), totalScenes=len(seg_ids))

        n_windows = len(seg_ids)

        # --- Stage 2: CLUSTER via Module 8 (modes: cluster, both) -------------
        # Clustering needs embed1, which can't co-run with reason1 on the L40S.
        # Borrow the GPU for embed only here (ingest above is GPU-free). The window
        # swaps reason1 -> embed1 on enter and restores reason1 on exit even if
        # clustering fails (guaranteed resting state). For mode "both" this exit is
        # exactly what leaves us on reason1 for the analyze stage below — no manual
        # swap. Stage labels: gpu:draining -> gpu:swapping_to_embed -> gpu:embed_ready
        # (cluster runs) -> gpu:swapping_to_reason -> gpu:reason_ready.
        if mode in ("cluster", "both"):
            cluster_env = {**base_env, "GOOGLE_APPLICATION_CREDENTIALS": signer_key}
            try:
                with gpu.embed_window(on_stage=lambda s: _update(stage=f"gpu:{s}")):
                    cluster = _run([sys.executable, "-u", "-m", "pipeline.run", "cluster",
                                    "--bucket", bucket, "--output", str(session_dir),
                                    "--cameras", "FRONT"], "cluster", cluster_env)
            except Exception as swap_error:  # noqa: BLE001 — GPU swap failed; ingest persists
                _update(status="failed", stage="cluster", completedAt=_now_iso(),
                        error=f"GPU swap failed: {swap_error}")
                return
            if cluster.returncode != 0:
                _update(status="failed", stage="cluster", completedAt=_now_iso(),
                        error=(cluster.stderr or cluster.stdout)[-2000:])
                return
            produced = session_dir / "clusters.json"
            if produced.exists():
                (OUTPUTS_ROOT / "clusters.json").write_text(produced.read_text())
                n_windows = len(json.loads(produced.read_text()).get("assignments", []))
                _ensure_flagged_windows(force=True)  # link cluster -> analysis input

        # --- Stage 3: ANALYZE (discovery -> Judge proposals) (modes: reason, both) ---
        # Runs the encoder (Reason1) -> Hypothesizer -> Scorer over the ingested
        # windows. After the embed window above closed, the GPU is already back on
        # reason1; reason_lease just confirms that resting state (and blocks any
        # concurrent embed swap for the duration). User ADC reads the clips; the
        # encoder inlines gs:// clips as base64 for the local NIM.
        if mode in ("reason", "both"):
            analyze_env = {**base_env, "GOOGLE_APPLICATION_CREDENTIALS": ""}
            try:
                with gpu.reason_lease():
                    analyze = _run([sys.executable, "-u", "-m", "pipeline.run", "analyze",
                                    "--bucket", bucket, "--output", str(session_dir),
                                    "--cameras", "FRONT"], "analyze", analyze_env)
            except GpuBusyError as busy:
                _update(status="failed", stage="analyze", completedAt=_now_iso(),
                        error=f"GPU busy, could not start reasoning: {busy}")
                return
            if analyze.returncode != 0:
                _update(status="failed", stage="analyze", completedAt=_now_iso(),
                        error=(analyze.stderr or analyze.stdout)[-2000:])
                return
            scored = session_dir / "scored.json"
            if scored.exists():
                # Canonical path the Judge UI server serves.
                (OUTPUTS_ROOT / "judge_scored.json").write_text(scored.read_text())

        _update(status="completed", stage="done", completedAt=_now_iso(),
                scenesProcessed=n_windows, totalScenes=n_windows)
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

    mode = payload.mode if payload.mode in ("cluster", "reason", "both") else "both"
    batch_id = f"batch-{uuid.uuid4().hex[:8]}"
    record = {
        "id": batch_id,
        "label": payload.label,
        "dataSourceUri": payload.dataSourceUri,
        "region": payload.region,
        "mode": mode,
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
        None, _run_batch_pipeline, batch_id, payload.dataSourceUri, payload.maxSegments, mode
    )
    return JSONResponse({"batch": record}, status_code=201)


# ---------------------------------------------------------------------------
# Cluster space  (Cluster Space tab)
# ---------------------------------------------------------------------------

@app.get("/cluster-space")
def cluster_space() -> JSONResponse:
    """Return 3D cluster points and per-cluster statistics.

    Built from clusters.json (ClusterReport) produced by Module 8 (clustering).
    Each assignment has window_id {segment_id, window_idx}, cluster_id,
    glosh_score, probability, and coords_3d.
    """
    report_path = OUTPUTS_ROOT / "clusters.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {"assignments": []}

    points: list[dict[str, Any]] = []
    per_cluster: dict[int, list[float]] = {}
    for a in report.get("assignments", []):
        coord = a.get("coords_3d") or [0.0, 0.0, 0.0]
        wid = a.get("window_id") or {}
        sid = f"{wid.get('segment_id', '')}/{int(wid.get('window_idx', 0)):04d}"
        cluster_label = int(a.get("cluster_id", -1))
        is_noise = cluster_label == -1
        points.append({
            "id": sid,
            "x": float(coord[0]),
            "y": float(coord[1]),
            "z": float(coord[2]),
            "clusterId": cluster_label,
            "sceneId": sid,
            "isNoise": is_noise,
        })
        if not is_noise:
            per_cluster.setdefault(cluster_label, []).append(float(a.get("probability", 0.0)))

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

@app.get("/scenes/{scene_id:path}")
def get_scene(scene_id: str) -> JSONResponse:
    """Return one scene's detail: video URL, thumbnail, and annotations.

    Annotations are sourced from the latest reasoning output for the window
    when available; otherwise from anomaly scenario tags. The video is served
    through the runner's /video proxy (streams from GCS via ADC) so it plays
    without signed URLs or an SA key.
    """
    flagged = _read_jsonl(OUTPUTS_ROOT / "flagged_windows.jsonl")
    flagged_by_window = {r.get("window_id"): r for r in flagged}
    row = flagged_by_window.get(scene_id, {})
    # scene_id is the window id "segment_id/window_idx". Parse the full segment
    # id from it (NOT scene_id.split("_")[0], which truncates the Waymo context
    # name and breaks every lookup).
    seg_from_id, _, widx_str = scene_id.rpartition("/")
    seg_from_id = seg_from_id or scene_id
    window_idx = int(widx_str) if widx_str.isdigit() else 0
    log_id = str(row.get("log_id") or seg_from_id)

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

    # Weather + time-of-day come from Waymo `stats` ground truth (no reasoning
    # model needed) — override whatever the reasoning stage left. Road type isn't
    # a Waymo label, so it stays as-is until a Reason pass fills it in.
    stats = _waymo_stats(seg_from_id)
    if stats.get("weather"):
        annotations["weather"] = stats["weather"]
    if stats.get("timeOfDay"):
        annotations["timeOfDay"] = stats["timeOfDay"]
    if stats.get("location"):
        annotations["location"] = stats["location"]

    # Serve via the /video proxy (streams the window clip from GCS via ADC).
    # Falls back to a pre-signed segment_index URL only if present.
    video_url = (
        f"{_VIDEO_API_BASE}/video/{seg_from_id}/{window_idx}?camera=FRONT"
        if seg_from_id else _video_url_for(log_id, "FRONT")
    )
    return JSONResponse({
        "id": scene_id,
        "logId": log_id,
        "videoUrl": video_url,
        "thumbnail": "",
        "sceneDescription": scene_description,
        "annotations": annotations,
        "cameraUrls": _segment_index().get(log_id, {}),
    })


@app.get("/video/{segment_id}/{window_idx}")
def video_proxy(segment_id: str, window_idx: int, request: Request, camera: str = "FRONT") -> Response:
    """Stream a window's camera MP4 from GCS via ADC (no signed URL needed).

    Supports HTTP Range so the browser <video> can seek. The clip lives at
    {verity}/windows/{segment_id}/{window_idx:04d}/camera_{camera}.mp4.
    """
    import re  # noqa: PLC0415
    from google.cloud import storage  # noqa: PLC0415

    no_scheme = _VIDEO_BUCKET_URI.removeprefix("gs://").rstrip("/")
    bucket_name, _, prefix = no_scheme.partition("/")
    blob_name = f"{prefix}/windows/{segment_id}/{int(window_idx):04d}/camera_{camera}.mp4"

    project = os.environ.get("GCS_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    try:
        client = storage.Client(project=project)
        blob = client.bucket(bucket_name).blob(blob_name)
        blob.reload()  # populates size; raises if missing
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(
            {"detail": f"video not found for {segment_id}/{window_idx:04d} ({camera}): {exc}"},
            status_code=404,
        )

    size = blob.size or 0
    range_header = request.headers.get("range") or request.headers.get("Range")
    if range_header and size:
        m = re.match(r"bytes=(\d+)-(\d*)", range_header)
        start = int(m.group(1)) if m else 0
        end = int(m.group(2)) if (m and m.group(2)) else size - 1
        end = min(end, size - 1)
        data = blob.download_as_bytes(start=start, end=end)  # GCS end is inclusive
        return Response(
            content=data, status_code=206, media_type="video/mp4",
            headers={
                "Content-Range": f"bytes {start}-{end}/{size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(end - start + 1),
                "Cache-Control": "public, max-age=3600",
            },
        )

    data = blob.download_as_bytes()
    return Response(
        content=data, status_code=200, media_type="video/mp4",
        headers={"Accept-Ranges": "bytes", "Content-Length": str(len(data)),
                 "Cache-Control": "public, max-age=3600"},
    )


_SEGMENT_VIDEO_BUCKET = os.environ.get("SEGMENT_VIDEO_BUCKET", "nvidia-adr-waymo-segment-videos")


@app.get("/segment-video/{segment_id}")
def segment_video(segment_id: str, request: Request, camera: str = "FRONT") -> Response:
    """Stream a RAW segment MP4 (not a window clip) from GCS via ADC, with Range.

    The salience scenes reference whole segments at
    gs://{SEGMENT_VIDEO_BUCKET}/segments/{seg}/{seg}_{camera}.mp4 — a different
    layout than the windows/ clips video_proxy serves.
    """
    import re  # noqa: PLC0415
    from google.cloud import storage  # noqa: PLC0415

    blob_name = f"segments/{segment_id}/{segment_id}_{camera}.mp4"
    project = os.environ.get("GCS_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    try:
        blob = storage.Client(project=project).bucket(_SEGMENT_VIDEO_BUCKET).blob(blob_name)
        blob.reload()
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"detail": f"segment video not found: {blob_name} ({exc})"}, status_code=404)

    size = blob.size or 0
    range_header = request.headers.get("range") or request.headers.get("Range")
    if range_header and size:
        m = re.match(r"bytes=(\d+)-(\d*)", range_header)
        start = int(m.group(1)) if m else 0
        end = int(m.group(2)) if (m and m.group(2)) else size - 1
        end = min(end, size - 1)
        data = blob.download_as_bytes(start=start, end=end)
        return Response(content=data, status_code=206, media_type="video/mp4",
                        headers={"Content-Range": f"bytes {start}-{end}/{size}",
                                 "Accept-Ranges": "bytes", "Content-Length": str(end - start + 1),
                                 "Cache-Control": "public, max-age=3600"})
    data = blob.download_as_bytes()
    return Response(content=data, status_code=200, media_type="video/mp4",
                    headers={"Accept-Ranges": "bytes", "Content-Length": str(len(data)),
                             "Cache-Control": "public, max-age=3600"})


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


def _scene_scoped_flagged(scene_id: str) -> Path | None:
    """Write a 1-row flagged file for the CLICKED window so describe processes it.

    Without this, describe_and_debate --top-k 1 would analyze the global
    top-anomaly window (not the one the user clicked), and its manifest wouldn't
    match. Returns the path, or None if the window isn't among the clustered set.
    """
    rows = _read_jsonl(OUTPUTS_ROOT / "flagged_windows.jsonl")
    target = next((r for r in rows if r.get("window_id") == scene_id), None)
    if target is None:
        return None
    target = {**target, "anomaly_rank": 1}
    out = OUTPUTS_ROOT / "flagged_visuals" / "flagged_scene.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(target) + "\n", encoding="utf-8")
    return out


def _ensure_visual_manifest(scene_id: str) -> Path:
    """Build a media manifest mapping the clicked window to a LOCAL FRONT clip.

    Downloads the window's clip from GCS via ADC (verity/windows/{seg}/{idx:04d}/
    camera_FRONT.mp4) — no signed URL needed, works for any ingested segment.
    Falls back to a cached clip or a segment_index signed URL if present.
    """
    OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUTS_ROOT / "flagged_visuals" / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    seg, _, widx = scene_id.rpartition("/")
    seg = seg or scene_id
    window_idx = int(widx) if widx.isdigit() else 0

    clip_dir = OUTPUTS_ROOT / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    local_clip = clip_dir / f"{seg}_{window_idx:04d}_FRONT.mp4"
    mp4_path = ""

    if local_clip.exists() and local_clip.stat().st_size > 0:
        mp4_path = str(local_clip)
    else:
        # Download the window clip from GCS via ADC (the durable, key-free path).
        try:
            from google.cloud import storage  # noqa: PLC0415
            no_scheme = _VIDEO_BUCKET_URI.removeprefix("gs://").rstrip("/")
            bucket_name, _, prefix = no_scheme.partition("/")
            blob_name = f"{prefix}/windows/{seg}/{window_idx:04d}/camera_FRONT.mp4"
            project = os.environ.get("GCS_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
            storage.Client(project=project).bucket(bucket_name).blob(blob_name).download_to_filename(str(local_clip))
            mp4_path = str(local_clip)
        except Exception as exc:  # noqa: BLE001
            import sys  # noqa: PLC0415
            print(f"[runner] window clip ADC download failed for {scene_id}: {exc}", file=sys.stderr)
            # Fallback: legacy segment_index signed URL.
            signed_url = _segment_index().get(seg, {}).get("FRONT", "")
            if signed_url:
                try:
                    import urllib.request  # noqa: PLC0415
                    urllib.request.urlretrieve(signed_url, str(local_clip))
                    mp4_path = str(local_clip)
                except Exception:  # noqa: BLE001
                    mp4_path = ""

    manifest_path.write_text(
        json.dumps({"window_id": scene_id, "grid_path": "", "mp4_path": mp4_path}) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _ensure_flagged_windows(force: bool = False) -> bool:
    """Make flagged_windows.jsonl available, deriving it from clusters.json.

    The clustering stage (Module 8) writes clusters.json; the agentic-analysis
    stage consumes flagged_windows.jsonl (AnomalyResultRecord rows). They were
    never linked, so derive the flagged file from the cluster report's per-window
    outlier (GLOSH) scores. Returns True if flagged windows exist, False if there
    are no clusters at all.
    """
    flagged = OUTPUTS_ROOT / "flagged_windows.jsonl"
    if not force and flagged.exists() and flagged.stat().st_size > 0:
        return True
    clusters = OUTPUTS_ROOT / "clusters.json"
    if not clusters.exists():
        return False
    try:
        assignments = json.loads(clusters.read_text()).get("assignments", [])
    except (json.JSONDecodeError, OSError):
        return False
    if not assignments:
        return False

    def _glosh(a: dict) -> float:
        return float(a.get("glosh_score", 0.0) or 0.0)

    ranked = sorted(assignments, key=_glosh, reverse=True)
    flagged.parent.mkdir(parents=True, exist_ok=True)
    with open(flagged, "w", encoding="utf-8") as f:
        for rank, a in enumerate(ranked, start=1):
            wid = a.get("window_id") or {}
            seg = str(wid.get("segment_id", ""))
            idx = int(wid.get("window_idx", 0))
            window_id = f"{seg}/{idx:04d}"
            cluster_label = int(a.get("cluster_id", -1))
            f.write(json.dumps({
                "window_id": window_id,
                "scene_token_hex": window_id,
                "log_id": seg,
                "scenario_tags": [],
                "cluster_label": cluster_label,
                "is_noise": cluster_label == -1,
                "cluster_probability": float(a.get("probability", 0.0) or 0.0),
                "outlier_score": _glosh(a),
                "anomaly_rank": rank,
                "quality": {},
                "metadata": {"dataset": "waymo", "derived_from": "clusters.json"},
            }) + "\n")
    return True


@app.post("/analysis/run-stream")
async def run_analysis_stream(payload: RunAnalysisRequest) -> StreamingResponse:
    """Run agentic analysis for one scene, streaming progress as SSE events.

    Event kinds (matching the reference contract):
      {"kind":"progress","payload":{"step","title","detail"}}
      {"kind":"complete", ...analysis result...}
      {"kind":"error","detail": ...}
    """
    suite_path = _ensure_regression_suite()
    reasoning_dir = OUTPUTS_ROOT / "reasoning"
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(PROJECT_ROOT)}

    async def event_gen() -> Any:
        # 1. Link cluster output -> analysis input (derive flagged file if needed).
        if not _ensure_flagged_windows():
            yield f"data: {json.dumps({'kind': 'error', 'detail': 'No clustered windows yet. Launch a batch first.'})}\n\n"
            return
        # 2. Scope the run to the CLICKED window so describe + manifest agree.
        scene_flagged = _scene_scoped_flagged(payload.sceneId)
        if scene_flagged is None:
            yield f"data: {json.dumps({'kind': 'error', 'detail': f'Scene {payload.sceneId!r} is not among the clustered windows.'})}\n\n"
            return
        # 3. Fetch the window clip (via ADC) and confirm it landed.
        manifest_path = _ensure_visual_manifest(payload.sceneId)
        try:
            media_ok = any(
                json.loads(line).get("mp4_path")
                for line in manifest_path.read_text().splitlines() if line.strip()
            )
        except (OSError, json.JSONDecodeError):
            media_ok = False
        if not media_ok:
            yield f"data: {json.dumps({'kind': 'error', 'detail': 'Could not fetch the video clip for this scene from the bucket (expected verity/windows/<segment>/<idx>/camera_FRONT.mp4).'})}\n\n"
            return

        args = [
            sys.executable, "-u", "-m", "waymo_pipeline.waymo_describe_and_debate",
            "--flagged-jsonl", str(scene_flagged),
            "--visual-manifest-jsonl", str(manifest_path),
            "--regression-suite-json", str(suite_path),
            "--output-dir", str(reasoning_dir),
            "--top-k", "1",
            "--debate-rounds", str(payload.debateRounds),
        ]
        # Reason1 runs the VLM debate; it shares the L40S with embed1 and can't
        # co-run. Take the reason lease for the subprocess's lifetime: if an ingest
        # batch is currently embedding (reason1 down), reject with "busy"; otherwise
        # hold it so a batch launched now drains/waits for us before swapping.
        try:
            reason_lease = gpu.reason_lease()
            reason_lease.__enter__()
        except GpuBusyError as busy:
            yield f"data: {json.dumps({'kind': 'error', 'detail': str(busy)})}\n\n"
            return
        try:
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
        finally:
            reason_lease.__exit__(None, None, None)
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
        # The tool-augmented debate uses four actors. Map them onto the
        # frontend's three-slot view: Scene Analyst + Risk Assessor argue for
        # inclusion (proposer), the Coverage Analyst is the skeptic (critic),
        # and the Synthesis Arbiter's raw output is the judge verdict.
        proponent = "\n\n".join(
            h for h in debate_history
            if "[Scene Analyst]" in h or "[Risk Assessor]" in h
        )
        critic = "\n\n".join(h for h in debate_history if "[Coverage Analyst]" in h)
        judge = (latest_debate.get("metadata", {}) or {}).get("judge_raw_output", "")
        if not judge:
            judge = "\n\n".join(h for h in debate_history if "[Synthesis Arbiter]" in h)

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
