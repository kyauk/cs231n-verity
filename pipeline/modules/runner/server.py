"""FastAPI runner for the Verity frontend — modules edition.

Serves the Cluster Space / Scenes / Dashboard / Analysis tabs off the canonical
``outputs/waymo`` artifacts (clusters.json, flagged_windows.jsonl, reasoning/*)
and routes agentic analysis through ``pipeline.modules.debate`` — the
non-deprecated debate module — instead of the removed waymo_pipeline subprocess.

Endpoints
  GET  /health                 -- connectivity probe
  GET  /gpu                    -- L40S arbiter state
  GET  /batches                -- embedding/cluster batch history
  POST /batches                -- launch a cluster batch (Embed1 window)
  GET  /probe-path             -- count parquet segments under a GCS URI
  GET  /cluster-space          -- 3D points + per-cluster stats
  GET  /scenes/{scene_id}      -- one scene's detail + video URL
  GET  /scenarios              -- flagged scenarios for the dashboard
  GET  /video/{seg}/{idx}      -- stream a window clip from GCS (Range)
  GET  /segment-video/{seg}    -- stream a raw segment clip from GCS (Range)
  POST /analysis/run-stream    -- SSE: agentic analysis via pipeline.modules.debate

Run:
  uvicorn pipeline.modules.runner.server:app --host 0.0.0.0 --port 8000
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

from pipeline.modules.runner import store
from pipeline.modules.runner.gpu_arbiter import GpuBusyError, gpu

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS_ROOT = PROJECT_ROOT / "outputs" / "waymo"

_VIDEO_API_BASE = os.environ.get("VERITY_PUBLIC_API_URL", "").rstrip("/")
_VIDEO_BUCKET_URI = os.environ.get("VERITY_BUCKET", "gs://nvidia-adr-waymo-segment-videos/verity")
_SEGMENT_VIDEO_BUCKET = os.environ.get("SEGMENT_VIDEO_BUCKET", "nvidia-adr-waymo-segment-videos")
_WAYMO_SOURCE_BUCKET = os.environ.get("WAYMO_SOURCE_BUCKET", "waymo_open_dataset_v_2_0_1")

load_dotenv(PROJECT_ROOT / ".env", override=True)

CAMERA_NAMES = ("FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT")
DEFAULT_REGRESSION_SUITE = [
    "Night-time right turn at signalized intersection.",
    "Pedestrian crossing in rain with limited visibility.",
    "Unprotected left turn with cross traffic.",
    "Vehicle emerging from occluded driveway.",
]

app = FastAPI(title="Verity Runner (modules)")
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
    return datetime.now(timezone.utc).isoformat()


def _segment_index() -> dict[str, dict[str, str]]:
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
    cameras = _segment_index().get(log_id, {})
    return cameras.get(camera) or cameras.get("FRONT") or ""


_STATS_CACHE: dict[str, dict[str, str]] = {}


def _waymo_stats(segment_id: str) -> dict[str, str]:
    """Weather / time-of-day / location from the Waymo `stats` component (GT)."""
    if segment_id in _STATS_CACHE:
        return _STATS_CACHE[segment_id]
    result: dict[str, str] = {}
    try:
        import io  # noqa: PLC0415
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
                result["weather"] = weather.title()
            if tod:
                result["timeOfDay"] = tod
            if loc:
                result["location"] = loc.replace("location_", "").upper()
            break
    except Exception as exc:  # noqa: BLE001
        print(f"[runner] Waymo stats lookup failed for {segment_id}: {exc}", file=sys.stderr)
    _STATS_CACHE[segment_id] = result
    return result


# ---------------------------------------------------------------------------
# Health / GPU
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "dataset": "waymo", "runner": "pipeline.modules.runner"}


@app.get("/gpu")
def gpu_status() -> JSONResponse:
    return JSONResponse(gpu.status())


# ---------------------------------------------------------------------------
# Batches  (Ingest tab)
# ---------------------------------------------------------------------------

class LaunchBatchRequest(BaseModel):
    dataSourceUri: str
    label: str
    region: str
    maxSegments: int = 5
    mode: str = "cluster"


@app.get("/batches")
def list_batches() -> JSONResponse:
    batches = store.read("batches", [])
    batches_sorted = sorted(batches, key=lambda b: b.get("startedAt", ""), reverse=True)
    return JSONResponse({"batches": batches_sorted})


def _run_cluster_batch(batch_id: str) -> None:
    """Cluster the canonical /verity windows in an Embed1 window (cluster-only).

    Uses the already-ingested windows + embeddings (the user keeps clusters), so
    no parquet ingest and no Reason/Judge stage — the judge panel is never
    touched. Reason1 is restored when the embed window exits.
    """
    def _update(**fields: Any) -> None:
        batches = store.read("batches", [])
        for b in batches:
            if b.get("id") == batch_id:
                b.update(fields)
        store.write("batches", batches)

    bucket = os.environ.get("VERITY_BUCKET", "gs://nvidia-adr-waymo-segment-videos/verity")
    session = OUTPUTS_ROOT / "sessions" / batch_id
    session.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(PROJECT_ROOT)}
    try:
        with gpu.embed_window(on_stage=lambda s: _update(stage=f"gpu:{s}")):
            proc = subprocess.run(
                [sys.executable, "-u", "-m", "pipeline.run", "cluster",
                 "--bucket", bucket, "--output", str(session), "--cameras", "FRONT"],
                cwd=str(PROJECT_ROOT), env=env, capture_output=True, text=True,
            )
        if proc.returncode != 0:
            _update(status="failed", stage="cluster", completedAt=_now_iso(),
                    error=(proc.stderr or proc.stdout)[-2000:])
            return
        produced = session / "clusters.json"
        if produced.exists():
            (OUTPUTS_ROOT / "clusters.json").write_text(produced.read_text())
            n = len(json.loads(produced.read_text()).get("assignments", []))
            _ensure_flagged_windows(force=True)
            _update(status="completed", stage="done", completedAt=_now_iso(),
                    scenesProcessed=n, totalScenes=n)
        else:
            _update(status="failed", stage="cluster", completedAt=_now_iso(),
                    error="cluster produced no clusters.json")
    except Exception as error:  # noqa: BLE001
        _update(status="failed", error=str(error), completedAt=_now_iso())


@app.post("/batches")
async def launch_batch(req: LaunchBatchRequest) -> JSONResponse:
    batch_id = uuid.uuid4().hex[:12]
    record = {
        "id": batch_id,
        "label": req.label,
        "region": req.region,
        "dataSourceUri": req.dataSourceUri,
        "maxSegments": req.maxSegments,
        "mode": "cluster",
        "status": "running",
        "stage": "cluster",
        "startedAt": _now_iso(),
    }
    store.append_list("batches", record)
    asyncio.get_event_loop().run_in_executor(None, _run_cluster_batch, batch_id)
    return JSONResponse({"batch": record})


@app.get("/probe-path")
def probe_path(uri: str) -> JSONResponse:
    """Count .parquet segments under a GCS URI (best-effort)."""
    try:
        import gcsfs  # noqa: PLC0415
        fs = gcsfs.GCSFileSystem(token="google_default")
        path = uri.removeprefix("gs://").rstrip("/")
        count = sum(1 for f in fs.ls(path) if str(f).endswith(".parquet"))
        return JSONResponse({"valid": count > 0, "segmentCount": count,
                             "detail": f"{count} parquet segment(s) found."})
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"valid": False, "segmentCount": 0, "detail": str(exc)})


# ---------------------------------------------------------------------------
# Cluster Space tab
# ---------------------------------------------------------------------------

@app.get("/cluster-space")
def cluster_space() -> JSONResponse:
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
            "id": sid, "x": float(coord[0]), "y": float(coord[1]), "z": float(coord[2]),
            "clusterId": cluster_label, "sceneId": sid, "isNoise": is_noise,
        })
        if not is_noise:
            per_cluster.setdefault(cluster_label, []).append(float(a.get("probability", 0.0)))

    cluster_stats = [
        {"id": cid, "sceneCount": len(probs),
         "density": round(sum(probs) / len(probs), 4) if probs else 0.0}
        for cid, probs in sorted(per_cluster.items())
    ]
    return JSONResponse({"points": points, "clusterStats": cluster_stats})


@app.get("/scenes/{scene_id:path}")
def get_scene(scene_id: str) -> JSONResponse:
    flagged = _read_jsonl(OUTPUTS_ROOT / "flagged_windows.jsonl")
    flagged_by_window = {r.get("window_id"): r for r in flagged}
    row = flagged_by_window.get(scene_id, {})
    seg_from_id, _, widx_str = scene_id.rpartition("/")
    seg_from_id = seg_from_id or scene_id
    window_idx = int(widx_str) if widx_str.isdigit() else 0
    log_id = str(row.get("log_id") or seg_from_id)

    descriptions = _read_jsonl(OUTPUTS_ROOT / "reasoning" / "description_outputs.jsonl")
    latest_desc = next((d for d in reversed(descriptions) if d.get("window_id") == scene_id), None)

    annotations: dict[str, Any] = {
        "weather": "Unknown", "timeOfDay": "Unknown", "roadType": "Unknown",
        "actors": [], "events": row.get("scenario_tags", []) or [],
    }
    scene_description = ""
    if latest_desc:
        annotations["roadType"] = "See description"
        scene_description = latest_desc.get("scene_description", "")

    stats = _waymo_stats(seg_from_id)
    if stats.get("weather"):
        annotations["weather"] = stats["weather"]
    if stats.get("timeOfDay"):
        annotations["timeOfDay"] = stats["timeOfDay"]
    if stats.get("location"):
        annotations["location"] = stats["location"]

    video_url = (
        f"{_VIDEO_API_BASE}/video/{seg_from_id}/{window_idx}?camera=FRONT"
        if seg_from_id else _video_url_for(log_id, "FRONT")
    )
    return JSONResponse({
        "id": scene_id, "logId": log_id, "videoUrl": video_url, "thumbnail": "",
        "sceneDescription": scene_description, "annotations": annotations,
        "cameraUrls": _segment_index().get(log_id, {}),
    })


def _stream_gcs_blob(bucket_name: str, blob_name: str, request: Request, label: str) -> Response:
    """Stream a GCS blob via ADC with HTTP Range support."""
    import re  # noqa: PLC0415
    from google.cloud import storage  # noqa: PLC0415

    project = os.environ.get("GCS_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    try:
        blob = storage.Client(project=project).bucket(bucket_name).blob(blob_name)
        blob.reload()
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"detail": f"{label} not found ({blob_name}): {exc}"}, status_code=404)

    size = blob.size or 0
    range_header = request.headers.get("range") or request.headers.get("Range")
    if range_header and size:
        m = re.match(r"bytes=(\d+)-(\d*)", range_header)
        start = int(m.group(1)) if m else 0
        end = int(m.group(2)) if (m and m.group(2)) else size - 1
        end = min(end, size - 1)
        data = blob.download_as_bytes(start=start, end=end)
        return Response(
            content=data, status_code=206, media_type="video/mp4",
            headers={"Content-Range": f"bytes {start}-{end}/{size}", "Accept-Ranges": "bytes",
                     "Content-Length": str(end - start + 1), "Cache-Control": "public, max-age=3600"},
        )
    data = blob.download_as_bytes()
    return Response(
        content=data, status_code=200, media_type="video/mp4",
        headers={"Accept-Ranges": "bytes", "Content-Length": str(len(data)),
                 "Cache-Control": "public, max-age=3600"},
    )


@app.get("/video/{segment_id}/{window_idx}")
def video_proxy(segment_id: str, window_idx: int, request: Request, camera: str = "FRONT") -> Response:
    no_scheme = _VIDEO_BUCKET_URI.removeprefix("gs://").rstrip("/")
    bucket_name, _, prefix = no_scheme.partition("/")
    blob_name = f"{prefix}/windows/{segment_id}/{int(window_idx):04d}/camera_{camera}.mp4"
    return _stream_gcs_blob(bucket_name, blob_name, request, "window video")


@app.get("/segment-video/{segment_id}")
def segment_video(segment_id: str, request: Request, camera: str = "FRONT") -> Response:
    blob_name = f"segments/{segment_id}/{segment_id}_{camera}.mp4"
    return _stream_gcs_blob(_SEGMENT_VIDEO_BUCKET, blob_name, request, "segment video")


# ---------------------------------------------------------------------------
# Dashboard tab
# ---------------------------------------------------------------------------

def _risk_to_score(priority_score: float, risk_level: str) -> int:
    base = int(round(priority_score * 100))
    floor_by_risk = {"critical": 90, "high": 80, "medium": 70, "low": 0}
    return max(base, floor_by_risk.get(risk_level, 0))


@app.get("/scenarios")
def list_scenarios() -> JSONResponse:
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
            "priorityScore": _risk_to_score(float(proposal.get("confidence", 0.0)), risk_level),
            "definingConditions": str(proposal.get("evidence_summary", "")),
            "hasSimulationSpec": bool(proposal.get("recommended_test_spec")),
            "region": default_region,
        })
    scenarios.sort(key=lambda s: s["priorityScore"], reverse=True)
    return JSONResponse({"scenarios": scenarios})


# ---------------------------------------------------------------------------
# Flagged-window derivation + media manifest (shared with analysis)
# ---------------------------------------------------------------------------

def _ensure_flagged_windows(force: bool = False) -> bool:
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
                "window_id": window_id, "scene_token_hex": window_id, "log_id": seg,
                "scenario_tags": [], "cluster_label": cluster_label,
                "is_noise": cluster_label == -1,
                "cluster_probability": float(a.get("probability", 0.0) or 0.0),
                "outlier_score": _glosh(a), "anomaly_rank": rank, "quality": {},
                "metadata": {"dataset": "waymo", "derived_from": "clusters.json"},
            }) + "\n")
    return True


def _fetch_window_clip(scene_id: str) -> str:
    """Download the clicked window's FRONT clip from GCS via ADC; return local path."""
    seg, _, widx = scene_id.rpartition("/")
    seg = seg or scene_id
    window_idx = int(widx) if widx.isdigit() else 0
    clip_dir = OUTPUTS_ROOT / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    local_clip = clip_dir / f"{seg}_{window_idx:04d}_FRONT.mp4"
    if local_clip.exists() and local_clip.stat().st_size > 0:
        return str(local_clip)
    try:
        from google.cloud import storage  # noqa: PLC0415
        no_scheme = _VIDEO_BUCKET_URI.removeprefix("gs://").rstrip("/")
        bucket_name, _, prefix = no_scheme.partition("/")
        blob_name = f"{prefix}/windows/{seg}/{window_idx:04d}/camera_FRONT.mp4"
        project = os.environ.get("GCS_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
        storage.Client(project=project).bucket(bucket_name).blob(blob_name).download_to_filename(str(local_clip))
        return str(local_clip)
    except Exception as exc:  # noqa: BLE001
        print(f"[runner] window clip ADC download failed for {scene_id}: {exc}", file=sys.stderr)
        return ""


def _regression_suite() -> list[str]:
    suite_path = OUTPUTS_ROOT / "regression_suite.json"
    if suite_path.exists():
        try:
            data = json.loads(suite_path.read_text())
            if isinstance(data, list):
                return [str(x) for x in data]
        except (json.JSONDecodeError, OSError):
            pass
    return list(DEFAULT_REGRESSION_SUITE)


# ---------------------------------------------------------------------------
# Agentic analysis  (Analysis tab) -- SSE, via pipeline.modules.debate
# ---------------------------------------------------------------------------

class RunAnalysisRequest(BaseModel):
    sceneId: str
    debateRounds: int = 2


def _sse(obj: dict[str, Any]) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def _progress(step: str, title: str, detail: str) -> str:
    return _sse({"kind": "progress", "payload": {"step": step, "title": title, "detail": detail}})


@app.post("/analysis/run-stream")
async def run_analysis_stream(payload: RunAnalysisRequest) -> StreamingResponse:
    """Run the four-actor modules debate for the clicked window, streaming SSE."""
    reasoning_dir = OUTPUTS_ROOT / "reasoning"

    async def event_gen() -> Any:
        from pipeline.interfaces.debate import DebateInput
        from pipeline.modules.debate import (
            DebateConfig, Debater, NIMTextLLMClient, NIMVLMClient,
        )
        from pipeline.modules.debate.actors import run_tool_augmented_debate
        from pipeline.modules.debate.proposals import build_proposal_from_debate_output

        if not _ensure_flagged_windows():
            yield _sse({"kind": "error", "detail": "No clustered windows yet. Launch a batch first."})
            return
        rows = _read_jsonl(OUTPUTS_ROOT / "flagged_windows.jsonl")
        row = next((r for r in rows if r.get("window_id") == payload.sceneId), None)
        if row is None:
            yield _sse({"kind": "error", "detail": f"Scene {payload.sceneId!r} is not among the clustered windows."})
            return

        yield _progress("start", "Pipeline started", "Loading anomaly inputs and media paths.")

        clip = await asyncio.get_event_loop().run_in_executor(None, _fetch_window_clip, payload.sceneId)
        if not clip:
            yield _sse({"kind": "error", "detail": "Could not fetch the video clip for this scene from the bucket."})
            return

        # Reason1 lease — rejected (busy) if an ingest batch is currently embedding.
        try:
            reason_lease = gpu.reason_lease()
            reason_lease.__enter__()
        except GpuBusyError as busy:
            yield _sse({"kind": "error", "detail": str(busy)})
            return

        seg = payload.sceneId.rpartition("/")[0] or payload.sceneId
        run_id = uuid.uuid4().hex[:12]
        loop = asyncio.get_event_loop()
        try:
            text_client = NIMTextLLMClient()
            vlm_client = NIMVLMClient()
            cfg = DebateConfig(debate_rounds=max(1, int(payload.debateRounds)))
            debater = Debater(text_client, vlm_client, cfg)

            base_input = DebateInput(
                run_id=run_id,
                window_id=payload.sceneId,
                scene_token_hex=str(row.get("scene_token_hex") or payload.sceneId),
                log_id=str(row.get("log_id") or seg),
                scene_description="",
                anomaly_rationale=(
                    f"Clustering flagged this window as an outlier "
                    f"(GLOSH={float(row.get('outlier_score', 0.0)):.2f}, "
                    f"cluster={row.get('cluster_label')}, noise={row.get('is_noise')})."
                ),
                severity_hint="high" if row.get("is_noise") else "medium",
                regression_suite=_regression_suite(),
                media_refs=[clip],
            )

            yield _progress("describe", "Scene description",
                            f"Window {payload.sceneId}: analyzing the Waymo clip with the VLM...")
            description = await loop.run_in_executor(None, debater._describe, base_input)

            yield _progress("debate", "Multi-agent debate",
                            "Scene Analyst → Risk Assessor ↔ Coverage Analyst → Synthesis Arbiter...")
            debate_record = DebateInput(
                run_id=run_id,
                window_id=payload.sceneId,
                scene_token_hex=base_input.scene_token_hex,
                log_id=base_input.log_id,
                scene_description=description.scene_description,
                anomaly_rationale=description.anomaly_rationale,
                severity_hint=base_input.severity_hint,
                regression_suite=list(base_input.regression_suite),
                media_refs=[clip],
            )

            def _run_debate() -> Any:
                return run_tool_augmented_debate(
                    debate_record, media_refs=[clip],
                    text_client=text_client, vlm_client=vlm_client, config=cfg,
                )

            debate_output, _meta = await loop.run_in_executor(None, _run_debate)
            proposal = build_proposal_from_debate_output(debate_output, run_id)
        except Exception as error:  # noqa: BLE001
            yield _sse({"kind": "error", "detail": f"Analysis run failed: {error}"})
            return
        finally:
            reason_lease.__exit__(None, None, None)

        # Persist reasoning outputs (consumed by /scenes + /scenarios).
        yield _progress("save", "Saving results", "Writing reasoning outputs.")
        reasoning_dir.mkdir(parents=True, exist_ok=True)
        _append_jsonl(reasoning_dir / "description_outputs.jsonl", description.to_json())
        debate_row = {
            "window_id": debate_output.window_id,
            "decision": debate_output.decision,
            "recommendation": debate_output.recommendation,
            "priority_score": debate_output.priority_score,
            "rationale": debate_output.rationale,
            "model_source": debate_output.model_source,
            "metadata": debate_output.metadata,
        }
        _append_jsonl(reasoning_dir / "debate_outputs.jsonl", debate_row)
        _append_jsonl(reasoning_dir / "proposals.jsonl", proposal.to_json())

        # Map the four actors onto the UI's proposer/critic/judge slots.
        debate_history = (debate_output.metadata or {}).get("debate_history", [])
        proponent = "\n\n".join(
            h for h in debate_history if "[Scene Analyst]" in h or "[Risk Assessor]" in h
        )
        critic = "\n\n".join(h for h in debate_history if "[Coverage Analyst]" in h)
        judge = (debate_output.metadata or {}).get("judge_raw_output", "") or "\n\n".join(
            h for h in debate_history if "[Synthesis Arbiter]" in h
        )
        priority_score = _risk_to_score(float(debate_output.priority_score), str(proposal.risk_level))
        verdict = "PRIORITY SCENARIO" if debate_output.decision == "yes" else "NOT PRIORITY"

        yield _sse({
            "kind": "complete", "ok": True, "sceneId": payload.sceneId,
            "agentOutputs": {
                "proposer": proponent or "No proponent argument produced.",
                "critic": critic or "No critic argument produced.",
                "judge": judge or debate_output.rationale,
            },
            "conclusion": {
                "sceneId": payload.sceneId, "verdict": verdict,
                "priorityScore": priority_score,
                "simulationSpec": proposal.recommended_test_spec,
            },
            "sceneDescription": description.scene_description,
        })

    return StreamingResponse(
        event_gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
