"""Remote GPU runner API for video upload -> description/debate pipeline execution."""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# Must match pipeline.stage_describe_and_debate.PROGRESS_PREFIX for stream parsing.
PIPELINE_PROGRESS_PREFIX = "PIPELINE_PROGRESS:"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
INPUTS_ROOT = PROJECT_ROOT / "inputs"


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """
    Purpose: Read a JSONL file into a list of dictionaries.
    Parameters:
        path (Path): Absolute path to the JSONL file.
    Returns:
        list[dict[str, Any]]: Parsed JSON rows; empty list when missing or unreadable.
    Called by: load_latest_outputs()
    Calls: Path.read_text(), json.loads()
    """

    try:
        content = path.read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        return []

    rows: list[dict[str, Any]] = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def _as_string(value: Any, fallback: str = "") -> str:
    """
    Purpose: Normalize unknown values into strings with a fallback.
    Parameters:
        value (Any): Value to normalize.
        fallback (str): Default when value is not string.
    Returns:
        str: Normalized string.
    Called by: load_latest_outputs()
    Calls: None
    """

    return value if isinstance(value, str) else fallback


def _as_number(value: Any, fallback: float = 0.0) -> float:
    """
    Purpose: Normalize unknown values into floats with a fallback.
    Parameters:
        value (Any): Value to normalize.
        fallback (float): Default when value is not numeric.
    Returns:
        float: Normalized number.
    Called by: load_latest_outputs()
    Calls: None
    """

    return float(value) if isinstance(value, (int, float)) else fallback


def _as_string_array(value: Any) -> list[str]:
    """
    Purpose: Normalize unknown values into a list of strings.
    Parameters:
        value (Any): Input that may be a list.
    Returns:
        list[str]: List of string entries.
    Called by: load_latest_outputs()
    Calls: None
    """

    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def load_latest_outputs(window_id: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """
    Purpose: Build latest structured reasoning + flagged payloads for the uploaded window.
    Parameters:
        window_id (str): Window id emitted for current upload.
    Returns:
        tuple[dict[str, Any] | None, dict[str, Any] | None]: latestReasoning, latestFlagged.
    Called by: run_video()
    Calls: _read_jsonl_rows(), _as_string(), _as_number(), _as_string_array()
    """

    description_rows = _read_jsonl_rows(OUTPUTS_ROOT / "reasoning" / "description_outputs.jsonl")
    debate_rows = _read_jsonl_rows(OUTPUTS_ROOT / "reasoning" / "debate_outputs.jsonl")
    flagged_rows = _read_jsonl_rows(OUTPUTS_ROOT / "flagged_windows.jsonl")
    manifest_rows = _read_jsonl_rows(OUTPUTS_ROOT / "flagged_visuals" / "manifest.jsonl")

    latest_description = next(
        (row for row in reversed(description_rows) if _as_string(row.get("window_id")) == window_id),
        None,
    )
    latest_debate = next(
        (row for row in reversed(debate_rows) if _as_string(row.get("window_id")) == window_id),
        None,
    )
    latest_flagged_raw = next(
        (row for row in reversed(flagged_rows) if _as_string(row.get("window_id")) == window_id),
        None,
    )
    latest_manifest_raw = next(
        (row for row in reversed(manifest_rows) if _as_string(row.get("window_id")) == window_id),
        None,
    )

    debate_metadata = latest_debate.get("metadata", {}) if isinstance(latest_debate, dict) else {}
    if not isinstance(debate_metadata, dict):
        debate_metadata = {}

    latest_reasoning: dict[str, Any] | None = None
    if latest_debate and latest_description:
        latest_reasoning = {
            "windowId": _as_string(latest_debate.get("window_id")),
            "sceneDescription": _as_string(latest_description.get("scene_description"), "Description pending."),
            "anomalyRationale": _as_string(latest_description.get("anomaly_rationale"), "Rationale pending."),
            "decision": _as_string(latest_debate.get("decision"), "no"),
            "recommendation": _as_string(latest_debate.get("recommendation"), "not_critical"),
            "priorityScore": _as_number(latest_debate.get("priority_score"), 0.0),
            "modelSource": _as_string(latest_debate.get("model_source"), "unknown"),
            "capabilityTag": _as_string(debate_metadata.get("capability_tag")),
            "debateHistory": _as_string_array(debate_metadata.get("debate_history")),
            "judgeRawOutput": _as_string(debate_metadata.get("judge_raw_output")),
        }

    latest_flagged: dict[str, Any] | None = None
    if latest_flagged_raw:
        latest_flagged = {
            "windowId": _as_string(latest_flagged_raw.get("window_id")),
            "sceneTokenHex": _as_string(latest_flagged_raw.get("scene_token_hex")),
            "logId": _as_string(latest_flagged_raw.get("log_id")),
            "clusterLabel": int(_as_number(latest_flagged_raw.get("cluster_label"), -1)),
            "isNoise": bool(latest_flagged_raw.get("is_noise")),
            "outlierScore": _as_number(latest_flagged_raw.get("outlier_score"), 0.0),
            "anomalyRank": int(_as_number(latest_flagged_raw.get("anomaly_rank"), 0)),
            "gridUrl": _as_string((latest_manifest_raw or {}).get("grid_path")) or None,
            "mp4Url": _as_string((latest_manifest_raw or {}).get("mp4_path")) or None,
        }

    return latest_reasoning, latest_flagged


def is_mock_model_source(value: Any) -> bool:
    """
    Purpose: Detect mock/fallback model sources so they can be rejected.
    Parameters:
        value (Any): model_source value from reasoning output.
    Returns:
        bool: True when value indicates mock output.
    Called by: run_video()
    Calls: _as_string()
    """

    return "mock" in _as_string(value).lower()


def sanitize_filename(name: str) -> str:
    """
    Purpose: Normalize uploaded filename to a safe ascii subset.
    Parameters:
        name (str): Original file name from upload.
    Returns:
        str: Sanitized filename.
    Called by: run_video()
    Calls: re.sub()
    """

    return re.sub(r"[^a-zA-Z0-9._-]", "_", name)


def ensure_default_regression_suite() -> Path:
    """
    Purpose: Create a default regression suite JSON file if one does not exist.
    Parameters:
        None
    Returns:
        Path: Absolute path to regression suite JSON.
    Called by: run_video()
    Calls: Path.exists(), Path.write_text(), json.dumps()
    """

    suite_path = OUTPUTS_ROOT / "regression_suite.json"
    if suite_path.exists():
        return suite_path

    defaults = [
        "Night-time right turn at signalized intersection.",
        "Pedestrian crossing in rain with limited visibility.",
        "Unprotected left turn with cross traffic.",
        "Vehicle emerging from occluded driveway.",
    ]
    suite_path.write_text(json.dumps(defaults, indent=2), encoding="utf-8")
    return suite_path


def run_pipeline_command(args: list[str], env_overrides: dict[str, str]) -> tuple[int, str, str]:
    """
    Purpose: Execute stage pipeline command and capture stdout/stderr.
    Parameters:
        args (list[str]): CLI args including executable and module.
        env_overrides (dict[str, str]): Environment overrides for subprocess.
    Returns:
        tuple[int, str, str]: Return code, stdout, stderr.
    Called by: run_video()
    Calls: subprocess.run()
    """

    merged_env = dict(os.environ)
    merged_env.update(env_overrides)
    merged_env.setdefault("PYTHONUNBUFFERED", "1")
    completed = subprocess.run(
        args,
        cwd=str(PROJECT_ROOT),
        env=merged_env,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode, completed.stdout, completed.stderr


app = FastAPI(title="Remote GPU Runner")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    """
    Purpose: Provide a lightweight health endpoint for connectivity checks.
    Parameters:
        None
    Returns:
        dict[str, str]: Health status payload.
    Called by: External clients
    Calls: None
    """

    return {"status": "ok"}


@app.post("/run-video")
async def run_video(video: UploadFile = File(...)) -> JSONResponse:
    """
    Purpose: Accept a video upload, run description/debate pipeline, and return results.
    Parameters:
        video (UploadFile): Uploaded video file from client.
    Returns:
        JSONResponse: Success or failure payload with logs.
    Called by: Frontend remote forwarding route
    Calls: sanitize_filename(), ensure_default_regression_suite(), run_pipeline_command()
    """

    extension = Path(video.filename or "").suffix.lower()
    if extension not in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
        return JSONResponse({"detail": "Unsupported video type."}, status_code=400)

    OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_ROOT / "flagged_visuals").mkdir(parents=True, exist_ok=True)
    (OUTPUTS_ROOT / "reasoning").mkdir(parents=True, exist_ok=True)
    INPUTS_ROOT.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time() * 1000)
    safe_name = sanitize_filename(video.filename or f"upload{extension}")
    stored_name = f"upload_{timestamp}_{safe_name}"
    absolute_video_path = INPUTS_ROOT / stored_name
    relative_video_path = f"inputs/{stored_name}"

    file_bytes = await video.read()
    absolute_video_path.write_bytes(file_bytes)

    window_id = f"upload_window_{timestamp}"
    flagged_row: dict[str, Any] = {
        "window_id": window_id,
        "scene_token_hex": f"upload_{timestamp}",
        "log_id": "manual_upload",
        "scenario_tags": ["manual_upload"],
        "window_start_ts": 0,
        "window_end_ts": 0,
        "cluster_label": -1,
        "is_noise": True,
        "cluster_probability": 0.0,
        "outlier_score": 0.9,
        "anomaly_rank": 1,
        "quality": {},
        "metadata": {"upload_source": "remote_gpu_runner"},
    }
    manifest_row = {"window_id": window_id, "grid_path": "", "mp4_path": relative_video_path}

    (OUTPUTS_ROOT / "flagged_windows.jsonl").write_text(
        json.dumps(flagged_row) + "\n",
        encoding="utf-8",
    )
    (OUTPUTS_ROOT / "flagged_visuals" / "manifest.jsonl").write_text(
        json.dumps(manifest_row) + "\n",
        encoding="utf-8",
    )
    suite_path = ensure_default_regression_suite()

    max_new_tokens = os.getenv("WORKSPACE_MAX_NEW_TOKENS", "2400")
    debate_rounds = os.getenv("WORKSPACE_DEBATE_ROUNDS", "2")
    video_fps = os.getenv("WORKSPACE_VIDEO_FPS", "8")

    code, stdout, stderr = run_pipeline_command(
        [
            sys.executable,
            "-u",
            "-m",
            "pipeline.stage_describe_and_debate",
            "--flagged-jsonl",
            "outputs/flagged_windows.jsonl",
            "--visual-manifest-jsonl",
            "outputs/flagged_visuals/manifest.jsonl",
            "--regression-suite-json",
            str(suite_path.relative_to(PROJECT_ROOT)),
            "--output-dir",
            "outputs/reasoning",
            "--hf-max-new-tokens",
            max_new_tokens,
            "--top-k",
            "1",
            "--debate-rounds",
            debate_rounds,
        ],
        env_overrides={"COSMOS_HF_VIDEO_FPS": video_fps},
    )
    if code != 0:
        detail = (
            "Pipeline run failed.\n"
            f"python: {sys.executable}\n"
            f"stderr: {stderr.strip()[-2500:]}\n"
            f"stdout: {stdout.strip()[-1200:]}"
        )
        return JSONResponse(
            {
                "detail": detail,
                "stdout": stdout[-8000:],
                "stderr": stderr[-8000:],
            },
            status_code=500,
        )

    summary_path = OUTPUTS_ROOT / "reasoning" / "summary.json"
    try:
        reasoning_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        reasoning_summary = None
    latest_reasoning, latest_flagged = load_latest_outputs(window_id)
    if latest_reasoning and is_mock_model_source(latest_reasoning.get("modelSource")):
        return JSONResponse(
            {
                "detail": (
                    "Run completed but returned mock output. Real scene description is required. "
                    "Please sync latest pipeline files and restart runner."
                ),
                "latestReasoning": latest_reasoning,
                "latestFlagged": latest_flagged,
            },
            status_code=500,
        )

    return JSONResponse(
        {
            "ok": True,
            "windowId": window_id,
            "videoPath": relative_video_path,
            "stdout": stdout[-4000:],
            "stderr": stderr[-4000:],
            "reasoningSummary": reasoning_summary,
            "latestReasoning": latest_reasoning,
            "latestFlagged": latest_flagged,
            "message": "Video uploaded and description/debate pipeline completed on remote GPU.",
        }
    )


@app.post("/run-video-stream")
async def run_video_stream(video: UploadFile = File(...)) -> StreamingResponse:
    """
    Same pipeline as /run-video but streams Server-Sent Events with structured progress lines.
    Each event: data: {"kind":"progress","payload":{"step","title","detail"}}
    Final success: {"kind":"complete", ...} ; failure: {"kind":"error", ...}
    """

    def error_stream(message: str, code: int = 400) -> StreamingResponse:
        async def gen() -> Any:
            yield f"data: {json.dumps({'kind': 'error', 'detail': message, 'code': code})}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream", status_code=code)

    extension = Path(video.filename or "").suffix.lower()
    if extension not in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
        return error_stream("Unsupported video type.", 400)

    OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_ROOT / "flagged_visuals").mkdir(parents=True, exist_ok=True)
    (OUTPUTS_ROOT / "reasoning").mkdir(parents=True, exist_ok=True)
    INPUTS_ROOT.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time() * 1000)
    safe_name = sanitize_filename(video.filename or f"upload{extension}")
    stored_name = f"upload_{timestamp}_{safe_name}"
    absolute_video_path = INPUTS_ROOT / stored_name
    relative_video_path = f"inputs/{stored_name}"

    file_bytes = await video.read()
    absolute_video_path.write_bytes(file_bytes)

    window_id = f"upload_window_{timestamp}"
    flagged_row: dict[str, Any] = {
        "window_id": window_id,
        "scene_token_hex": f"upload_{timestamp}",
        "log_id": "manual_upload",
        "scenario_tags": ["manual_upload"],
        "window_start_ts": 0,
        "window_end_ts": 0,
        "cluster_label": -1,
        "is_noise": True,
        "cluster_probability": 0.0,
        "outlier_score": 0.9,
        "anomaly_rank": 1,
        "quality": {},
        "metadata": {"upload_source": "remote_gpu_runner_stream"},
    }
    manifest_row = {"window_id": window_id, "grid_path": "", "mp4_path": relative_video_path}

    (OUTPUTS_ROOT / "flagged_windows.jsonl").write_text(
        json.dumps(flagged_row) + "\n",
        encoding="utf-8",
    )
    (OUTPUTS_ROOT / "flagged_visuals" / "manifest.jsonl").write_text(
        json.dumps(manifest_row) + "\n",
        encoding="utf-8",
    )
    suite_path = ensure_default_regression_suite()

    max_new_tokens = os.getenv("WORKSPACE_MAX_NEW_TOKENS", "2400")
    debate_rounds = os.getenv("WORKSPACE_DEBATE_ROUNDS", "2")
    video_fps = os.getenv("WORKSPACE_VIDEO_FPS", "8")

    merged_env = dict(os.environ)
    merged_env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "COSMOS_HF_VIDEO_FPS": video_fps,
        }
    )

    pipeline_args = [
        sys.executable,
        "-u",
        "-m",
        "pipeline.stage_describe_and_debate",
        "--flagged-jsonl",
        "outputs/flagged_windows.jsonl",
        "--visual-manifest-jsonl",
        "outputs/flagged_visuals/manifest.jsonl",
        "--regression-suite-json",
        str(suite_path.relative_to(PROJECT_ROOT)),
        "--output-dir",
        "outputs/reasoning",
        "--hf-max-new-tokens",
        max_new_tokens,
        "--top-k",
        "1",
        "--debate-rounds",
        debate_rounds,
    ]

    async def event_gen() -> Any:
        proc = await asyncio.create_subprocess_exec(
            *pipeline_args,
            cwd=str(PROJECT_ROOT),
            env=merged_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        assert proc.stdout is not None
        log_lines: list[str] = []
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip("\n\r")
            log_lines.append(text)
            if text.startswith(PIPELINE_PROGRESS_PREFIX):
                body = text[len(PIPELINE_PROGRESS_PREFIX) :]
                try:
                    payload = json.loads(body)
                except json.JSONDecodeError:
                    continue
                yield f"data: {json.dumps({'kind': 'progress', 'payload': payload})}\n\n"

        code = await proc.wait()
        combined = "\n".join(log_lines)
        if code != 0:
            yield f"data: {json.dumps({'kind': 'error', 'code': code, 'detail': 'Pipeline run failed.', 'logTail': combined[-12000:]})}\n\n"
            return

        summary_path = OUTPUTS_ROOT / "reasoning" / "summary.json"
        try:
            reasoning_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            reasoning_summary = None
        latest_reasoning, latest_flagged = load_latest_outputs(window_id)
        if latest_reasoning and is_mock_model_source(latest_reasoning.get("modelSource")):
            yield f"data: {json.dumps({'kind': 'error', 'detail': 'Run completed but returned mock output. Real scene description is required.', 'latestReasoning': latest_reasoning, 'latestFlagged': latest_flagged})}\n\n"
            return

        done_payload = {
            "kind": "complete",
            "ok": True,
            "windowId": window_id,
            "videoPath": relative_video_path,
            "message": "Video uploaded and description/debate pipeline completed on remote GPU.",
            "reasoningSummary": reasoning_summary,
            "latestReasoning": latest_reasoning,
            "latestFlagged": latest_flagged,
            "stdout": combined[-4000:],
        }
        yield f"data: {json.dumps(done_payload)}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

