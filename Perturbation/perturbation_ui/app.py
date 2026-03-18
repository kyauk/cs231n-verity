#!/usr/bin/env python3
"""
FastAPI backend for the Perturbation Pipeline Web UI.

Provides endpoints for listing example scenes, uploading BDDL files,
running the perturbation pipeline with SSE progress, serving results,
and downloading generated files as a zip.
"""

import asyncio
import json
import os
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from pipeline.perturbations import (
    PERTURBATION_GENERATORS,
    infer_init_states_path,
    infer_problem_folder,
    _extract_language,
    _bddl_basename,
    _strip_perturbation_suffix,
    read_bddl,
)

# ---------------------------------------------------------------------------
# ImageMagick env vars (needed by wand for rendering)
# ---------------------------------------------------------------------------
if not os.environ.get("MAGICK_HOME"):
    brew_magick = "/opt/homebrew/opt/imagemagick"
    if os.path.isdir(brew_magick):
        os.environ["MAGICK_HOME"] = brew_magick
        os.environ.setdefault(
            "DYLD_LIBRARY_PATH",
            os.path.join(brew_magick, "lib"),
        )

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(title="LIBERO Perturbation Pipeline")

UI_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=UI_DIR / "static"), name="static")
templates = Jinja2Templates(directory=UI_DIR / "templates")

PROJECT_ROOT = UI_DIR.parent
BDDL_ROOT = PROJECT_ROOT / "LIBERO-plus-main" / "libero" / "libero" / "bddl_files"
JOBS_DIR = PROJECT_ROOT / "output" / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

DIMENSION_NAMES = list(PERTURBATION_GENERATORS.keys())

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/examples")
async def list_examples():
    """Return example BDDL scenes from libero_90, grouped by scene type."""
    suite_dir = BDDL_ROOT / "libero_90"
    if not suite_dir.is_dir():
        return {"groups": []}

    scenes = []
    for f in sorted(suite_dir.glob("*.bddl")):
        try:
            text = f.read_text()
            lang = _extract_language(text) or f.stem
        except Exception:
            lang = f.stem
        scenes.append({
            "name": f.stem,
            "language": lang,
            "path": str(f),
        })

    groups = {}
    for s in scenes:
        prefix = s["name"].split("_")[0]
        groups.setdefault(prefix, []).append(s)

    return {
        "groups": [
            {"label": k, "scenes": v}
            for k, v in groups.items()
        ]
    }


@app.post("/api/upload")
async def upload_bddl(file: UploadFile = File(...)):
    """Save an uploaded BDDL file and return its temp path + language."""
    tmp_dir = JOBS_DIR / "uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dest = tmp_dir / f"{uuid.uuid4().hex}_{file.filename}"
    content = await file.read()
    dest.write_bytes(content)
    lang = _extract_language(content.decode("utf-8", errors="replace")) or ""
    return {"path": str(dest), "language": lang, "filename": file.filename}


@app.post("/api/generate")
async def generate(request: Request):
    """
    SSE endpoint: run the perturbation pipeline, streaming progress events.
    Expects JSON body with: input_path, perturbations (dict of dim -> count),
    severity_object_layout, severity_robot_init, noise_type, seed.
    """
    body = await request.json()
    input_path = body["input_path"]
    perturbations = body.get("perturbations", {})
    severity_layout = body.get("severity_object_layout", 3)
    severity_robot = body.get("severity_robot_init", 3)
    noise_type = body.get("noise_type", None)
    seed = body.get("seed", 42)

    async def event_stream():
        job_id = uuid.uuid4().hex[:12]
        output_root = str(JOBS_DIR / job_id)
        os.makedirs(output_root, exist_ok=True)

        yield _sse({"type": "job_start", "job_id": job_id})

        problem_folder = infer_problem_folder(input_path)
        init_states_path = infer_init_states_path(input_path)
        if init_states_path and not os.path.isfile(init_states_path):
            init_states_path = None

        rng = np.random.RandomState(seed)
        bddl_text = read_bddl(input_path)
        source_language = _extract_language(bddl_text) or ""
        source_name = _bddl_basename(input_path)

        all_tasks = []
        total_dims = len(perturbations)

        for idx, (dim, num_variants) in enumerate(perturbations.items()):
            if dim not in PERTURBATION_GENERATORS or num_variants <= 0:
                continue

            yield _sse({
                "type": "dim_start",
                "dimension": dim,
                "index": idx,
                "total": total_dims,
            })

            generator = PERTURBATION_GENERATORS[dim]
            kwargs = {
                "bddl_filepath": input_path,
                "output_root": output_root,
                "problem_folder": problem_folder,
                "init_states_path": init_states_path,
                "num_variants": num_variants,
                "rng": rng,
            }
            if dim == "object_layout":
                kwargs["severity"] = severity_layout
            elif dim == "robot_init":
                kwargs["severity"] = severity_robot
            elif dim == "noise" and noise_type:
                kwargs["noise_type"] = noise_type

            await asyncio.sleep(0)  # yield control so SSE flushes
            tasks = generator(**kwargs)
            all_tasks.extend(tasks)

            yield _sse({
                "type": "dim_done",
                "dimension": dim,
                "count": len(tasks),
                "index": idx,
                "total": total_dims,
            })

        yield _sse({"type": "render_start", "total": len(all_tasks)})

        # Write manifest before rendering so the subprocess can read it
        manifest = {
            "generated_at": datetime.now().isoformat(),
            "source_task": {
                "name": source_name,
                "language": source_language,
                "problem_folder": problem_folder,
            },
            "output_root": output_root,
            "seed": seed,
            "tasks": all_tasks,
        }
        manifest_path = os.path.join(output_root, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        # Render previews in a subprocess to isolate MuJoCo/OpenGL context
        rendered = 0
        render_script = str(PROJECT_ROOT / "pipeline" / "render.py")
        proc = await asyncio.create_subprocess_exec(
            sys.executable, render_script, manifest_path, input_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            if msg.get("type") == "progress":
                rendered = msg.get("rendered", rendered)
                yield _sse({
                    "type": "render_progress",
                    "rendered": rendered,
                    "current": msg["current"],
                    "total": msg["total"],
                    "task_name": msg.get("task_name", ""),
                    "perturbation": msg.get("perturbation", ""),
                })
            elif msg.get("type") == "error":
                yield _sse({
                    "type": "render_error",
                    "message": msg.get("message", "Unknown render error"),
                })
            elif msg.get("type") == "done":
                rendered = msg.get("rendered", rendered)

        await proc.wait()

        if proc.returncode != 0:
            stderr_out = await proc.stderr.read()
            yield _sse({
                "type": "render_error",
                "message": f"Render subprocess failed (exit {proc.returncode}): "
                           + stderr_out.decode(errors="replace").strip()[-500:],
            })

        # Re-read manifest to pick up preview fields set by the renderer
        with open(manifest_path) as f:
            updated = json.load(f)
        all_tasks = updated.get("tasks", all_tasks)

        yield _sse({
            "type": "complete",
            "job_id": job_id,
            "total_tasks": len(all_tasks),
            "rendered": rendered,
            "tasks": all_tasks,
        })

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/results/{job_id}/images/{filename:path}")
async def serve_image(job_id: str, filename: str):
    """Serve a preview image for a given job."""
    job_dir = JOBS_DIR / job_id
    # Search in previews subdirectory
    for p in job_dir.rglob(filename):
        if p.is_file():
            return FileResponse(p, media_type="image/png")
    return {"error": "not found"}


@app.get("/api/download/{job_id}")
async def download_zip(job_id: str):
    """Zip and return all generated files for a job (including previews)."""
    job_dir = JOBS_DIR / job_id
    if not job_dir.is_dir():
        return {"error": "job not found"}

    zip_path = JOBS_DIR / f"{job_id}.zip"
    # Always regenerate so previews rendered after first download are included
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", str(job_dir))

    return FileResponse(
        str(zip_path),
        media_type="application/zip",
        filename=f"perturbations_{job_id}.zip",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
