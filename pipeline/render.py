#!/usr/bin/env python3
"""
Standalone preview renderer for the perturbation pipeline.

Reads a job manifest, renders preview images for each task, and updates
the manifest with preview filenames. Prints JSON progress lines to stdout
so callers can monitor progress.

Usage:
    python -m pipeline.render <manifest_path> <source_bddl_path>
"""

import json
import os
import shutil
import sys
from pathlib import Path

# Ensure ImageMagick/wand can find MagickWand shared library
if not os.environ.get("MAGICK_HOME"):
    brew_magick = "/opt/homebrew/opt/imagemagick"
    if os.path.isdir(brew_magick):
        os.environ["MAGICK_HOME"] = brew_magick
        os.environ.setdefault(
            "DYLD_LIBRARY_PATH",
            os.path.join(brew_magick, "lib"),
        )


def render_previews(manifest_path: str, source_bddl: str):
    """Render preview images for every task in a job manifest."""
    import torch
    from PIL import Image
    from libero.libero.envs import OffScreenRenderEnv

    with open(manifest_path) as f:
        manifest = json.load(f)

    output_root = manifest["output_root"]
    tasks = manifest["tasks"]

    for pf in {t["problem_folder"] for t in tasks}:
        dest = os.path.join(output_root, "bddl_files", pf, Path(source_bddl).name)
        if not os.path.isfile(dest):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(source_bddl, dest)

    rendered = 0
    for i, task in enumerate(tasks):
        pf = task["problem_folder"]
        bddl_path = os.path.join(output_root, "bddl_files", pf, task["bddl_file"])
        init_path = os.path.join(output_root, "init_files", pf, task["init_states_file"])
        preview_dir = os.path.join(output_root, "previews", pf)
        os.makedirs(preview_dir, exist_ok=True)
        preview_path = os.path.join(preview_dir, f"{task['name']}.png")

        if not os.path.isfile(bddl_path):
            _progress(i + 1, len(tasks), task["name"], task["perturbation"], rendered, skip="bddl not found")
            continue

        try:
            env = OffScreenRenderEnv(
                bddl_file_name=bddl_path,
                camera_heights=256,
                camera_widths=256,
            )
            env.seed(0)
            if os.path.isfile(init_path):
                states = torch.load(init_path, weights_only=False)
                obs = env.set_init_state(states[0])
            else:
                obs = env.reset()
            Image.fromarray(obs["agentview_image"][::-1]).save(preview_path)
            task["preview"] = f"{task['name']}.png"
            rendered += 1
            env.close()
        except Exception as e:
            task["preview"] = None
            _progress(i + 1, len(tasks), task["name"], task["perturbation"], rendered, error=str(e))
            continue

        _progress(i + 1, len(tasks), task["name"], task["perturbation"], rendered)

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps({"type": "done", "rendered": rendered, "total": len(tasks)}), flush=True)


def _progress(current, total, task_name, perturbation, rendered, error=None, skip=None):
    msg = {
        "type": "progress",
        "current": current,
        "total": total,
        "task_name": task_name,
        "perturbation": perturbation,
        "rendered": rendered,
    }
    if error:
        msg["error"] = error
    if skip:
        msg["skip"] = skip
    print(json.dumps(msg), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <manifest_path> <source_bddl_path>", file=sys.stderr)
        sys.exit(1)

    manifest_path = sys.argv[1]
    source_bddl = sys.argv[2]

    if not os.path.isfile(manifest_path):
        print(json.dumps({"type": "error", "message": f"Manifest not found: {manifest_path}"}), flush=True)
        sys.exit(1)
    if not os.path.isfile(source_bddl):
        print(json.dumps({"type": "error", "message": f"Source BDDL not found: {source_bddl}"}), flush=True)
        sys.exit(1)

    try:
        render_previews(manifest_path, source_bddl)
    except Exception as e:
        print(json.dumps({"type": "error", "message": str(e)}), flush=True)
        sys.exit(1)
