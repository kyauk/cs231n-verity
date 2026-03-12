#!/usr/bin/env python3
"""
Failure Case Perturbation Pipeline

Given a LIBERO Task (BDDL file + init states), generates perturbed variants
across user-selected perturbation dimensions and writes them as complete
Task bundles in a LIBERO-compatible directory layout.

Usage:
    python -m pipeline.generate \\
        --input path/to/failure_case.bddl \\
        --output-dir ./output/perturbed/ \\
        --perturbations object_layout camera noise \\
        --num-variants-object-layout 5 \\
        --num-variants-camera 5 \\
        --seed 42 \\
        --render
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

from pipeline.perturbations import (
    PERTURBATION_GENERATORS,
    infer_init_states_path,
    infer_problem_folder,
    _extract_language,
    _bddl_basename,
    _strip_perturbation_suffix,
    read_bddl,
)

DIMENSION_NAMES = list(PERTURBATION_GENERATORS.keys())

DEFAULT_NUM_VARIANTS = {
    "object_layout": 5,
    "camera": 5,
    "robot_init": 5,
    "noise": 5,
    "texture": 5,
    "lighting": 5,
    "language": 3,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate perturbed failure-case Task bundles from a source LIBERO task.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to the source BDDL file (the failure case).",
    )
    parser.add_argument(
        "--init-states",
        default=None,
        help=(
            "Path to the base .pruned_init file. "
            "If omitted, inferred from the BDDL path using LIBERO conventions."
        ),
    )
    parser.add_argument(
        "--problem-folder",
        default=None,
        help=(
            "LIBERO suite / problem folder name (e.g. 'libero_mix'). "
            "If omitted, inferred from the BDDL file's parent directory."
        ),
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./output/perturbed",
        help="Root directory for generated Task bundles (default: ./output/perturbed).",
    )
    parser.add_argument(
        "--perturbations", "-p",
        nargs="+",
        choices=DIMENSION_NAMES,
        default=DIMENSION_NAMES,
        help=f"Perturbation dimensions to apply. Choices: {', '.join(DIMENSION_NAMES)}. Default: all.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--render",
        action="store_true",
        help=(
            "Render a preview image for each generated task. "
            "Requires MuJoCo + robosuite + libero to be installed."
        ),
    )

    for dim in DIMENSION_NAMES:
        flag = f"--num-variants-{dim.replace('_', '-')}"
        parser.add_argument(
            flag, type=int, default=DEFAULT_NUM_VARIANTS[dim], metavar="N",
            help=f"Number of variants for {dim} (default: {DEFAULT_NUM_VARIANTS[dim]}).",
        )

    parser.add_argument(
        "--severity-object-layout",
        type=int, default=3, choices=range(1, 6),
        help="Severity 1-5 for object layout perturbation (default: 3).",
    )
    parser.add_argument(
        "--severity-robot-init",
        type=int, default=3, choices=range(1, 6),
        help="Severity 1-5 for robot init perturbation (default: 3).",
    )
    parser.add_argument(
        "--noise-type",
        default=None,
        choices=["motion_blur", "gaussian_blur", "zoom_blur", "fog", "glass_blur"],
        help="Restrict sensor noise to a specific type (default: sample all).",
    )

    return parser


def _apply_sensor_noise(img_array, noise_level):
    """Apply the same sensor noise that ControlEnv applies during step/reset.

    noise_level encoding (matches ControlEnv):
      1-10  motion blur    (severity = noise_level)
      11-20 gaussian blur  (severity = noise_level - 10)
      21-30 zoom blur      (severity = noise_level - 20)
      31-40 fog            (severity = noise_level - 30)
      41-50 glass blur     (severity = noise_level - 40)
    """
    from PIL import Image as _PILImage
    import numpy as np
    from libero.libero.envs.env_wrapper import (
        motion_blur, gaussian_blur, zoom_blur, fog, glass_blur,
    )

    if img_array.dtype != np.uint8:
        img_array = (img_array * 255).astype(np.uint8)
    pil_image = _PILImage.fromarray(img_array)

    if noise_level <= 10:
        return motion_blur(pil_image, severity=noise_level)
    elif noise_level <= 20:
        return gaussian_blur(pil_image, severity=noise_level - 10).astype(np.uint8)
    elif noise_level <= 30:
        return zoom_blur(pil_image, severity=noise_level - 20).astype(np.uint8)
    elif noise_level <= 40:
        return fog(pil_image, severity=noise_level - 30).astype(np.uint8)
    elif noise_level <= 50:
        return glass_blur(pil_image, severity=noise_level - 40).astype(np.uint8)
    return img_array


def _render_tasks(tasks: list, output_root: str, source_bddl: str):
    """Render a preview image for each generated task. Deferred imports."""
    try:
        import torch
        from PIL import Image
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError as e:
        print(f"[error] Cannot render: {e}")
        print("[error] Install LIBERO dependencies: cd LIBERO-plus-main && pip install -e . && pip install -r requirements.txt")
        return

    # ControlEnv strips filename-encoded suffixes (camera/robot_init/noise)
    # and loads the base BDDL from the same directory. Copy the source BDDL
    # into each problem_folder so LIBERO can find it during rendering.
    for pf in {t["problem_folder"] for t in tasks}:
        dest = os.path.join(output_root, "bddl_files", pf, Path(source_bddl).name)
        if not os.path.isfile(dest):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(source_bddl, dest)

    rendered = 0
    for task in tasks:
        problem_folder = task["problem_folder"]
        bddl_path = os.path.join(output_root, "bddl_files", problem_folder, task["bddl_file"])
        init_path = os.path.join(output_root, "init_files", problem_folder, task["init_states_file"])

        preview_dir = os.path.join(output_root, "previews", problem_folder)
        os.makedirs(preview_dir, exist_ok=True)
        preview_path = os.path.join(preview_dir, f"{task['name']}.png")

        if not os.path.isfile(bddl_path):
            print(f"  [skip] BDDL not found: {bddl_path}")
            continue

        try:
            env = OffScreenRenderEnv(
                bddl_file_name=bddl_path,
                camera_heights=256,
                camera_widths=256,
            )
            env.seed(0)

            if os.path.isfile(init_path):
                init_states = torch.load(init_path, weights_only=False)
                obs = env.set_init_state(init_states[0])
            else:
                obs = env.reset()

            img_array = obs["agentview_image"][::-1]

            # set_init_state bypasses ControlEnv.reset(), so sensor noise
            # parsed from the filename is never applied. Apply it here.
            noise_level = getattr(env, "noise", 0)
            if noise_level:
                img_array = _apply_sensor_noise(img_array, noise_level)

            Image.fromarray(img_array).save(preview_path)
            task["preview"] = preview_path
            rendered += 1
            env.close()
        except Exception as e:
            print(f"  [error] Failed to render {task['name']}: {e}")

    print(f"  Rendered {rendered}/{len(tasks)} preview images")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = os.path.abspath(args.input)
    if not os.path.isfile(input_path):
        print(f"[error] Input file not found: {input_path}")
        sys.exit(1)

    problem_folder = args.problem_folder or infer_problem_folder(input_path)
    init_states_path = args.init_states
    if init_states_path:
        init_states_path = os.path.abspath(init_states_path)
    else:
        init_states_path = infer_init_states_path(input_path)

    if init_states_path and not os.path.isfile(init_states_path):
        print(f"[warning] Init states file not found: {init_states_path}")
        init_states_path = None

    output_root = os.path.abspath(args.output_dir)
    os.makedirs(output_root, exist_ok=True)

    import numpy as np
    rng = np.random.RandomState(args.seed)

    bddl_text = read_bddl(input_path)
    source_language = _extract_language(bddl_text) or ""
    source_name = _bddl_basename(input_path)
    source_base_name = _strip_perturbation_suffix(source_name)
    init_ext = Path(init_states_path).suffix if init_states_path else ".pruned_init"

    source_task = {
        "name": source_name,
        "language": source_language,
        "problem": "Libero",
        "problem_folder": problem_folder,
        "bddl_file": f"{source_name}.bddl",
        "init_states_file": f"{source_base_name}{init_ext}" if init_states_path else None,
    }

    print(f"Source BDDL:    {input_path}")
    print(f"Init states:    {init_states_path or '(not found)'}")
    print(f"Problem folder: {problem_folder}")
    print(f"Output:         {output_root}")
    print(f"Seed:           {args.seed}")
    print(f"Dimensions:     {', '.join(args.perturbations)}")
    if args.render:
        print(f"Rendering:      enabled")
    print("-" * 60)

    all_tasks = []
    summary = {}

    for dim in args.perturbations:
        num_key = f"num_variants_{dim}"
        num_variants = getattr(args, num_key, DEFAULT_NUM_VARIANTS[dim])
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
            kwargs["severity"] = args.severity_object_layout
        elif dim == "robot_init":
            kwargs["severity"] = args.severity_robot_init
        elif dim == "noise":
            kwargs["noise_type"] = args.noise_type

        tasks = generator(**kwargs)
        all_tasks.extend(tasks)
        summary[dim] = len(tasks)
        print(f"  {dim:20s}  ->  {len(tasks)} Task bundles generated")

    print("-" * 60)
    print(f"Total: {len(all_tasks)} perturbed Task bundles")
    if init_states_path:
        print(f"       (each with BDDL + init states copy)")

    if args.render:
        print("Rendering previews...")
        _render_tasks(all_tasks, output_root, input_path)

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "source_task": source_task,
        "output_root": output_root,
        "seed": args.seed,
        "summary": summary,
        "tasks": all_tasks,
    }
    manifest_path = os.path.join(output_root, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
