"""
Perturbation generators for all 7 LIBERO-plus dimensions.

Each generator takes a source BDDL file path, base init-states path,
and problem folder, then produces N perturbed Task bundles (BDDL file +
init-states copy + Task metadata dict) in a LIBERO-compatible output layout.

All generators work at the BDDL text level to avoid heavy MuJoCo/robosuite
imports during generation.
"""

import copy
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# BDDL text-level helpers
# ---------------------------------------------------------------------------

def read_bddl(filepath: str) -> str:
    with open(filepath, "r") as f:
        return f.read()


def write_bddl(filepath: str, content: str) -> str:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(content)
    return filepath


def _bddl_basename(filepath: str) -> str:
    return Path(filepath).stem


def _extract_problem_name(bddl_text: str) -> Optional[str]:
    match = re.search(r'\(define\s+\(problem\s+(\S+)\)', bddl_text)
    return match.group(1) if match else None


def _replace_problem_name(bddl_text: str, new_name: str) -> str:
    return re.sub(
        r'(\(define\s+\(problem\s+)\S+(\))',
        rf'\g<1>{new_name}\2',
        bddl_text,
        count=1,
    )


def _extract_language(bddl_text: str) -> Optional[str]:
    match = re.search(r'\(:language\s+(.+?)\)\s*\n', bddl_text)
    if match:
        return match.group(1).strip()
    match = re.search(r'\(:language\s+(.+?)(?=\n\s*\(:)', bddl_text, re.DOTALL)
    if match:
        return match.group(1).strip().rstrip(')')
    return None


def _replace_language(bddl_text: str, new_language: str) -> str:
    return re.sub(
        r'(\(:language\s+).+?\)',
        rf'\g<1>{new_language})',
        bddl_text,
        count=1,
    )


def _find_all_ranges(bddl_text: str) -> List[re.Match]:
    """Find all (:ranges ( (xmin ymin xmax ymax) )) blocks."""
    return list(re.finditer(
        r'\(:ranges\s*\(\s*\n\s*\(([^)]+)\)',
        bddl_text,
    ))


def _perturb_range(range_str: str, dx: float, dy: float) -> str:
    """Shift a range (xmin ymin xmax ymax) by (dx, dy)."""
    vals = [float(v) for v in range_str.split()]
    if len(vals) != 4:
        return range_str
    xmin, ymin, xmax, ymax = vals
    return f"{xmin + dx} {ymin + dy} {xmax + dx} {ymax + dy}"


# ---------------------------------------------------------------------------
# Task-bundle helpers
# ---------------------------------------------------------------------------

def _strip_perturbation_suffix(name: str) -> str:
    """
    Strip known LIBERO-plus perturbation suffixes to recover the base task name.
    Mirrors the logic in benchmark/__init__.py get_task_init_states().
    """
    patterns = [
        r'_view_[\d_]+_initstate_\d+(_noise_\d+)?$',
        r'_language_\d+$',
        r'_table_\d+$',
        r'_tb_\d+$',
        r'_light_\d+$',
        r'_add_\d+$',
        r'_level\d+_sample\d+$',
        r'_moved_level\d+_sample\d+$',
        r'_layout_\d+$',
        r'_texture_\d+$',
    ]
    result = name
    for pat in patterns:
        result = re.sub(pat, '', result)
    return result


def infer_problem_folder(bddl_filepath: str) -> str:
    """Infer problem_folder from the BDDL file's parent directory name."""
    return Path(bddl_filepath).parent.name


def infer_init_states_path(bddl_filepath: str) -> Optional[str]:
    """
    Infer the init-states file path from a BDDL file path by:
    1. Stripping perturbation suffixes from the filename
    2. Navigating from bddl_files/{suite}/ to init_files/{suite}/
    3. Looking for {base_name}.pruned_init, falling back to .init
    """
    bddl_dir = Path(bddl_filepath).parent
    suite_name = bddl_dir.name
    base_name = _strip_perturbation_suffix(_bddl_basename(bddl_filepath))

    bddl_files_root = bddl_dir.parent
    libero_root = bddl_files_root.parent
    init_dir = libero_root / "init_files" / suite_name

    for ext in (".pruned_init", ".init"):
        candidate = init_dir / f"{base_name}{ext}"
        if candidate.exists():
            return str(candidate)

    return None


def _copy_init_states(
    src_path: str,
    output_root: str,
    problem_folder: str,
    variant_name: str,
) -> str:
    """Copy base init-states file to the LIBERO-compatible output layout."""
    dest_dir = os.path.join(output_root, "init_files", problem_folder)
    os.makedirs(dest_dir, exist_ok=True)
    ext = Path(src_path).suffix if Path(src_path).suffix else ".pruned_init"
    dest_path = os.path.join(dest_dir, f"{variant_name}{ext}")
    shutil.copy2(src_path, dest_path)
    return dest_path


def _write_bddl_to_layout(
    content: str,
    output_root: str,
    problem_folder: str,
    variant_name: str,
) -> str:
    """Write BDDL content to the LIBERO-compatible output layout."""
    dest_dir = os.path.join(output_root, "bddl_files", problem_folder)
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, f"{variant_name}.bddl")
    with open(dest_path, "w") as f:
        f.write(content)
    return dest_path


def _make_task_dict(
    variant_name: str,
    language: str,
    problem_folder: str,
    init_states_ext: str,
    perturbation: str,
    **extra,
) -> Dict:
    """Build a Task-compatible dict with perturbation metadata."""
    return {
        "name": variant_name,
        "language": language,
        "problem": "Libero",
        "problem_folder": problem_folder,
        "bddl_file": f"{variant_name}.bddl",
        "init_states_file": f"{variant_name}{init_states_ext}",
        "perturbation": perturbation,
        **extra,
    }


# ---------------------------------------------------------------------------
# Discover registered texture / lighting problem classes from source files
# ---------------------------------------------------------------------------

_PROBLEM_FILE_DIR = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    "LIBERO-plus-main", "libero", "libero", "envs", "problems",
)


def _discover_variant_classes(
    base_problem_name: str,
    variant_keyword: str,
) -> List[str]:
    problem_dir = os.path.normpath(_PROBLEM_FILE_DIR)
    if not os.path.isdir(problem_dir):
        return []

    base_lower = base_problem_name.lower()
    pattern = re.compile(r'class\s+(Libero_\w+)\(')
    matches = []

    for fname in os.listdir(problem_dir):
        if not fname.endswith(".py"):
            continue
        fpath = os.path.join(problem_dir, fname)
        with open(fpath, "r") as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    cls_name = m.group(1)
                    if (
                        cls_name.lower().startswith(base_lower)
                        and variant_keyword in cls_name.lower()
                        and cls_name.lower() != base_lower
                    ):
                        matches.append(cls_name)
    return matches


def _infer_base_problem_name(problem_name: str) -> str:
    lower = problem_name.lower()
    for marker in ("_tabletop_table_", "_tabletop_light_",
                    "_kitchen_table_", "_kitchen_light_",
                    "_liv_table_", "_liv_light_",
                    "_table_", "_bg_", "_light_"):
        idx = lower.find(marker)
        if idx != -1:
            return problem_name[:idx]
    return problem_name


# ---------------------------------------------------------------------------
# 1. Object Layout Perturbation
# ---------------------------------------------------------------------------

LAYOUT_SEVERITY = {1: 0.01, 2: 0.02, 3: 0.03, 4: 0.04, 5: 0.05}


def generate_object_layout(
    bddl_filepath: str,
    output_root: str,
    problem_folder: str,
    init_states_path: Optional[str] = None,
    num_variants: int = 5,
    severity: int = 3,
    rng: Optional[np.random.RandomState] = None,
) -> List[Dict]:
    if rng is None:
        rng = np.random.RandomState()

    bddl_text = read_bddl(bddl_filepath)
    language = _extract_language(bddl_text) or ""
    basename = _bddl_basename(bddl_filepath)
    magnitude = LAYOUT_SEVERITY.get(severity, 0.03)
    init_ext = Path(init_states_path).suffix if init_states_path else ".pruned_init"
    results = []

    for i in range(1, num_variants + 1):
        text = bddl_text
        range_matches = _find_all_ranges(text)
        for match in reversed(range_matches):
            dx = rng.uniform(-magnitude, magnitude)
            dy = rng.uniform(-magnitude, magnitude)
            old_range = match.group(1)
            new_range = _perturb_range(old_range, dx, dy)
            start, end = match.start(1), match.end(1)
            text = text[:start] + new_range + text[end:]

        variant_name = f"{basename}_layout_{i}"
        _write_bddl_to_layout(text, output_root, problem_folder, variant_name)
        if init_states_path:
            _copy_init_states(init_states_path, output_root, problem_folder, variant_name)

        results.append(_make_task_dict(
            variant_name=variant_name,
            language=language,
            problem_folder=problem_folder,
            init_states_ext=init_ext,
            perturbation="object_layout",
            variant=i,
            severity=severity,
            magnitude=magnitude,
        ))
    return results


# ---------------------------------------------------------------------------
# 2. Camera Viewpoint Perturbation
# ---------------------------------------------------------------------------

HORIZON_VALUES = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
VERTICAL_VALUES = [0, 15, 345]
SCALE_VALUES = [80, 90, 100, 110, 125]
ENDPOINT_ROT_VALUES = [0, 15, 30, 345, 330]
ENDPOINT_VERT_VALUES = [0, 15, 345]


def generate_camera_viewpoint(
    bddl_filepath: str,
    output_root: str,
    problem_folder: str,
    init_states_path: Optional[str] = None,
    num_variants: int = 5,
    rng: Optional[np.random.RandomState] = None,
) -> List[Dict]:
    if rng is None:
        rng = np.random.RandomState()

    bddl_text = read_bddl(bddl_filepath)
    language = _extract_language(bddl_text) or ""
    basename = _bddl_basename(bddl_filepath)
    init_ext = Path(init_states_path).suffix if init_states_path else ".pruned_init"
    results = []
    seen = set()

    attempts = 0
    while len(results) < num_variants and attempts < num_variants * 10:
        attempts += 1
        h = int(rng.choice(HORIZON_VALUES))
        v = int(rng.choice(VERTICAL_VALUES))
        s = int(rng.choice(SCALE_VALUES))
        er = int(rng.choice(ENDPOINT_ROT_VALUES))
        ev = int(rng.choice(ENDPOINT_VERT_VALUES))

        if h == 0 and v == 0 and s == 100 and er == 0 and ev == 0:
            continue
        key = (h, v, s, er, ev)
        if key in seen:
            continue
        seen.add(key)

        variant_name = f"{basename}_view_{h}_{v}_{s}_{er}_{ev}_initstate_0"
        _write_bddl_to_layout(bddl_text, output_root, problem_folder, variant_name)
        if init_states_path:
            _copy_init_states(init_states_path, output_root, problem_folder, variant_name)

        results.append(_make_task_dict(
            variant_name=variant_name,
            language=language,
            problem_folder=problem_folder,
            init_states_ext=init_ext,
            perturbation="camera_viewpoint",
            variant=len(results) + 1,
            horizon_view=h,
            vertical_view=v,
            scale_factor=s / 100.0,
            endpoint_rot=er,
            endpoint_vertical=ev,
        ))
    return results


# ---------------------------------------------------------------------------
# 3. Robot Initial State Perturbation
# ---------------------------------------------------------------------------

ROBOT_INIT_TIERS = {
    1: (1, 100), 2: (101, 200), 3: (201, 300),
    4: (301, 400), 5: (401, 500),
}


def generate_robot_init(
    bddl_filepath: str,
    output_root: str,
    problem_folder: str,
    init_states_path: Optional[str] = None,
    num_variants: int = 5,
    severity: int = 3,
    rng: Optional[np.random.RandomState] = None,
) -> List[Dict]:
    if rng is None:
        rng = np.random.RandomState()

    bddl_text = read_bddl(bddl_filepath)
    language = _extract_language(bddl_text) or ""
    basename = _bddl_basename(bddl_filepath)
    init_ext = Path(init_states_path).suffix if init_states_path else ".pruned_init"
    lo, hi = ROBOT_INIT_TIERS.get(severity, (201, 300))
    results = []

    ids = rng.choice(range(lo, hi + 1), size=min(num_variants, hi - lo + 1), replace=False)
    for idx, init_id in enumerate(ids, 1):
        variant_name = f"{basename}_view_0_0_100_0_0_initstate_{init_id}"
        _write_bddl_to_layout(bddl_text, output_root, problem_folder, variant_name)
        if init_states_path:
            _copy_init_states(init_states_path, output_root, problem_folder, variant_name)

        results.append(_make_task_dict(
            variant_name=variant_name,
            language=language,
            problem_folder=problem_folder,
            init_states_ext=init_ext,
            perturbation="robot_init",
            variant=idx,
            init_state_id=int(init_id),
            severity=severity,
        ))
    return results


# ---------------------------------------------------------------------------
# 4. Sensor Noise Perturbation
# ---------------------------------------------------------------------------

NOISE_TYPES = {
    "motion_blur": (1, 10),
    "gaussian_blur": (11, 20),
    "zoom_blur": (21, 30),
    "fog": (31, 40),
    "glass_blur": (41, 50),
}


def generate_sensor_noise(
    bddl_filepath: str,
    output_root: str,
    problem_folder: str,
    init_states_path: Optional[str] = None,
    num_variants: int = 5,
    noise_type: Optional[str] = None,
    rng: Optional[np.random.RandomState] = None,
) -> List[Dict]:
    if rng is None:
        rng = np.random.RandomState()

    bddl_text = read_bddl(bddl_filepath)
    language = _extract_language(bddl_text) or ""
    basename = _bddl_basename(bddl_filepath)
    init_ext = Path(init_states_path).suffix if init_states_path else ".pruned_init"
    results = []

    if noise_type and noise_type in NOISE_TYPES:
        lo, hi = NOISE_TYPES[noise_type]
        pool = list(range(lo, hi + 1))
    else:
        pool = list(range(1, 51))

    ids = rng.choice(pool, size=min(num_variants, len(pool)), replace=False)
    for idx, noise_id in enumerate(sorted(ids), 1):
        ntype = "unknown"
        for name, (lo, hi) in NOISE_TYPES.items():
            if lo <= noise_id <= hi:
                ntype = name
                break

        variant_name = f"{basename}_view_0_0_100_0_0_initstate_0_noise_{noise_id}"
        _write_bddl_to_layout(bddl_text, output_root, problem_folder, variant_name)
        if init_states_path:
            _copy_init_states(init_states_path, output_root, problem_folder, variant_name)

        results.append(_make_task_dict(
            variant_name=variant_name,
            language=language,
            problem_folder=problem_folder,
            init_states_ext=init_ext,
            perturbation="sensor_noise",
            variant=idx,
            noise_id=int(noise_id),
            noise_type=ntype,
        ))
    return results


# ---------------------------------------------------------------------------
# 5. Background Texture Perturbation
# ---------------------------------------------------------------------------

def generate_background_texture(
    bddl_filepath: str,
    output_root: str,
    problem_folder: str,
    init_states_path: Optional[str] = None,
    num_variants: int = 5,
    rng: Optional[np.random.RandomState] = None,
) -> List[Dict]:
    if rng is None:
        rng = np.random.RandomState()

    bddl_text = read_bddl(bddl_filepath)
    language = _extract_language(bddl_text) or ""
    basename = _bddl_basename(bddl_filepath)
    init_ext = Path(init_states_path).suffix if init_states_path else ".pruned_init"
    problem_name = _extract_problem_name(bddl_text)
    if not problem_name:
        print("[warning] Could not extract problem name; skipping texture perturbation")
        return []

    base_name = _infer_base_problem_name(problem_name)
    variants = _discover_variant_classes(base_name, "_table_")
    if not variants:
        variants = _discover_variant_classes(base_name, "_bg_")
    if not variants:
        print(f"[warning] No texture variants found for base '{base_name}'")
        return []

    chosen = rng.choice(variants, size=min(num_variants, len(variants)), replace=False)
    results = []
    for idx, variant_cls in enumerate(chosen, 1):
        new_text = _replace_problem_name(bddl_text, variant_cls)
        variant_name = f"{basename}_texture_{idx}"
        _write_bddl_to_layout(new_text, output_root, problem_folder, variant_name)
        if init_states_path:
            _copy_init_states(init_states_path, output_root, problem_folder, variant_name)

        results.append(_make_task_dict(
            variant_name=variant_name,
            language=language,
            problem_folder=problem_folder,
            init_states_ext=init_ext,
            perturbation="background_texture",
            variant=idx,
            problem_class=variant_cls,
        ))
    return results


# ---------------------------------------------------------------------------
# 6. Lighting Perturbation
# ---------------------------------------------------------------------------

def generate_lighting(
    bddl_filepath: str,
    output_root: str,
    problem_folder: str,
    init_states_path: Optional[str] = None,
    num_variants: int = 5,
    rng: Optional[np.random.RandomState] = None,
) -> List[Dict]:
    if rng is None:
        rng = np.random.RandomState()

    bddl_text = read_bddl(bddl_filepath)
    language = _extract_language(bddl_text) or ""
    basename = _bddl_basename(bddl_filepath)
    init_ext = Path(init_states_path).suffix if init_states_path else ".pruned_init"
    problem_name = _extract_problem_name(bddl_text)
    if not problem_name:
        print("[warning] Could not extract problem name; skipping lighting perturbation")
        return []

    base_name = _infer_base_problem_name(problem_name)
    variants = _discover_variant_classes(base_name, "_light_sync_modified_")
    if not variants:
        print(f"[warning] No lighting variants found for base '{base_name}'")
        return []

    chosen = rng.choice(variants, size=min(num_variants, len(variants)), replace=False)
    results = []
    for idx, variant_cls in enumerate(chosen, 1):
        new_text = _replace_problem_name(bddl_text, variant_cls)
        variant_name = f"{basename}_light_{idx}"
        _write_bddl_to_layout(new_text, output_root, problem_folder, variant_name)
        if init_states_path:
            _copy_init_states(init_states_path, output_root, problem_folder, variant_name)

        results.append(_make_task_dict(
            variant_name=variant_name,
            language=language,
            problem_folder=problem_folder,
            init_states_ext=init_ext,
            perturbation="lighting",
            variant=idx,
            problem_class=variant_cls,
        ))
    return results


# ---------------------------------------------------------------------------
# 7. Language Instruction Perturbation
# ---------------------------------------------------------------------------

SYNONYM_MAP = {
    "pick up": ["grab", "take", "lift", "get"],
    "place": ["put", "set", "lay", "position"],
    "put": ["place", "set", "move"],
    "move": ["slide", "push", "shift", "transfer"],
    "open": ["pull open", "unlatch"],
    "close": ["shut", "push closed"],
    "on top of": ["on", "atop", "onto"],
    "on the": ["onto the", "on a"],
    "inside": ["into", "in"],
    "in the": ["inside the", "into the"],
    "and": ["then", "and then"],
    "the top": ["the upper", "the top"],
    "the bottom": ["the lower", "the bottom"],
    "left": ["left-hand", "left side"],
    "right": ["right-hand", "right side"],
}


def _paraphrase(language: str, rng: np.random.RandomState) -> str:
    result = language.lower()
    keys = list(SYNONYM_MAP.keys())
    rng.shuffle(keys)
    replacements_made = 0
    for phrase in keys:
        if phrase in result and replacements_made < 2:
            replacement = rng.choice(SYNONYM_MAP[phrase])
            result = result.replace(phrase, replacement, 1)
            replacements_made += 1
    return result.capitalize() if language[0].isupper() else result


def generate_language(
    bddl_filepath: str,
    output_root: str,
    problem_folder: str,
    init_states_path: Optional[str] = None,
    num_variants: int = 3,
    rng: Optional[np.random.RandomState] = None,
) -> List[Dict]:
    if rng is None:
        rng = np.random.RandomState()

    bddl_text = read_bddl(bddl_filepath)
    original_lang = _extract_language(bddl_text) or ""
    if not original_lang:
        print("[warning] Could not extract language instruction; skipping")
        return []

    basename = _bddl_basename(bddl_filepath)
    init_ext = Path(init_states_path).suffix if init_states_path else ".pruned_init"
    results = []
    seen = {original_lang.lower()}

    attempts = 0
    while len(results) < num_variants and attempts < num_variants * 20:
        attempts += 1
        new_lang = _paraphrase(original_lang, rng)
        if new_lang.lower() in seen:
            continue
        seen.add(new_lang.lower())

        new_text = _replace_language(bddl_text, new_lang)
        variant_name = f"{basename}_language_{len(results) + 1}"
        _write_bddl_to_layout(new_text, output_root, problem_folder, variant_name)
        if init_states_path:
            _copy_init_states(init_states_path, output_root, problem_folder, variant_name)

        results.append(_make_task_dict(
            variant_name=variant_name,
            language=new_lang,
            problem_folder=problem_folder,
            init_states_ext=init_ext,
            perturbation="language",
            variant=len(results) + 1,
            original_language=original_lang,
        ))
    return results


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PERTURBATION_GENERATORS = {
    "object_layout": generate_object_layout,
    "camera": generate_camera_viewpoint,
    "robot_init": generate_robot_init,
    "noise": generate_sensor_noise,
    "texture": generate_background_texture,
    "lighting": generate_lighting,
    "language": generate_language,
}
