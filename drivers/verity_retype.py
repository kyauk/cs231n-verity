"""Interpretation-layer re-typing via LABEL-LEVEL surgery (a pure recompute).

Canonicalize the immutable evidence ONCE, then surgically remap specific LABELS:
merge cosmetic/type-fragmented vehicle labels, retype mis-axed behaviors,
relocate scene-context to a new axis, drop genuinely-empty meta. The clustering
and every untouched label stay exactly as canonicalized — no text mutation, no
re-clustering, no collateral. Original evidence is never modified.

Returns a corrected per-scene atom map (scene -> set of "axis:name") that the
selection stage consumes. Report mode prints before/after label sets.

    .venv/bin/python -m drivers.verity_retype
"""
from __future__ import annotations

from collections import defaultdict

from pipeline.interfaces.taxonomy import EMPTY_TAXONOMY
from pipeline.modules.curator import CuratorConfig, TaxonomyStore, canonicalize, project

CFG = CuratorConfig(cohesion_threshold=0.36, merge_threshold=0.20, support_threshold=2)
SRC = "outputs/waymo/taxonomy_store_salience"

# (axis, name) -> (new_axis, new_name)  |  None = drop.  Absent = keep as-is.
REMAP = {
    # --- vehicle-presence: collapse colour/type fragments into one (agents; novelty-excluded) ---
    ("agents", "red_car"): ("agents", "vehicle"),
    ("agents", "silver_sedan"): ("agents", "vehicle"),
    ("agents", "silver_minivan"): ("agents", "vehicle"),
    ("agents", "silver_minivan_turning_left_at_an_intersection"): ("agents", "vehicle"),
    ("agents", "oncoming_cars"): ("agents", "vehicle"),
    ("agents", "car_driving_ahead"): ("agents", "vehicle"),
    ("agents", "stopped_cars"): ("agents", "vehicle"),
    ("agents", "cyclist_in_yellow_jacket"): ("agents", "cyclist"),   # strip cosmetic colour
    # --- mis-axed real behaviors -> correct behavior axis (now visible to selection) ---
    ("agents", "yielding_at_crosswalk"): ("interactions", "yielding_to_pedestrians"),  # consolidate
    ("agents", "maintaining_lane_discipline"): ("ego_maneuver", "maintaining_lane"),
    ("agents", "cautious_driving"): ("ego_maneuver", "cautious_driving"),
    ("agents", "accelerating_on_green_light"): ("ego_maneuver", "accelerating_on_green"),
    ("ego_maneuver", "following_a_silver_sedan"): ("interactions", "following_a_vehicle"),  # de-colour
    # --- scene-context -> new 'context' axis (out of novelty) ---
    ("conditions", "suburban_area"): ("context", "suburban_area"),
    ("conditions", "urban_setting"): ("context", "urban_setting"),
    ("conditions", "typical_highway_scenery"): ("context", "highway_scenery"),
    ("conditions", "residential_buildings"): ("context", "residential"),
    ("conditions", "mature_trees_and_shrubs"): ("context", "vegetation"),
    ("agents", "residential_houses"): ("context", "residential"),
    ("agents", "trees"): ("context", "vegetation"),
    ("agents", "power_lines"): ("context", "roadside_objects"),
    ("agents", "driveways"): ("context", "roadside_objects"),
    ("agents", "trash_bins"): ("context", "roadside_objects"),
    # --- hard-drop: genuinely-empty meta / absence (NOT calm states like steady_speed) ---
    ("conditions", "functioning_correctly"): None,
    ("agents", "not_prominently_visible"): None,
    ("interactions", "no_active_interactions"): None,
    ("agents", "various_makes_models_and_colors"): None,
}


def corrected_atoms(ds):
    """Pure: canonicalize, then apply label surgery. Returns
    (scene -> set[(axis,name)] all-axis, label-support dict, before/after summaries)."""
    tax = canonicalize(ds, EMPTY_TAXONOMY, CFG)
    name = {l.label_id: (l.axis, l.name) for l in tax.labels}
    asg = project(ds, tax, CFG).as_dict()

    before = defaultdict(int)
    after = defaultdict(int)
    scene_atoms = defaultdict(set)
    for d in ds:
        lid = asg.get(d.descriptor_id)
        if not lid:
            continue
        orig = name[lid]
        before[orig] += 1
        tgt = REMAP.get(orig, orig)
        if tgt is None:
            continue
        after[tgt] += 1
        scene_atoms[d.scene_id].add(tgt)
    return scene_atoms, after, before


def summarize(counts):
    byax = defaultdict(list)
    for (ax, nm), c in counts.items():
        byax[ax].append((nm, c))
    return byax


def main():
    ds = TaxonomyStore(SRC).load_descriptors()
    scene_atoms, after, before = corrected_atoms(ds)
    b, a = summarize(before), summarize(after)
    print(f"labels: {len(before)} -> {len(after)}  (dropped/merged the difference)\n")
    for ax in sorted(set(b) | set(a)):
        print(f"=== {ax}: {len(b.get(ax,[]))} -> {len(a.get(ax,[]))} labels ===")
        print("  " + ", ".join(f"{n}({c})" for n, c in sorted(a.get(ax, []), key=lambda x: -x[1])))
    # confirm no colour survives in any behavior axis
    BEH = {"interactions", "conditions", "ego_maneuver"}
    colours = {"silver","red","white","black","blue","grey","gray","green","yellow"}
    leaks = [f"{ax}:{n}" for (ax, n) in after if ax in BEH and (set(n.split("_")) & colours)]
    print(f"\ncolour-in-behavior-axis after surgery: {leaks or 'NONE'}")
    print(f"vehicle label support: {after.get(('agents','vehicle'))}")
    print(f"yielding_to_pedestrians: {after.get(('interactions','yielding_to_pedestrians'))} (was 27)")


if __name__ == "__main__":
    main()
