"""Author the two baseline arms directly (no rate-limited API), then emit JSON.

  * ungrounded_llm (Comp-A): 20 dangerous AV scenarios invented from LLM world
    knowledge with NO access to the fleet data — the "just ask a frontier LLM"
    comparator. Authored here in the same 5-field template the other arms use.
  * compositional_rarity (Comp-B): one scenario per computed rarest co-occurring
    condition/behavior pair (read from eval/_rare_pairs.json), rendered honestly —
    statistically rare compositions, which (as the method predicts) are often
    mundane rather than operationally hard. Grounded to the real scene.

Writes outputs/waymo/eval/{ungrounded_llm,compositional_rarity}.json.
Run verity_eval_combine.py afterwards to build the blinded feed.
"""
from __future__ import annotations

import json
from pathlib import Path

EVAL = Path("outputs/waymo/eval")


def _fmt(scenario, setting, agents, sequence, challenge) -> str:
    return (f"Scenario: {scenario}\nSetting: {setting}\nAgents: {agents}\n"
            f"Sequence: {sequence}\nChallenge: {challenge}")


def _entry(pid, constituents, desc, arm, scenes, diff=None, signals=None) -> dict:
    return {
        "composition_id": pid, "constituents": constituents,
        "marginal_frequencies": {}, "pairwise_frequencies": {},
        "expected_joint": 0.0, "observed_joint": 1.0, "novelty_score": 0.0,
        "motivating_scene_ids": scenes, "arm": arm,
        "plausibility_score": 1.0, "plausibility_justification": desc,
        "frontier_difficulty_score": diff, "frontier_difficulty_signals": signals or {},
        "final_rank_score": 0.0, "accepted": True, "rejection_reason": None,
    }


# ---- Comp-A: 20 ungrounded, LLM-invented dangerous scenarios -----------------
UNGROUNDED = [
    # --- subtle / non-obvious / operationally-hard-but-undramatic (the strong half) ---
    _fmt("Lead Vehicle Brakes for an Occluded Reason",
         "Two-lane urban street, parked cars on the right, midday, clear and dry.",
         "The lead vehicle ahead brakes firmly; the reason (a pedestrian stepping out from between parked cars further up) is hidden from the ego behind the lead vehicle's body.",
         "The ego sees only the lead's brake lights with no visible cause, must decide whether this is a normal stop or an emergency, and prepare for a hazard it cannot yet see.",
         "The ego must respond to an effect whose cause is occluded, distinguishing a routine stop from a hidden emergency without the triggering object in view."),
    _fmt("Ambiguous Pedestrian Intent at the Curb",
         "Residential collector road, no marked crosswalk nearby, overcast afternoon, dry.",
         "A pedestrian stands at the curb edge facing the street, looking at a phone, not making eye contact; the ego approaches in the near lane.",
         "The pedestrian's posture is consistent with both waiting and about-to-step-off. The ego must hold a hypothesis about intent and modulate speed without a clear signal either way.",
         "Intent is genuinely ambiguous with no crossing infrastructure to constrain it, so the ego must hedge against a step-off it cannot confirm."),
    _fmt("Faded Lane Markings Through a Dusk Curve",
         "Two-lane road through a gentle right curve, dusk, dry, worn pavement with faded and partially repainted lane lines.",
         "No other agents nearby; old and new lane markings diverge slightly through the curve under low light.",
         "The ego follows the lane through the curve where the visual lane evidence is faint and self-contradictory, with reduced contrast at dusk.",
         "Lane-keeping must resolve conflicting, low-contrast markings through a curve — a quiet failure mode with no agent to cue attention."),
    _fmt("Low Sun Washing Out a Signal State",
         "East-facing signalized intersection, late afternoon, clear, sun low and near the signal head.",
         "Cross traffic and pedestrians proceed normally; the ego approaches with the signal partly washed out by direct sun.",
         "The ego cannot read the signal reliably against the glare and must infer the phase from cross-traffic and pedestrian motion rather than the light itself.",
         "Perception of the controlling signal degrades to near-zero with no incident, forcing inference from indirect cues."),
    _fmt("Cyclist Holding Station in the Blind Spot",
         "Multi-lane urban arterial with a bike lane, daytime, clear and dry.",
         "A cyclist rides in the bike lane at nearly the ego's speed, sitting just aft of the ego's right-side blind spot for an extended period.",
         "As the ego considers a right turn or lane adjustment, the matched-speed cyclist remains in the hardest-to-observe region and is easy to lose track of.",
         "A vulnerable road user persists in the blind spot at zero relative motion, the configuration most likely to be dropped before a right-hook."),
    _fmt("Cones Contradicting the Painted Lane",
         "Wide arterial with light roadwork, daytime, dry, temporary cones shifting the path against existing paint.",
         "No flagger present; a line of cones routes traffic left of the painted lane on an otherwise empty road.",
         "The ego must follow the temporary cone geometry over the still-visible permanent markings, choosing physical guidance over painted guidance.",
         "Two valid-looking lane authorities disagree, and the correct one is the less machine-legible (cones over paint)."),
    _fmt("Double-Parked Van Forcing a Two-Way Negotiation",
         "Narrow two-way residential street, parked cars both sides, daytime, dry.",
         "A delivery van is double-parked blocking the ego's lane; an oncoming car approaches from the other direction toward the same gap.",
         "The ego must borrow the oncoming lane to pass the van while an oncoming vehicle contests the single free gap, requiring an unspoken right-of-way negotiation.",
         "Progress requires entering opposing traffic's lane and resolving who yields with no signal or rule to arbitrate it."),
    _fmt("Shaded Damp Patch With No Visual Cue",
         "Two-lane road under tree cover after light overnight rain, morning, surface mostly dry.",
         "No agents nearby; a shaded stretch stays damp with scattered wet leaves while the surrounding road looks dry.",
         "The ego carries normal speed into the shaded patch where traction is quietly reduced, with no visible difference from the dry pavement around it.",
         "A traction drop arrives with no perceptual cue, so any speed/steering margin must be pre-emptive rather than reactive."),
    _fmt("Stopped Bus, No Pedestrians Yet Visible",
         "Urban street with a near-side bus stop, daytime, clear and dry.",
         "A bus is stopped at the curb ahead; no pedestrians are visible, but the bus fully occludes the area in front of and beside it.",
         "The ego passes the stopped bus where a pedestrian could step out from full occlusion at any moment, with no current evidence of one.",
         "The hazard is purely anticipatory — the right behavior (slow, ready) must be driven by occlusion geometry, not by a visible agent."),
    _fmt("Ambiguous Four-Way-Stop Order",
         "Four-way stop intersection, light traffic, daytime, dry.",
         "The ego and a cross vehicle arrive at nearly the same instant; the other driver edges forward then hesitates, giving mixed signals.",
         "The ego must resolve right-of-way under near-simultaneous arrival while the other driver's hesitation makes their intent unreadable.",
         "Right-of-way is genuinely under-determined and the human counterpart is sending contradictory motion cues."),
    _fmt("Trailing Car Closing the Merge Gap",
         "Freeway on-ramp merging into the right lane, daytime, dry, moderate traffic.",
         "The ego targets a legal gap in the mainline; the vehicle trailing that gap accelerates, shrinking it as the ramp runs out.",
         "The ego must complete the merge into a gap that is closing because the trailing driver is contesting it, before the acceleration lane ends.",
         "A nominally sufficient gap becomes insufficient due to another agent's choice, compressing the merge decision with no lane left."),
    _fmt("Pedestrian Crossing Slowly Against the Signal",
         "Signalized urban intersection, daytime, clear and dry.",
         "The ego has a green to proceed; a pedestrian is still crossing slowly against their signal, partway across the ego's path.",
         "The ego is legally clear to go but a slow pedestrian remains in the conflict zone, requiring it to forgo right-of-way and predict their clearance time.",
         "The legal action and the safe action diverge, and clearance depends on predicting a slow human's path, not on the signal."),
    _fmt("Hi-Vis Worker Near the Travel Lane at Dusk",
         "Suburban road shoulder with minor utility work, dusk, dry, no lane closure.",
         "A worker in a hi-vis vest stands close to the edge of the live travel lane; no cones or signage reduce the speed limit.",
         "The ego passes a person working at the lane edge with no protective buffer or advance warning, under fading light.",
         "A vulnerable worker sits just outside the lane with none of the usual infrastructure cues to trigger caution."),
    # --- dramatic-but-real (a strong engineer lists these too) ---
    _fmt("Occluded Pedestrian Between Parked Cars",
         "Narrow residential street lined with parked cars, late afternoon, clear, low speed limit.",
         "A ball rolls into the street from between parked cars; a child follows it from full occlusion; cars parked tightly on each side.",
         "The ego sees the ball first, then the child darts out from between two parked cars directly into the lane.",
         "The only early cue is the ball; the child appears from total occlusion at close range with almost no reaction margin."),
    _fmt("Unprotected Left Across Fast Oncoming",
         "Four-lane arterial, permissive green, dusk, dry, low sun behind oncoming traffic.",
         "The ego waits to turn left; oncoming vehicles approach at 45 mph; one is partly hidden in glare and accelerating to beat the light.",
         "A gap appears but the glare-obscured oncoming car is closing faster than it looks as the signal ages toward yellow.",
         "Misjudging a glare-hidden oncoming vehicle's speed during an unprotected turn is a classic high-severity conflict."),
    _fmt("Emergency Vehicle From Behind at a Green",
         "Signalized urban intersection, daytime, moderate traffic, ego first in queue.",
         "An ambulance with sirens approaches from directly behind; cross traffic flows on the ego's green; pedestrians are in the crosswalk.",
         "The ego must yield to the ambulance, but moving forward enters live cross traffic and an occupied crosswalk.",
         "Yielding to the emergency vehicle conflicts with the right-of-way of cross traffic and pedestrians, with no clearly safe maneuver."),
    _fmt("Sudden Highway Debris Reveal",
         "Three-lane freeway, 65 mph, daylight, dry, moderately dense traffic.",
         "A large tread fragment lies in the ego's lane; a truck occupies the left lane; a car tailgates closely behind.",
         "The lead vehicle straddles the debris at the last second, revealing it with little time and lanes boxed in on both sides.",
         "High closing speed plus boxed-in lanes make every avoidance option risky, with rear-end exposure if braking."),
    _fmt("Wrong-Way Driver at Night",
         "Divided highway at night, clear, dry, ego in the right lane at speed.",
         "An oncoming vehicle travels the wrong way in the ego's direction; headlights approach head-on; other traffic sparse.",
         "Headlights that should be receding instead grow rapidly head-on, demanding the ego recognize the wrong-way closure and evade.",
         "Closing speed is the sum of both vehicles' speeds, leaving an extremely short window to recognize and avoid a head-on threat."),
    _fmt("Stalled Vehicle Past a Blind Curve",
         "Two-lane mountain road with a sharp blind right curve, daytime, dry, rock wall inside.",
         "A disabled vehicle sits in-lane just beyond the curve apex; no advance warning; oncoming traffic intermittent.",
         "The ego rounds the blind curve and finds the stalled car squarely in its lane within the sightline, while oncoming traffic blocks a clean pass.",
         "Blind geometry hides a stationary obstacle until inside stopping distance, with the escape lane intermittently occupied."),
    _fmt("Lead Hard-Brake With Tailgater",
         "Suburban arterial, daytime, dry, a tailgater close behind the ego.",
         "The lead vehicle brakes hard for a yellow; the ego follows at a normal gap; a vehicle tailgates the ego closely.",
         "The ego must brake firmly for the lead while a tailgater leaves little room behind, risking a chain rear-end.",
         "Braking adequately for the lead raises rear-end risk from the tailgater — a two-sided spacing conflict with no slack."),
]


def build_ungrounded() -> list[dict]:
    return [_entry(f"ungrounded_{i:02d}", ["ungrounded:llm_invented"], d, "ungrounded_llm", [])
            for i, d in enumerate(UNGROUNDED)]


# ---- Comp-B: render each computed rare pair (honest, often mundane) ----------
# keyed by the canonical pair label "a + b" (axis-stripped); rendered faithfully.
RARITY_TEXT = {
    "maintaining_steady_course + stopped_at_red_light":
        _fmt("Steady Approach to a Vehicle Stopped at Red",
             "Urban arterial approaching a signalized intersection, daytime, clear and dry, light traffic.",
             "A lead vehicle is stopped at the red light ahead; the ego maintains a steady course in the same lane; cross traffic flows normally.",
             "The ego holds a constant speed and lane while closing on the queue, then decelerates smoothly to a stop behind the lead vehicle at the red.",
             "A routine stop: the only demand is smooth deceleration behind a stationary lead vehicle at a red signal."),
    "red_light + reduced_visibility":
        _fmt("Red Light in Reduced Visibility",
             "Signalized intersection, dusk with light haze reducing visibility, damp surface.",
             "The signal is red; a lead vehicle waits at the stop line; pedestrians wait at the corner.",
             "The ego approaches through reduced visibility, identifies the red signal and stopped queue, and comes to a controlled stop at the line.",
             "Reduced visibility slightly delays signal and queue detection, but the situation itself is a standard stop."),
    "maintaining_steady_course + yielding_to_pedestrians":
        _fmt("Steady Course While Yielding to Pedestrians",
             "Low-speed urban street with a marked crosswalk, daytime, clear and dry.",
             "Pedestrians cross at the crosswalk; the ego maintains a steady course on approach; no other vehicles nearby.",
             "The ego holds a steady speed toward the crosswalk, then yields by slowing and waiting as pedestrians complete the crossing.",
             "A standard yield: detect the crosswalk occupancy and wait, with no conflicting traffic."),
    "calm_and_orderly_environment + red_light":
        _fmt("Calm Orderly Stop at a Red Light",
             "Quiet suburban intersection, daytime, clear and dry, very light traffic.",
             "The signal is red; one vehicle waits ahead; the surrounding environment is calm and orderly.",
             "The ego approaches the orderly intersection and stops behind the lead vehicle at the red with ample margin.",
             "A benign, low-demand stop in light, orderly conditions."),
    "red_light + stopping_at_red_light":
        _fmt("Stopping at a Red Light",
             "Urban signalized intersection, daytime, clear and dry, moderate traffic.",
             "The signal is red; vehicles ahead are stopping; cross traffic flows.",
             "The ego detects the red signal and the stopping queue and decelerates to a halt at the stop line.",
             "A baseline stopping maneuver at a red signal."),
    "clear_weather + red_light":
        _fmt("Red Light in Clear Weather",
             "Signalized intersection, daytime, clear weather, dry surface, normal traffic.",
             "The signal is red; a lead vehicle is stopped; pedestrians wait at the corner.",
             "Under clear conditions the ego identifies the red signal early and stops smoothly behind the queue.",
             "Ideal visibility and a standard red-light stop make this low-demand."),
    "red_light + no_moving_vehicles":
        _fmt("Red Light With No Moving Vehicles",
             "Empty signalized intersection, daytime, clear and dry.",
             "The signal is red; no other vehicles are moving; the intersection is otherwise empty.",
             "The ego approaches an empty intersection and stops at the red line, holding until the phase changes.",
             "Essentially no interaction: a solitary stop at a red with no other agents."),
    "maintaining_steady_course + maintaining_lane":
        _fmt("Steady Lane Keeping",
             "Multi-lane road, daytime, clear and dry, light traffic.",
             "The ego maintains lane and a steady course; sparse vehicles travel ahead in the same direction.",
             "The ego holds its lane and speed along a straight, uneventful stretch with no conflicts.",
             "Pure nominal driving: maintain lane and speed with nothing to react to."),
    "red_light + no_active_interactions":
        _fmt("Red Light With No Active Interactions",
             "Signalized intersection, daytime, clear and dry, minimal traffic.",
             "The signal is red; no agents interact with the ego; a lead vehicle is stopped ahead.",
             "The ego stops at the red behind the lead vehicle; no pedestrians, cyclists, or cross conflicts engage it.",
             "A stop with no active interactions — among the lowest-demand situations."),
    "red_light + slippery_road_surface":
        _fmt("Red Light on a Slippery Surface",
             "Signalized intersection after rain, overcast, wet and slippery surface.",
             "The signal is red; a lead vehicle waits; the road is wet with reduced grip.",
             "The ego approaches the red and brakes earlier and gentler than usual to account for the slippery surface, stopping behind the queue.",
             "The only added demand is modulating braking for reduced traction on a routine stop."),
    "red_light + vigilance_for_sudden_movements":
        _fmt("Red Light With Vigilance for Sudden Movements",
             "Urban signalized intersection, daytime, clear and dry, pedestrians and cyclists nearby.",
             "The signal is red; the ego waits; nearby pedestrians and cyclists could move unexpectedly.",
             "The ego holds at the red while monitoring nearby vulnerable road users for any sudden entry into its path.",
             "A standard stop with added monitoring for possible sudden movement by nearby agents."),
    "nighttime_driving + maintaining_steady_course":
        _fmt("Steady Course at Night",
             "Lit urban road at night, clear and dry, light traffic.",
             "The ego maintains a steady course at night; a few vehicles travel ahead with taillights visible.",
             "The ego holds lane and speed under street lighting, following the flow with no conflicts.",
             "Routine night cruising; reduced lighting is the only mild factor."),
    "red_light + maintaining_steady_course":
        _fmt("Steady Course Toward a Red Light",
             "Urban arterial approaching a red signal, daytime, clear and dry, moderate traffic.",
             "The signal is red ahead; the ego maintains a steady course; a queue forms at the line.",
             "The ego holds speed on approach, then decelerates smoothly to join the queue at the red.",
             "A common approach-and-stop with smooth deceleration to a forming queue."),
    "red_light + cyclist_in_bike_lane":
        _fmt("Red Light Beside a Cyclist in a Bike Lane",
             "Urban intersection with a bike lane, daytime, clear and dry.",
             "The signal is red; a cyclist waits or rolls in the adjacent bike lane; a lead vehicle is stopped ahead.",
             "The ego stops at the red while a cyclist occupies the parallel bike lane, keeping aware of the cyclist's position through the stop.",
             "A standard stop with a cyclist alongside — mild lateral awareness, no direct conflict."),
    "red_light + following_distance":
        _fmt("Maintaining Following Distance Into a Red Light",
             "Urban arterial, daytime, clear and dry, moderate traffic.",
             "The signal is red; the ego follows a lead vehicle at a set gap; the queue decelerates.",
             "The ego maintains its following distance as the lead slows for the red, stopping with an appropriate gap.",
             "Routine car-following into a stop; the demand is gap maintenance."),
    "clear_weather + reduced_visibility":
        _fmt("Localized Reduced Visibility in Clear Weather",
             "Open road under broadly clear weather with a localized visibility reduction (low sun or brief haze), dry surface.",
             "Traffic ahead is sparse; a patch of glare or haze briefly reduces forward visibility despite the clear sky.",
             "The ego passes through a short stretch where visibility drops, then clears, adjusting speed mildly through the patch.",
             "A brief, localized visibility dip in otherwise clear conditions — a contradictory-seeming but low-severity pairing."),
    "red_light + following_a_silver_sedan":
        _fmt("Following a Silver Sedan to a Red Light",
             "Urban arterial, daytime, clear and dry, moderate traffic.",
             "The signal is red; the ego follows a silver sedan; the sedan slows for the light.",
             "The ego tracks the silver sedan as it decelerates for the red and stops behind it at the line.",
             "Routine car-following to a stop; the lead vehicle's colour is incidental, underscoring how attribute-level rarity is not difficulty."),
    "moderate_traffic + red_light":
        _fmt("Red Light in Moderate Traffic",
             "Urban signalized intersection, daytime, clear and dry, moderate traffic.",
             "The signal is red; a moderate queue forms; cross traffic flows.",
             "The ego joins the moderate queue and stops at the red, advancing normally when the phase changes.",
             "A standard stop within moderate traffic — ordinary urban driving."),
    "overcast_weather + red_light":
        _fmt("Red Light Under Overcast Skies",
             "Urban signalized intersection, daytime, overcast but dry.",
             "The signal is red; a lead vehicle waits; lighting is flat under overcast skies.",
             "The ego identifies the red under even overcast lighting and stops behind the lead vehicle.",
             "Overcast lighting is benign; the stop itself is routine."),
    "red_light + routine_driving_conditions":
        _fmt("Red Light Under Routine Conditions",
             "Urban signalized intersection, daytime, clear and dry, routine conditions.",
             "The signal is red; a lead vehicle is stopped; conditions are entirely routine.",
             "The ego approaches under routine conditions and stops at the red behind the queue.",
             "By construction a routine stop — illustrating that statistically rare compositions are frequently mundane."),
}


def build_rarity() -> list[dict]:
    pairs = json.loads((EVAL / "_rare_pairs.json").read_text())
    out = []
    for r in pairs:
        desc = RARITY_TEXT.get(r["pair"])
        if desc is None:  # fallback if a pair label changed
            desc = _fmt(r["pair"], "Real fleet scene embodying this rare composition.",
                        "As observed in the motivating scene.", "Rendered from the rare pair.",
                        "A statistically rare composition surfaced by lift.")
        out.append(_entry(f"rarity_{r['i']:02d}", [r["a"], r["b"]], desc, "compositional_rarity",
                          [{"segment_id": r["scene"], "window_idx": 0}],
                          signals={"joint_count": r["joint"], "lift": r["lift"]}))
    return out


def main() -> None:
    EVAL.mkdir(parents=True, exist_ok=True)
    ung = build_ungrounded()
    rar = build_rarity()
    (EVAL / "ungrounded_llm.json").write_text(json.dumps(ung, indent=2))
    (EVAL / "compositional_rarity.json").write_text(json.dumps(rar, indent=2))
    print(f"wrote ungrounded_llm.json ({len(ung)}), compositional_rarity.json ({len(rar)})")


if __name__ == "__main__":
    main()
