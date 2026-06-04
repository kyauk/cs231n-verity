"""Selection module tests — pin the pure ranking logic + the firewall."""

from __future__ import annotations

from pipeline.modules.selection import (
    SelectionConfig,
    behavior_novelty,
    combined_score,
)


def test_behavior_novelty_rewards_rare_signatures():
    # 'common' appears in all 3 scenes; 'rare' in one. The scene with the rare
    # atom should score highest novelty; a scene with no atoms scores 0.
    by_scene = {
        "a": {"interactions:common"},
        "b": {"interactions:common"},
        "c": {"interactions:common", "conditions:rare"},
        "d": set(),
    }
    nov = behavior_novelty(by_scene)
    assert nov["c"] == max(nov.values())          # rarest signature -> top
    assert nov["d"] == 0.0                          # no behaviors -> 0
    assert all(0.0 <= v <= 1.0 for v in nov.values())


def test_behavior_novelty_handles_empty():
    assert behavior_novelty({}) == {}


def test_combined_score_is_difficulty_heavy():
    cfg = SelectionConfig(w_difficulty=0.7, w_novelty=0.3)
    # a hard-but-common scene beats an easy-but-novel scene
    hard_common = combined_score(difficulty=0.8, novelty=0.0, config=cfg)
    easy_novel = combined_score(difficulty=0.0, novelty=1.0, config=cfg)
    assert hard_common > easy_novel
    # a failed (-1) difficulty is floored to 0
    assert combined_score(difficulty=-1.0, novelty=1.0, config=cfg) == 0.3


def test_imports_only_interfaces_and_self():
    """Selection source must not reach into other pipeline modules (lego-brick)."""
    import re
    from pathlib import Path
    base = Path(__file__).resolve().parents[1]
    bad = []
    for py in base.rglob("*.py"):
        if "tests" in py.parts:
            continue
        for line in py.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*from (pipeline\.modules\.\w+)", line)
            if m and not m.group(1).startswith("pipeline.modules.selection"):
                bad.append((py.name, line.strip()))
    assert not bad, f"selection reached into another module: {bad}"
