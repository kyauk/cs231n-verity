"""The firewall, enforced structurally — the curator cannot see the discoverer.

If the curator could import the hypothesizer, it could (accidentally or not)
tune labels to flatter novelty scores — the system grading its own homework.
We make that unbuildable: no source file under the curator package may reference
the hypothesizer, and importing the curator must not pull the hypothesizer in.
"""

from __future__ import annotations

import sys
from pathlib import Path

CURATOR_DIR = Path(__file__).resolve().parents[1]


def test_no_hypothesizer_import_in_curator_source():
    # The firewall is about IMPORT PATHS, not prose — docstrings may explain it.
    import re
    offenders = []
    for py in CURATOR_DIR.rglob("*.py"):
        if "tests" in py.parts:
            continue
        for line in py.read_text(encoding="utf-8").splitlines():
            if re.match(r"\s*(from|import)\b.*hypothesizer", line):
                offenders.append((py.name, line.strip()))
    assert not offenders, f"curator must not IMPORT the hypothesizer: {offenders}"


def test_importing_curator_does_not_import_hypothesizer():
    for mod in list(sys.modules):
        if "hypothesizer" in mod:
            del sys.modules[mod]
    import pipeline.modules.curator  # noqa: F401
    assert not any("hypothesizer" in m for m in sys.modules), (
        "importing the curator pulled in the hypothesizer — the firewall leaks"
    )


def test_curator_depends_only_on_interfaces():
    """Curator source may import pipeline.interfaces.* but no other pipeline.modules.*"""
    import re
    bad = []
    for py in CURATOR_DIR.rglob("*.py"):
        if "tests" in py.parts:
            continue
        for line in py.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*from (pipeline\.modules\.\w+)", line)
            if m and not m.group(1).startswith("pipeline.modules.curator"):
                bad.append((py.name, line.strip()))
    assert not bad, f"curator reached into another module: {bad}"
