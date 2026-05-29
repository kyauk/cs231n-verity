"""Module 3: Hypothesizer — compositional scenario discovery.

Finds compositionally novel scenario combinations in the Encoder's
SchemaRecords and emits ranked CompositionProposals. Pure function — no I/O,
no network.

Public surface (import from the package root):
    from pipeline.modules.hypothesizer import Hypothesizer, HypothesizerConfig
"""

from pipeline.modules.hypothesizer.config import HypothesizerConfig
from pipeline.modules.hypothesizer.hypothesizer import Hypothesizer

__all__ = ["Hypothesizer", "HypothesizerConfig"]
