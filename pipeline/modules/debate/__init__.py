"""Module 9: Debate — multi-agent, tool-augmented Risk-vs-Coverage analysis.

Consumes a flagged/anomalous window (typically from Module 8: Clustering) and
runs a four-actor debate (Scene Analyst -> Risk Assessor <-> Coverage Analyst
-> Synthesis Arbiter) to produce a RegressionCaseProposal. Independent lego
module: depends only on pipeline.interfaces + injected model-client Protocols.

Public surface:
    from pipeline.modules.debate import (
        Debater, DebateConfig, StubTextLLMClient, StubVLMClient,
        NIMTextLLMClient, NIMVLMClient,
    )
"""

from pipeline.modules.debate.clients import NIMTextLLMClient, NIMVLMClient
from pipeline.modules.debate.config import (
    DebateConfig,
    DebateError,
    DebateModelUnavailableError,
    StubTextLLMClient,
    StubVLMClient,
    TextLLMClient,
    VLMClient,
)
from pipeline.modules.debate.debate import Debater

__all__ = [
    "Debater",
    "DebateConfig",
    "TextLLMClient",
    "VLMClient",
    "StubTextLLMClient",
    "StubVLMClient",
    "NIMTextLLMClient",
    "NIMVLMClient",
    "DebateError",
    "DebateModelUnavailableError",
]
