"""Module 6: Evaluation — seeded recall, rating stats, and the final report.

Aggregates ScoredProposals and human Ratings into an EvaluationReport (seeded
recall, inter-rater agreement, differential examples). evaluate() is pure;
save() is the only I/O.

Public surface (import from the package root):
    from pipeline.modules.evaluation import Evaluator, EvaluationInput
"""

from pipeline.modules.evaluation.evaluator import (
    ArmMismatchError,
    EvaluationInput,
    Evaluator,
    MissingSubsetLabelsError,
    RECALL_K_PRIMARY,
)

__all__ = [
    "Evaluator",
    "EvaluationInput",
    "MissingSubsetLabelsError",
    "ArmMismatchError",
    "RECALL_K_PRIMARY",
]
