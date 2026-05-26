"""Integration tests: Module 6 (Evaluation) → downstream consumers.

Module 6 is the terminal module in Phase 1. Its downstream consumers are:
  - Paper writeup (render_markdown)
  - Demo UI tab (render_html)
  - Any future Module 7 that consumes EvaluationReport

This file tests that the EvaluationReport boundary works from Module 6's
output to each consumer:

1. render_markdown: produces a non-empty string with expected section headers
2. render_html (standalone): produces a full HTML page with plotly charts
3. render_html (embeddable): produces an injectable fragment without <html>
4. EvaluationReport.to_json() / from_json(): boundary contract for any
   future module or API endpoint that reads the persisted report

# TODO: replace stubs when a frontend evaluation tab API is implemented.
"""

from __future__ import annotations

import json

import pytest

from pipeline.interfaces.proposal import ScoredProposal
from pipeline.interfaces.rating import Rating
from pipeline.interfaces.report import EvaluationReport
from pipeline.interfaces.window import WindowKey
from pipeline.modules.evaluation.evaluator import EvaluationInput, Evaluator
from pipeline.modules.evaluation.renderer import render_html, render_markdown

try:
    import plotly  # noqa: F401
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False

plotly_required = pytest.mark.skipif(
    not _PLOTLY_AVAILABLE,
    reason="plotly not installed — install it to run chart tests",
)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def _make_eval_input(n_raters: int = 3, n_proposals: int = 15) -> EvaluationInput:
    seeded = [WindowKey("seg", i) for i in range(10)]
    labels = {w: ("familiar" if i < 5 else "unfamiliar") for i, w in enumerate(seeded)}

    proposals = [
        ScoredProposal(
            composition_id=f"cid_{i:02d}",
            constituents=["conditions:fog", "time_of_day:night"],
            marginal_frequencies={"conditions:fog": 0.2, "time_of_day:night": 0.3},
            pairwise_frequencies={},
            expected_joint=0.06,
            observed_joint=0.001,
            novelty_score=4.0 - i * 0.1,
            motivating_scene_ids=[seeded[i % len(seeded)]],
            arm="reasoning",
            plausibility_score=0.85,
            plausibility_justification="Plausible.",
            frontier_difficulty_score=0.65,
            frontier_difficulty_signals={},
            final_rank_score=0.90 - i * 0.02,
            accepted=True,
            rejection_reason=None,
        )
        for i in range(n_proposals)
    ]

    ratings = [
        Rating(
            rater_id=f"rater_{ri:02d}",
            proposal_id=f"cid_{pi:02d}",
            arm="reasoning",
            coherence_score=(ri % 3) + 3,
            usefulness_score=(pi % 3) + 2,
            timestamp="2026-05-26T00:00:00Z",
        )
        for ri in range(n_raters) for pi in range(min(n_proposals, 8))
    ]

    return EvaluationInput(
        proposals_by_arm={"reasoning": proposals},
        ratings=ratings,
        seeded_window_ids=seeded,
        seeded_subset_labels=labels,
    )


@pytest.fixture(scope="module")
def report() -> EvaluationReport:
    return Evaluator().evaluate(_make_eval_input())


# ---------------------------------------------------------------------------
# Consumer 1: Paper writeup — render_markdown
# ---------------------------------------------------------------------------

class TestMarkdownConsumer:

    def test_markdown_is_non_empty_string(self, report):
        md = render_markdown(report)
        assert isinstance(md, str)
        assert len(md) > 200

    def test_markdown_has_seeded_recall_section(self, report):
        md = render_markdown(report)
        assert "Seeded Recall" in md

    def test_markdown_has_expert_ratings_section(self, report):
        md = render_markdown(report)
        assert "Expert Ratings" in md

    def test_markdown_has_ira_section(self, report):
        md = render_markdown(report)
        assert "Krippendorff" in md

    def test_markdown_has_methodology_section(self, report):
        md = render_markdown(report)
        assert "Methodology" in md

    def test_markdown_contains_primary_k(self, report):
        md = render_markdown(report)
        assert str(report.recall_k_primary) in md

    def test_markdown_contains_arm_name(self, report):
        md = render_markdown(report)
        assert "reasoning" in md

    def test_markdown_contains_n_raters(self, report):
        md = render_markdown(report)
        assert str(report.n_raters) in md

    def test_markdown_recall_values_are_numeric(self, report):
        md = render_markdown(report)
        import re
        # Match both fractional recalls (0.XXX) and 1.000 (perfect recall in fixture)
        numbers = re.findall(r'\b(?:0\.\d{3}|1\.000)\b', md)
        assert len(numbers) >= 3  # at least @10, @30, @all for overall

    def test_markdown_is_round_trippable(self, report):
        # Same input → same output (deterministic renderer)
        md1 = render_markdown(report)
        md2 = render_markdown(report)
        assert md1 == md2


# ---------------------------------------------------------------------------
# Consumer 2: Demo UI — render_html standalone
# ---------------------------------------------------------------------------

class TestHTMLStandaloneConsumer:

    def test_html_is_non_empty_string(self, report):
        html = render_html(report)
        assert isinstance(html, str)
        assert len(html) > 500

    def test_html_has_html_tag(self, report):
        html = render_html(report)
        assert "<!DOCTYPE html>" in html or "<html" in html

    def test_html_has_plotly_reference(self, report):
        html = render_html(report)
        assert "plotly" in html.lower()

    @plotly_required
    def test_html_has_recall_chart_div(self, report):
        html = render_html(report)
        assert "chart-recall" in html

    def test_html_has_stats_table(self, report):
        html = render_html(report)
        assert "Expert Ratings" in html

    def test_html_has_ira_block(self, report):
        html = render_html(report)
        assert "Krippendorff" in html


# ---------------------------------------------------------------------------
# Consumer 3: Demo UI — render_html embeddable
# ---------------------------------------------------------------------------

class TestHTMLEmbeddableConsumer:

    def test_embeddable_has_no_html_tag(self, report):
        html = render_html(report, embeddable=True)
        assert "<html" not in html
        assert "<!DOCTYPE" not in html

    @plotly_required
    def test_embeddable_has_chart_div(self, report):
        html = render_html(report, embeddable=True)
        assert "<div" in html

    def test_embeddable_does_not_load_plotly_cdn(self, report):
        html = render_html(report, embeddable=True)
        assert "cdn.plot.ly" not in html

    def test_embeddable_is_shorter_than_standalone(self, report):
        standalone = render_html(report)
        embeddable = render_html(report, embeddable=True)
        assert len(embeddable) < len(standalone)


# ---------------------------------------------------------------------------
# Consumer 4: Serialized boundary — JSON round-trip (future API / Module 7)
# ---------------------------------------------------------------------------

class TestSerializedBoundaryConsumer:
    """
    Simulates any future consumer that reads the persisted report.json.
    # TODO: replace with real Module 7 / frontend API when built.
    """

    def test_json_boundary_preserves_seeded_recall_structure(self, report):
        restored = EvaluationReport.from_json(report.to_json())
        for arm in report.seeded_recall:
            for subset in ("overall", "familiar", "unfamiliar"):
                for k in ("@10", "@30", "@all"):
                    assert k in restored.seeded_recall[arm][subset]

    def test_json_boundary_preserves_float_or_none_ira(self, report):
        restored = EvaluationReport.from_json(report.to_json())
        assert restored.inter_rater_agreement_coherence is None or isinstance(
            restored.inter_rater_agreement_coherence, float
        )

    def test_json_boundary_preserves_ci_none_correctly(self, report):
        restored = EvaluationReport.from_json(report.to_json())
        for arm, ci in restored.coherence_ci_95.items():
            assert ci is None or (isinstance(ci, tuple) and len(ci) == 2)

    def test_json_boundary_is_fully_deterministic(self, report):
        j1 = json.dumps(report.to_json(), sort_keys=True)
        j2 = json.dumps(report.to_json(), sort_keys=True)
        assert j1 == j2

    def test_json_boundary_all_required_keys_survive(self, report):
        d = report.to_json()
        required = [
            "seeded_recall", "recall_k_primary", "mean_coherence", "mean_usefulness",
            "coherence_ci_95", "usefulness_ci_95", "n_ratings_per_arm",
            "inter_rater_agreement_coherence", "inter_rater_agreement_usefulness",
            "n_raters_overlapping", "differential_examples", "failure_mode_distribution",
            "n_proposals_per_arm", "n_proposals_filtered", "n_raters", "seeded_set_size",
        ]
        for key in required:
            assert key in d
