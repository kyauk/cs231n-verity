"""Module 6: Evaluation — report rendering.

render_markdown(report)   → GitHub-flavored markdown string (for the paper)
render_html(report, ...)  → HTML string (standalone by default; embeddable fragment
                             via embeddable=True for injection into a React dashboard)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.interfaces.report import EvaluationReport


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------

def render_markdown(report: "EvaluationReport") -> str:
    """Render an EvaluationReport as GitHub-flavored markdown.

    Suitable for pasting directly into a paper or README.
    All numbers are rounded to 3 decimal places.
    CIs are shown as (lo, hi); None is shown as "—".
    """
    lines: list[str] = []
    a = lines.append

    a("# Phase 1 Evaluation Report\n")

    # --- Seeded Recall ---
    a("## Seeded Recall\n")
    a(f"Primary K = {report.recall_k_primary} (pre-registered)\n")
    arms = sorted(report.seeded_recall)
    subsets = ["overall", "familiar", "unfamiliar"]
    ks = ["@10", f"@{report.recall_k_primary}", "@all"]
    # Deduplicate k list while preserving order
    seen: set[str] = set()
    unique_ks = [k for k in ks if not (k in seen or seen.add(k))]  # type: ignore[func-returns-value]

    # Header
    header_cols = ["Arm", "Subset"] + unique_ks
    a("| " + " | ".join(header_cols) + " |")
    a("| " + " | ".join(["---"] * len(header_cols)) + " |")

    for arm in arms:
        for subset in subsets:
            row = [arm, subset]
            for k in unique_ks:
                val = report.seeded_recall.get(arm, {}).get(subset, {}).get(k)
                row.append(f"{val:.3f}" if val is not None else "—")
            a("| " + " | ".join(row) + " |")

    a("")

    # --- Rating Statistics ---
    a("## Expert Ratings\n")
    rating_arms = sorted(report.mean_coherence)
    a("| Arm | Mean Coherence | CI 95% | Mean Usefulness | CI 95% | N ratings |")
    a("| --- | --- | --- | --- | --- | --- |")
    for arm in rating_arms:
        mc = report.mean_coherence.get(arm)
        mu = report.mean_usefulness.get(arm)
        cc = report.coherence_ci_95.get(arm)
        uc = report.usefulness_ci_95.get(arm)
        n = report.n_ratings_per_arm.get(arm, 0)
        cc_str = f"({cc[0]:.3f}, {cc[1]:.3f})" if cc else "— (n < 30)"
        uc_str = f"({uc[0]:.3f}, {uc[1]:.3f})" if uc else "— (n < 30)"
        mc_str = f"{mc:.3f}" if mc is not None and not math.isnan(mc) else "—"
        mu_str = f"{mu:.3f}" if mu is not None and not math.isnan(mu) else "—"
        a(f"| {arm} | {mc_str} | {cc_str} | {mu_str} | {uc_str} | {n} |")

    a("")

    # --- Inter-Rater Agreement ---
    a("## Inter-Rater Agreement (Krippendorff's α)\n")
    a("| Metric | α |")
    a("| --- | --- |")
    ira_c = report.inter_rater_agreement_coherence
    ira_u = report.inter_rater_agreement_usefulness
    ira_c_str = f"{ira_c:.3f}" if ira_c is not None else f"— (insufficient raters; overlapping={report.n_raters_overlapping})"
    ira_u_str = f"{ira_u:.3f}" if ira_u is not None else f"— (insufficient raters; overlapping={report.n_raters_overlapping})"
    a(f"| Coherence | {ira_c_str} |")
    a(f"| Usefulness | {ira_u_str} |")
    a("")

    # --- Differential Examples ---
    if report.differential_examples:
        a("## Differential Examples (Top Arm Divergences)\n")
        a("Proposals ranked by largest rank gap between arms.\n")
        arm_list = sorted({a for ex in report.differential_examples for a in ex.arm_ranks})
        rank_cols = [f"Rank ({arm})" for arm in arm_list]
        score_cols = [f"Score ({arm})" for arm in arm_list]
        coh_cols = [f"Coherence ({arm})" for arm in arm_list]
        cols = ["Proposal ID", "Constituents"] + rank_cols + score_cols + coh_cols
        a("| " + " | ".join(cols) + " |")
        a("| " + " | ".join(["---"] * len(cols)) + " |")
        for ex in report.differential_examples[:10]:
            constituents_str = ", ".join(ex.constituents)
            row = [ex.proposal_id[:8], constituents_str]
            for arm in arm_list:
                row.append(str(ex.arm_ranks.get(arm, "—")))
            for arm in arm_list:
                s = ex.arm_scores.get(arm)
                row.append(f"{s:.3f}" if s is not None else "—")
            for arm in arm_list:
                c = ex.coherence_ratings.get(arm)
                row.append(f"{c:.2f}" if c is not None else "—")
            a("| " + " | ".join(row) + " |")
        a("")

    # --- Failure Mode Distribution ---
    if report.failure_mode_distribution:
        a("## Encoder Failure Mode Distribution\n")
        a("| Failure Mode | Count |")
        a("| --- | --- |")
        for fm, count in sorted(report.failure_mode_distribution.items(), key=lambda x: -x[1]):
            a(f"| {fm} | {count} |")
        a("")

    # --- Methodology ---
    a("## Methodology\n")
    a(f"- Raters: {report.n_raters} total")
    a(f"- Seeded set: {report.seeded_set_size.get('familiar', 0)} familiar + {report.seeded_set_size.get('unfamiliar', 0)} unfamiliar")
    for arm in sorted(report.n_proposals_per_arm):
        npa = report.n_proposals_per_arm[arm]
        npf = report.n_proposals_filtered.get(arm, 0)
        a(f"- {arm}: {npa} accepted proposals, {npf} filtered")
    a("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML renderer
# ---------------------------------------------------------------------------

def render_html(report: "EvaluationReport", embeddable: bool = False) -> str:
    """Render an EvaluationReport as an HTML page (or embeddable fragment).

    Parameters
    ----------
    report
        The EvaluationReport to render.
    embeddable
        If False (default): returns a full standalone HTML page with plotly
        loaded from CDN. Open directly in a browser.
        If True: returns only the chart <div>s and inline Plotly.newPlot()
        calls (include_plotlyjs=False). Suitable for injection into a React
        dashboard that already loads plotly as a global.
    """
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        has_plotly = True
    except ImportError:
        has_plotly = False

    chart_divs: list[str] = []

    if has_plotly:
        chart_divs.append(_recall_chart(report, embeddable))
        chart_divs.append(_rating_chart(report, embeddable))

    stats_table = _html_stats_table(report)
    ira_block = _html_ira_block(report)
    diff_block = _html_diff_block(report)

    body = "\n".join([
        stats_table,
        ira_block,
        *chart_divs,
        diff_block,
    ])

    if embeddable:
        return body

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Verity Phase 1 Evaluation Report</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 960px; margin: 40px auto; padding: 0 24px; color: #1a1a1a; }}
  h1 {{ font-size: 1.5rem; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; }}
  h2 {{ font-size: 1.1rem; margin-top: 32px; color: #2d3748; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 0.875rem; }}
  th, td {{ border: 1px solid #e2e8f0; padding: 8px 12px; text-align: left; }}
  th {{ background: #f7fafc; font-weight: 600; }}
  .note {{ color: #718096; font-size: 0.8rem; margin: 4px 0; }}
  .chart {{ margin: 24px 0; }}
</style>
</head>
<body>
<h1>Verity Phase 1 Evaluation Report</h1>
{body}
</body>
</html>"""


def _recall_chart(report: "EvaluationReport", embeddable: bool) -> str:
    import plotly.graph_objects as go
    import plotly.io as pio

    arms = sorted(report.seeded_recall)
    subsets = ["overall", "familiar", "unfamiliar"]
    k_label = f"@{report.recall_k_primary}"

    fig = go.Figure()
    for arm in arms:
        y_vals = [
            report.seeded_recall.get(arm, {}).get(sub, {}).get(k_label, 0.0)
            for sub in subsets
        ]
        fig.add_trace(go.Bar(name=arm, x=subsets, y=y_vals))

    fig.update_layout(
        title=f"Seeded Recall @ {report.recall_k_primary} (pre-registered K)",
        barmode="group",
        yaxis=dict(range=[0, 1], title="Recall"),
        xaxis_title="Subset",
        height=350,
        margin=dict(t=50, b=40),
    )

    return pio.to_html(
        fig,
        include_plotlyjs=not embeddable,
        full_html=False,
        div_id="chart-recall",
    )


def _rating_chart(report: "EvaluationReport", embeddable: bool) -> str:
    import plotly.graph_objects as go
    import plotly.io as pio

    arms = sorted(report.mean_coherence)
    fig = go.Figure()

    for metric, vals, ci_dict, color in [
        ("Coherence", report.mean_coherence, report.coherence_ci_95, "#4299e1"),
        ("Usefulness", report.mean_usefulness, report.usefulness_ci_95, "#48bb78"),
    ]:
        means = [vals.get(a, float("nan")) for a in arms]
        error_y: dict | None = None
        if any(ci_dict.get(a) is not None for a in arms):
            errs = [
                ((ci_dict[a][1] - ci_dict[a][0]) / 2) if ci_dict.get(a) else 0.0
                for a in arms
            ]
            error_y = dict(type="data", array=errs, visible=True)

        fig.add_trace(go.Bar(
            name=metric,
            x=arms,
            y=means,
            error_y=error_y,
            marker_color=color,
        ))

    fig.update_layout(
        title="Mean Expert Ratings (±95% CI where n≥30)",
        barmode="group",
        yaxis=dict(range=[1, 5], title="Score (1–5)"),
        xaxis_title="Arm",
        height=350,
        margin=dict(t=50, b=40),
    )

    return pio.to_html(
        fig,
        include_plotlyjs=False,
        full_html=False,
        div_id="chart-ratings",
    )


def _html_stats_table(report: "EvaluationReport") -> str:
    arms = sorted(report.mean_coherence)
    rows = ""
    for arm in arms:
        mc = report.mean_coherence.get(arm)
        mu = report.mean_usefulness.get(arm)
        n = report.n_ratings_per_arm.get(arm, 0)
        cc = report.coherence_ci_95.get(arm)
        uc = report.usefulness_ci_95.get(arm)
        import math
        mc_str = f"{mc:.3f}" if mc is not None and not math.isnan(mc) else "—"
        mu_str = f"{mu:.3f}" if mu is not None and not math.isnan(mu) else "—"
        cc_str = f"({cc[0]:.3f}, {cc[1]:.3f})" if cc else "<em>n &lt; 30</em>"
        uc_str = f"({uc[0]:.3f}, {uc[1]:.3f})" if uc else "<em>n &lt; 30</em>"
        rows += f"<tr><td>{arm}</td><td>{mc_str}</td><td>{cc_str}</td><td>{mu_str}</td><td>{uc_str}</td><td>{n}</td></tr>\n"

    return f"""<h2>Expert Ratings</h2>
<table>
  <thead><tr><th>Arm</th><th>Mean Coherence</th><th>CI 95%</th><th>Mean Usefulness</th><th>CI 95%</th><th>N ratings</th></tr></thead>
  <tbody>{rows}</tbody>
</table>"""


def _html_ira_block(report: "EvaluationReport") -> str:
    ira_c = report.inter_rater_agreement_coherence
    ira_u = report.inter_rater_agreement_usefulness
    n_ov = report.n_raters_overlapping

    def _fmt(v: float | None, label: str) -> str:
        if v is None:
            return f"<td>— <span class='note'>(insufficient overlapping raters: {n_ov})</span></td>"
        return f"<td>{v:.3f}</td>"

    return f"""<h2>Inter-Rater Agreement (Krippendorff's α, ordinal)</h2>
<table>
  <thead><tr><th>Metric</th><th>α</th></tr></thead>
  <tbody>
    <tr><td>Coherence</td>{_fmt(ira_c, 'coherence')}</tr>
    <tr><td>Usefulness</td>{_fmt(ira_u, 'usefulness')}</tr>
  </tbody>
</table>
<p class='note'>Raters in session: {report.n_raters} | Raters with overlapping ratings: {n_ov}</p>"""


def _html_diff_block(report: "EvaluationReport") -> str:
    if not report.differential_examples:
        return "<h2>Differential Examples</h2><p class='note'>Single-arm run — differential analysis requires ≥2 arms.</p>"

    arm_list = sorted({a for ex in report.differential_examples for a in ex.arm_ranks})
    header = "<tr><th>Proposal ID</th><th>Constituents</th>"
    for arm in arm_list:
        header += f"<th>Rank ({arm})</th>"
    for arm in arm_list:
        header += f"<th>Coherence ({arm})</th>"
    header += "</tr>"

    rows = ""
    for ex in report.differential_examples[:10]:
        rows += "<tr>"
        rows += f"<td><code>{ex.proposal_id[:8]}</code></td>"
        rows += f"<td>{', '.join(ex.constituents)}</td>"
        for arm in arm_list:
            rows += f"<td>{ex.arm_ranks.get(arm, '—')}</td>"
        for arm in arm_list:
            c = ex.coherence_ratings.get(arm)
            rows += f"<td>{c:.2f}</td>" if c is not None else "<td>—</td>"
        rows += "</tr>"

    return f"""<h2>Differential Examples (Top Arm Divergences)</h2>
<table><thead>{header}</thead><tbody>{rows}</tbody></table>"""
