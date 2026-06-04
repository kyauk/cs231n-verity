"""Round-trip contract tests for the debate interface types."""

from pipeline.interfaces.debate import (
    DebateInput,
    DebateResult,
    RegressionCaseProposal,
    SceneDescription,
)


def test_debate_input_roundtrip():
    d = DebateInput(
        run_id="r1", window_id="seg/0001", scene_token_hex="abc", log_id="seg",
        scene_description="a cyclist swerves", anomaly_rationale="rare maneuver",
        severity_hint="high", regression_suite=["clear day cruising"],
        media_refs=["https://x/clip.mp4"], metadata={"k": 1},
    )
    assert DebateInput.from_json(d.to_json()) == d


def test_proposal_roundtrip():
    p = RegressionCaseProposal(
        case_id="c1", run_id="r1", window_id="seg/0001", scene_token_hex="abc",
        log_id="seg", generated_at="2026-06-04T00:00:00+00:00",
        failure_mode="late detection", why_anomalous="occlusion",
        evidence_summary="...", risk_level="high", affected_capability="VRU detection",
        affected_odds=["urban"], counterarguments=["already covered?"],
        rebuttal_summary="not at this TTC", decision="add_to_suite",
        recommended_test_spec="spec", scenario_variants=["v1", "v2"], confidence=0.8,
        uncertainty_factors=["lighting"], debate_transcript=["round 1..."],
        model_source="stub", metadata={},
    )
    assert RegressionCaseProposal.from_json(p.to_json()) == p


def test_debate_result_roundtrip_nested():
    r = DebateResult(
        window_id="seg/0001", decision="yes", recommendation="add_immediately",
        priority_score=0.91, rationale="high risk, low coverage",
        proposal=RegressionCaseProposal(
            case_id="c1", run_id="r1", window_id="seg/0001", scene_token_hex="abc",
            log_id="seg", generated_at="t", failure_mode="x", why_anomalous="y",
            evidence_summary="z", risk_level="high", affected_capability="c",
            decision="add_to_suite", recommended_test_spec="s", confidence=0.8,
        ),
        description=SceneDescription(
            run_id="r1", window_id="seg/0001", scene_description="d",
            anomaly_rationale="a", confidence="high", model_source="stub",
        ),
        model_source="stub",
    )
    r2 = DebateResult.from_json(r.to_json())
    assert r2 == r
    assert r2.proposal.decision == "add_to_suite"
    assert r2.description.confidence == "high"
