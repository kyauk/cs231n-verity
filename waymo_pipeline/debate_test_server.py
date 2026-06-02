"""Standalone test UI for the tool-augmented debate -- no Waymo data required.

This is a lightweight harness that lets you upload a single video clip and run
*only* the new four-actor tool-augmented debate (Scene Analyst, Risk Assessor,
Coverage Analyst, Synthesis Arbiter) against it. It deliberately skips the
Waymo dataset download, embedding, clustering, and anomaly-flagging stages so
you can iterate on the debate / VLM-follow-up behavior in isolation.

All VLM work (initial description + ``vlm_followup`` re-queries) goes through
the hosted NVIDIA NIM vision API -- no local model is loaded.

Run (from the repo root):

    uvicorn waymo_pipeline.debate_test_server:app --host 0.0.0.0 --port 8100 --reload

then open http://localhost:8100 in a browser.

Endpoints
  GET  /            -- the upload + results UI (single self-contained page)
  GET  /health      -- connectivity probe
  POST /api/debate  -- multipart: video file (+ optional context) -> debate JSON
"""

from __future__ import annotations

import os
import tempfile
import traceback
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from waymo_pipeline.debate_actors import run_tool_augmented_debate
from waymo_pipeline.models.handoff_contracts import DebateInputRecord
from waymo_pipeline.proposal_builder import build_proposal_from_debate_output
from waymo_pipeline.waymo_describe_and_debate import (
    _nim_vlm_describe,
    _parse_description,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=True)

_VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".webm")

app = FastAPI(title="Debate Test UI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    """Simple connectivity + config probe."""
    return {
        "ok": True,
        "nvidia_api_key_set": bool(os.getenv("NVIDIA_API_KEY")),
        "describe_model": os.getenv("DESCRIBE_NIM_MODEL_ID", "nvidia/nemotron-nano-12b-v2-vl"),
        "debate_model": os.getenv("DEBATE_NIM_MODEL_ID", "meta/llama-3.1-8b-instruct"),
        "vlm_followup_enabled": os.getenv("REACT_ENABLE_VLM_FOLLOWUP", "1"),
    }


@app.post("/api/debate")
async def run_debate(
    video: UploadFile = File(...),
    scene_description: str = Form(""),
    anomaly_rationale: str = Form(""),
    regression_suite: str = Form(""),
    severity_hint: str = Form("medium"),
    rounds: int = Form(2),
) -> JSONResponse:
    """Run description (optional) + tool-augmented debate on one uploaded clip.

    If ``scene_description`` is provided, the VLM description stage is skipped
    and the debate runs directly on the supplied text (the clip is still used
    for ``vlm_followup`` re-queries). Otherwise the clip is described via the
    NIM VLM first.
    """

    if not os.getenv("NVIDIA_API_KEY"):
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": "NVIDIA_API_KEY is not set in the environment / .env."},
        )

    suffix = Path(video.filename or "clip.mp4").suffix.lower()
    if suffix not in _VIDEO_EXTS:
        suffix = ".mp4"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    try:
        tmp.write(await video.read())
        tmp.close()

        run_id = f"test_{uuid.uuid4().hex[:8]}"
        window_id = f"win_{uuid.uuid4().hex[:6]}"

        # --- Description stage (optional) -------------------------------------
        description = scene_description.strip()
        rationale = anomaly_rationale.strip()
        confidence = "unknown"
        described_via = "user_supplied"
        if not description:
            priors = {
                "window_id": window_id,
                "source": "debate_test_ui",
                "user_note": "Ad-hoc uploaded clip (no Waymo anomaly context).",
            }
            raw = _nim_vlm_describe(os.path.abspath(tmp_path), priors)
            parsed = _parse_description(raw)
            description = parsed["scene_description"]
            rationale = rationale or parsed["anomaly_rationale"]
            confidence = parsed["confidence"]
            described_via = "nim_vlm"
        if not rationale:
            rationale = "Ad-hoc clip submitted via the debate test UI for evaluation."

        severity = severity_hint.strip().lower()
        if severity not in {"low", "medium", "high", "critical", "unknown"}:
            severity = "unknown"

        suite = [line.strip() for line in regression_suite.splitlines() if line.strip()]

        record = DebateInputRecord(
            run_id=run_id,
            window_id=window_id,
            scene_token_hex=uuid.uuid4().hex,
            log_id="debate_test_ui",
            scene_description=description,
            anomaly_rationale=rationale,
            severity_hint=severity,  # type: ignore[arg-type]
            regression_suite=suite,
            metadata={"source": "debate_test_ui"},
        )

        # --- Tool-augmented debate (VLM follow-ups hit the NIM API) -----------
        debate_output, _proposal_metadata = run_tool_augmented_debate(
            record, [os.path.abspath(tmp_path)], rounds=rounds
        )
        proposal = build_proposal_from_debate_output(debate_output, run_id)
        meta = debate_output.metadata or {}

        return JSONResponse(
            content={
                "ok": True,
                "describedVia": described_via,
                "sceneDescription": description,
                "anomalyRationale": rationale,
                "descriptionConfidence": confidence,
                "decision": debate_output.decision,
                "recommendation": debate_output.recommendation,
                "priorityScore": debate_output.priority_score,
                "rationale": debate_output.rationale,
                "debateTranscript": meta.get("debate_history", []),
                "judgeRawOutput": meta.get("judge_raw_output", ""),
                "scoring": meta.get("scoring", {}),
                "proposal": proposal.model_dump(),
            }
        )
    except Exception as error:  # noqa: BLE001
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": str(error),
                "trace": traceback.format_exc()[-3000:],
            },
        )
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Serve the single-page upload + results UI."""
    return _PAGE


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Debate Test UI</title>
<style>
  :root {
    --bg: #0b0e14; --panel: #141925; --panel2: #1b2230; --border: #2a3344;
    --text: #e6edf3; --muted: #8b98a9; --accent: #4f9dff; --good: #36c98f;
    --bad: #ff6b6b; --warn: #f0b429;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  }
  header { padding: 20px 28px; border-bottom: 1px solid var(--border); }
  header h1 { margin: 0; font-size: 18px; }
  header p { margin: 4px 0 0; color: var(--muted); font-size: 13px; }
  .wrap { display: grid; grid-template-columns: 380px 1fr; gap: 0; min-height: calc(100vh - 70px); }
  .left { padding: 20px 24px; border-right: 1px solid var(--border); }
  .right { padding: 20px 28px; overflow: auto; }
  label { display: block; font-weight: 600; margin: 14px 0 6px; font-size: 13px; }
  .hint { color: var(--muted); font-weight: 400; font-size: 12px; }
  input[type=file], textarea, select {
    width: 100%; background: var(--panel2); color: var(--text);
    border: 1px solid var(--border); border-radius: 8px; padding: 9px 10px; font: inherit;
  }
  textarea { resize: vertical; min-height: 64px; }
  button {
    margin-top: 18px; width: 100%; padding: 11px; border: 0; border-radius: 8px;
    background: var(--accent); color: #fff; font-weight: 700; font-size: 14px; cursor: pointer;
  }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 16px 18px; margin-bottom: 16px; }
  .card h3 { margin: 0 0 10px; font-size: 14px; color: var(--accent); letter-spacing: .3px; }
  .pill { display: inline-block; padding: 3px 10px; border-radius: 999px; font-weight: 700; font-size: 12px; }
  .pill.good { background: rgba(54,201,143,.15); color: var(--good); }
  .pill.bad { background: rgba(255,107,107,.15); color: var(--bad); }
  .pill.warn { background: rgba(240,180,41,.15); color: var(--warn); }
  .kv { display: grid; grid-template-columns: 160px 1fr; gap: 6px 12px; }
  .kv div:nth-child(odd) { color: var(--muted); }
  .muted { color: var(--muted); }
  pre { white-space: pre-wrap; word-break: break-word; margin: 0; font-size: 12.5px; }
  .turn { border-left: 3px solid var(--border); padding: 6px 0 6px 12px; margin: 10px 0; }
  .turn .who { font-weight: 700; font-size: 12px; }
  .turn.scene .who { color: #6cc6ff; } .turn.scene { border-color: #6cc6ff; }
  .turn.risk .who { color: #ffb86c; }  .turn.risk { border-color: #ffb86c; }
  .turn.coverage .who { color: #ff79c6; } .turn.coverage { border-color: #ff79c6; }
  .turn.arbiter .who { color: #36c98f; } .turn.arbiter { border-color: #36c98f; }
  .spinner { display: inline-block; width: 16px; height: 16px; border: 2px solid rgba(255,255,255,.3);
    border-top-color: #fff; border-radius: 50%; animation: spin .8s linear infinite; vertical-align: -3px; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .empty { color: var(--muted); margin-top: 40px; text-align: center; }
  ul { margin: 6px 0; padding-left: 18px; }
  li { margin: 3px 0; }
  .err { color: var(--bad); }
</style>
</head>
<body>
<header>
  <h1>Tool-Augmented Debate &mdash; Test UI</h1>
  <p>Upload a clip and run only the four-actor debate (no Waymo dataset / clustering). VLM follow-ups use the NIM API.</p>
</header>
<div class="wrap">
  <div class="left">
    <form id="form">
      <label>Video clip <span class="hint">(mp4 / mov / mkv / avi / webm)</span></label>
      <input type="file" id="video" accept="video/*" required />

      <label>Scene description <span class="hint">(optional &mdash; leave blank to auto-describe via VLM)</span></label>
      <textarea id="scene_description" placeholder="Leave blank to run the NIM VLM description stage first."></textarea>

      <label>Anomaly / value rationale <span class="hint">(optional)</span></label>
      <textarea id="anomaly_rationale" placeholder="Why might this be worth adding to a regression suite?"></textarea>

      <label>Severity hint</label>
      <select id="severity_hint">
        <option value="low">low</option>
        <option value="medium" selected>medium</option>
        <option value="high">high</option>
        <option value="critical">critical</option>
        <option value="unknown">unknown</option>
      </select>

      <label>Existing regression suite <span class="hint">(optional, one per line)</span></label>
      <textarea id="regression_suite" placeholder="Night-time right turn at signalized intersection.&#10;Pedestrian jaywalking mid-block."></textarea>

      <button type="submit" id="run">Run debate</button>
    </form>
    <p class="muted" id="status" style="margin-top:14px;"></p>
  </div>
  <div class="right" id="results">
    <div class="empty">Results will appear here after you run a debate.</div>
  </div>
</div>

<script>
const form = document.getElementById('form');
const statusEl = document.getElementById('status');
const runBtn = document.getElementById('run');
const results = document.getElementById('results');

function esc(s) {
  return String(s == null ? '' : s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
}

function actorClass(line) {
  if (line.includes('[Scene Analyst]')) return 'scene';
  if (line.includes('[Risk Assessor]')) return 'risk';
  if (line.includes('[Coverage Analyst]')) return 'coverage';
  if (line.includes('[Synthesis Arbiter]')) return 'arbiter';
  return '';
}

function renderTranscript(lines) {
  if (!lines || !lines.length) return '<p class="muted">No transcript produced.</p>';
  return lines.map(l => {
    const m = l.match(/^\\[(.*?)\\]/);
    const who = m ? m[1] : 'Actor';
    const rest = m ? l.slice(m[0].length).trim() : l;
    return `<div class="turn ${actorClass(l)}"><div class="who">${esc(who)}</div><pre>${esc(rest)}</pre></div>`;
  }).join('');
}

function list(items) {
  if (!items || !items.length) return '<span class="muted">none</span>';
  return '<ul>' + items.map(i => `<li>${esc(i)}</li>`).join('') + '</ul>';
}

function render(d) {
  const decisionGood = d.decision === 'yes';
  const p = d.proposal || {};
  const riskClass = ({critical:'bad', high:'bad', medium:'warn', low:'good'})[p.risk_level] || 'warn';
  results.innerHTML = `
    <div class="card">
      <h3>VERDICT</h3>
      <span class="pill ${decisionGood ? 'good' : 'bad'}">${esc((d.decision||'').toUpperCase())} &mdash; ${esc(d.recommendation)}</span>
      <span class="pill ${riskClass}" style="margin-left:8px;">risk: ${esc(p.risk_level)}</span>
      <span class="pill warn" style="margin-left:8px;">priority: ${(d.priorityScore*100||0).toFixed(0)}%</span>
      <div style="margin-top:12px;"><pre>${esc(d.rationale)}</pre></div>
    </div>

    <div class="card">
      <h3>SCENE DESCRIPTION <span class="muted" style="font-weight:400">(${esc(d.describedVia)}${d.descriptionConfidence !== 'unknown' ? ', confidence: '+esc(d.descriptionConfidence) : ''})</span></h3>
      <pre>${esc(d.sceneDescription)}</pre>
      <div style="margin-top:10px;" class="muted">Rationale:</div>
      <pre>${esc(d.anomalyRationale)}</pre>
    </div>

    <div class="card">
      <h3>REGRESSION-CASE PROPOSAL</h3>
      <div class="kv">
        <div>Failure mode</div><div>${esc(p.failure_mode)}</div>
        <div>Why anomalous</div><div>${esc(p.why_anomalous)}</div>
        <div>Evidence</div><div>${esc(p.evidence_summary)}</div>
        <div>Affected capability</div><div>${esc(p.affected_capability)}</div>
        <div>Affected ODDs</div><div>${list(p.affected_odds)}</div>
        <div>Counterarguments</div><div>${list(p.counterarguments)}</div>
        <div>Rebuttal</div><div>${esc(p.rebuttal_summary) || '<span class="muted">none</span>'}</div>
        <div>Decision</div><div>${esc(p.decision)}</div>
        <div>Confidence</div><div>${((p.confidence||0)*100).toFixed(0)}%</div>
        <div>Uncertainty</div><div>${list(p.uncertainty_factors)}</div>
        <div>Scenario variants</div><div>${list(p.scenario_variants)}</div>
      </div>
      <div style="margin-top:10px;" class="muted">Recommended test spec:</div>
      <pre>${esc(p.recommended_test_spec)}</pre>
    </div>

    <div class="card">
      <h3>DEBATE TRANSCRIPT</h3>
      ${renderTranscript(d.debateTranscript)}
    </div>

    <div class="card">
      <h3>SYNTHESIS ARBITER (raw judge output)</h3>
      <pre>${esc(d.judgeRawOutput) || '<span class="muted">n/a</span>'}</pre>
    </div>`;
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const file = document.getElementById('video').files[0];
  if (!file) { statusEl.textContent = 'Please choose a video file.'; return; }

  const fd = new FormData();
  fd.append('video', file);
  fd.append('scene_description', document.getElementById('scene_description').value);
  fd.append('anomaly_rationale', document.getElementById('anomaly_rationale').value);
  fd.append('regression_suite', document.getElementById('regression_suite').value);
  fd.append('severity_hint', document.getElementById('severity_hint').value);

  runBtn.disabled = true;
  statusEl.innerHTML = '<span class="spinner"></span> Running description + 4-actor debate (this can take 1&ndash;3 minutes)...';
  results.innerHTML = '<div class="empty"><span class="spinner"></span> Working...</div>';

  try {
    const res = await fetch('/api/debate', { method: 'POST', body: fd });
    const data = await res.json();
    if (!data.ok) {
      results.innerHTML = `<div class="card"><h3 class="err">ERROR</h3><pre class="err">${esc(data.error)}</pre>${data.trace ? `<pre class="muted" style="margin-top:10px;">${esc(data.trace)}</pre>` : ''}</div>`;
      statusEl.textContent = 'Failed.';
    } else {
      render(data);
      statusEl.textContent = 'Done.';
    }
  } catch (err) {
    results.innerHTML = `<div class="card"><h3 class="err">REQUEST FAILED</h3><pre class="err">${esc(err)}</pre></div>`;
    statusEl.textContent = 'Request failed.';
  } finally {
    runBtn.disabled = false;
  }
});
</script>
</body>
</html>"""
