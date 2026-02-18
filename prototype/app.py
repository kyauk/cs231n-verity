"""
AV Failure Triage Engine — Streamlit prototype.
Ingests failure logs, simulates triage pipeline, generates Isaac Sim-compatible
JSON via OpenAI gpt-4o.
"""

from __future__ import annotations

import json
import re
import time

import streamlit as st
from openai import OpenAI

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AV Failure Triage Engine",
    page_icon="https://www.nvidia.com/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Full CSS overhaul — Datadog / Google-Drive-inspired dark dashboard
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
/* ---------- fonts ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
*, html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }

/* ---------- base ---------- */
.stApp {
    background: linear-gradient(168deg, #0b0e14 0%, #121620 50%, #0f1318 100%);
}
header[data-testid="stHeader"] { background: transparent !important; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {
    background: #141922;
    border-right: 1px solid #1e2533;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label { color: #a0aec0; }

/* ---------- typography ---------- */
h1 { color: #f0f4f8 !important; font-weight: 700 !important; letter-spacing: -0.5px; }
h2 { color: #cbd5e1 !important; font-weight: 600 !important; }
h3 { color: #94a3b8 !important; font-weight: 600 !important; }
p, label, .stText, .stMarkdown { color: #cbd5e1; }

/* ---------- text area ---------- */
.stTextArea > div > div > textarea {
    background: #1a2030 !important;
    color: #e2e8f0 !important;
    border: 1px solid #2d3748 !important;
    border-radius: 10px !important;
    padding: 14px !important;
    font-size: 14px !important;
    transition: border-color 0.2s;
}
.stTextArea > div > div > textarea:focus {
    border-color: #76b900 !important;
    box-shadow: 0 0 0 2px rgba(118,185,0,0.15) !important;
}

/* ---------- text input (sidebar key) ---------- */
.stTextInput > div > div > input {
    background: #1a2030 !important;
    color: #e2e8f0 !important;
    border: 1px solid #2d3748 !important;
    border-radius: 8px !important;
    padding: 10px 12px !important;
    font-size: 13px !important;
}
.stTextInput > div > div > input:focus {
    border-color: #76b900 !important;
    box-shadow: 0 0 0 2px rgba(118,185,0,0.15) !important;
}

/* ---------- primary button (Analyze) ---------- */
.stButton > button {
    background: linear-gradient(135deg, #76b900 0%, #5a9a00 100%) !important;
    color: #fff !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.55rem 2.2rem !important;
    transition: all 0.2s ease;
    box-shadow: 0 2px 8px rgba(118,185,0,0.25);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #8ed100 0%, #6db300 100%) !important;
    box-shadow: 0 4px 16px rgba(118,185,0,0.35);
    transform: translateY(-1px);
}
.stButton > button:active { transform: translateY(0); }

/* ---------- download button ---------- */
.stDownloadButton > button {
    background: transparent !important;
    color: #76b900 !important;
    border: 1px solid #76b900 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease;
}
.stDownloadButton > button:hover {
    background: rgba(118,185,0,0.1) !important;
    box-shadow: 0 2px 12px rgba(118,185,0,0.2);
}

/* ---------- status widget ---------- */
div[data-testid="stStatusWidget"] {
    background: #161d2b !important;
    border: 1px solid #1e2a3a !important;
    border-radius: 12px !important;
}

/* ---------- json viewer ---------- */
div[data-testid="stJson"] {
    background: #131a27 !important;
    border: 1px solid #1e2a3a !important;
    border-radius: 12px !important;
    padding: 4px !important;
}

/* ---------- expander ---------- */
div[data-testid="stExpander"] {
    background: #161d2b !important;
    border: 1px solid #1e2a3a !important;
    border-radius: 12px !important;
}

/* ---------- cards (custom via markdown) ---------- */
.dash-card {
    background: #161d2b;
    border: 1px solid #1e2a3a;
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 16px;
}
.dash-card h3 { margin-top: 0; }

/* ---------- branded header bar ---------- */
.header-bar {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 18px 0 10px 0;
    border-bottom: 1px solid #1e2a3a;
    margin-bottom: 28px;
}
.header-bar .logo {
    width: 38px; height: 38px;
    background: linear-gradient(135deg, #76b900, #5a9a00);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; font-weight: 700; color: #fff;
}
.header-bar .title {
    font-size: 22px; font-weight: 700; color: #f0f4f8;
    letter-spacing: -0.3px;
}
.header-bar .subtitle {
    font-size: 13px; color: #64748b; margin-left: auto;
}

/* ---------- pipeline step chips ---------- */
.pipeline-steps {
    display: flex; gap: 10px; flex-wrap: wrap;
    margin: 16px 0 8px 0;
}
.step-chip {
    display: inline-flex; align-items: center; gap: 7px;
    background: #1a2235; border: 1px solid #253044;
    border-radius: 20px; padding: 6px 16px;
    font-size: 13px; color: #94a3b8; font-weight: 500;
}
.step-chip .dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #76b900;
}
.step-chip.active { border-color: #76b900; color: #e2e8f0; }
.step-chip.done .dot { background: #76b900; }
.step-chip.pending .dot { background: #334155; }

/* ---------- result header ---------- */
.result-header {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 12px;
}
.result-header .badge {
    background: rgba(118,185,0,0.12);
    color: #76b900;
    font-size: 12px; font-weight: 600;
    padding: 4px 12px; border-radius: 20px;
}

/* ---------- hide streamlit footer / menu ---------- */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — API key
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<div style='padding:4px 0 18px 0'>"
        "<span style='font-size:20px;font-weight:700;color:#f0f4f8'>Settings</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<span style='font-size:12px;font-weight:600;color:#64748b;"
        "text-transform:uppercase;letter-spacing:0.5px'>API Access</span>",
        unsafe_allow_html=True,
    )
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Required for gpt-4o. Never stored or logged.",
        label_visibility="collapsed",
    )
    if api_key:
        st.session_state["openai_api_key"] = api_key
    elif "openai_api_key" not in st.session_state:
        st.session_state["openai_api_key"] = ""

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<span style='font-size:12px;font-weight:600;color:#64748b;"
        "text-transform:uppercase;letter-spacing:0.5px'>About</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size:13px;color:#64748b;line-height:1.6'>"
        "Paste an AV failure log and the engine generates a <b style='color:#94a3b8'>"
        "plausible new scenario</b> as Isaac&nbsp;Sim-compatible JSON using GPT-4o.</p>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Header bar
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="header-bar">
        <div class="logo">N</div>
        <div class="title">AV Failure Triage Engine</div>
        <div class="subtitle">NVIDIA Isaac Sim &middot; Scenario Generator</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Input layer
# ---------------------------------------------------------------------------
st.markdown(
    "<h2 style='margin-bottom:4px'>Failure Log Input</h2>"
    "<p style='font-size:14px;color:#64748b;margin-bottom:14px'>"
    "Paste a DMV / NHTSA accident report or any unstructured AV failure description.</p>",
    unsafe_allow_html=True,
)

failure_logs = st.text_area(
    "Failure logs",
    height=180,
    placeholder="Example: On 03/15/2024 at approx. 14:32, a Waymo Jaguar I-PACE operating in autonomous mode struck a pedestrian…",
    key="failure_logs",
    label_visibility="collapsed",
)

col_btn, col_spacer = st.columns([1, 5])
with col_btn:
    analyze_clicked = st.button("Analyze", use_container_width=True)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> str:
    """Pull the first JSON object or fenced block from model output."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        return fence.group(1).strip()
    brace = re.search(r"\{[\s\S]*\}", text)
    if brace:
        return brace.group(0).strip()
    return text


def _validate_isaac_schema(data: dict) -> tuple[bool, str]:
    """Lightweight check for the three required top-level keys."""
    if not isinstance(data, dict):
        return False, "Output is not a JSON object."
    for key in ("scene_info", "assets", "lighting"):
        if key not in data:
            return False, f"Missing required key: '{key}'"
    if not isinstance(data.get("assets"), list):
        return False, "'assets' must be an array."
    return True, ""


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an expert at turning real-world AV / robot failure reports into \
simulation scenarios for NVIDIA Isaac Sim.

TASK
1. The user will paste a real failure case (DMV / NHTSA accident report or \
   robot / AV incident description).
2. Generate ONE new, plausible, similar failure scenario — same kind of \
   situation but different specifics (locations, objects, numbers).
3. Output ONLY valid JSON (no markdown fences, no explanation) that describes \
   the generated failure scenario as an Isaac Sim scene.

REQUIRED JSON SCHEMA (follow exactly):

{
  "scene_info": {
    "name": "<PascalCase_Scenario_Name>",
    "units": "meters",
    "up_axis": "Z",
    "stage_units_in_meters": 1.0
  },
  "assets": [
    {
      "name": "string",
      "type": "Robot | Object",
      "prim_path": "/World/...",
      "usd_path": "omniverse://localhost/NVIDIA/Assets/Isaac/4.0/Isaac/...",
      "transform": {
        "position": [x, y, z],
        "orientation": [w, x, y, z],
        "scale": [sx, sy, sz]
      },
      "physics": {
        "rigid_body_enabled": true,
        "collision_enabled": true,
        "mass": <number>,
        "articulation_root_enabled": true,
        "collision_approximation": "convexDecomposition"
      }
    }
  ],
  "lighting": {
    "prim_path": "/World/DefaultLight",
    "type": "DomeLight",
    "intensity": 1000.0,
    "color": [1.0, 1.0, 1.0]
  }
}

GUIDELINES
- Use realistic omniverse USD paths \
  (e.g. omniverse://localhost/NVIDIA/Assets/Isaac/4.0/Isaac/Robots/Franka/franka_alt_fingers.usd).
- Include "articulation_root_enabled" only for assets of type "Robot".
- Generate at least two assets (one Robot, one or more Objects).
- Output ONLY the JSON object — nothing else."""

# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------
if analyze_clicked:
    resolved_key = st.session_state.get("openai_api_key", "")

    if not resolved_key:
        st.error("Please enter your OpenAI API Key in the sidebar.")
    elif not failure_logs or not failure_logs.strip():
        st.warning("Please paste failure log text before analyzing.")
    else:
        # -- Pipeline status chips (visual) ------------------------------------
        steps = [
            ("Extracting Causal Factors", 1.0),
            ("Mapping Assets", 1.0),
            ("Calculating Novelty", 1.0),
        ]

        progress_bar = st.empty()
        chip_area = st.empty()

        for i, (label, delay) in enumerate(steps):
            chips_html = '<div class="pipeline-steps">'
            for j, (s, _) in enumerate(steps):
                if j < i:
                    cls = "step-chip done"
                elif j == i:
                    cls = "step-chip active"
                else:
                    cls = "step-chip pending"
                chips_html += f'<span class="{cls}"><span class="dot"></span>{s}</span>'
            chips_html += "</div>"
            chip_area.markdown(chips_html, unsafe_allow_html=True)
            time.sleep(delay)

        chips_html = '<div class="pipeline-steps">'
        for s, _ in steps:
            chips_html += f'<span class="step-chip done"><span class="dot"></span>{s}</span>'
        chips_html += "</div>"
        chip_area.markdown(chips_html, unsafe_allow_html=True)

        # -- Logic layer: OpenAI gpt-4o ----------------------------------------
        raw_content: str | None = None
        try:
            client = OpenAI(api_key=resolved_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": failure_logs.strip()},
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            raw_content = response.choices[0].message.content
        except Exception as exc:
            msg = str(exc).lower()
            if "api_key" in msg or "authentication" in msg or "auth" in msg:
                st.error("Invalid or missing API key — check the sidebar.")
            elif "rate" in msg or "overloaded" in msg:
                st.error("Rate-limited or server overloaded. Try again shortly.")
            else:
                st.error(f"OpenAI request failed: {exc}")

        chip_area.empty()

        # -- Parse and persist result in session_state -------------------------
        if raw_content:
            try:
                json_str = _extract_json(raw_content)
                data = json.loads(json_str)
                ok, err = _validate_isaac_schema(data)
                if not ok:
                    st.session_state["last_result"] = None
                    st.session_state["last_error"] = f"Schema validation failed: {err}"
                    st.session_state["last_raw"] = raw_content
                else:
                    st.session_state["last_result"] = data
                    st.session_state["last_error"] = None
                    st.session_state["last_raw"] = None
            except json.JSONDecodeError as exc:
                st.session_state["last_result"] = None
                st.session_state["last_error"] = f"Model returned invalid JSON: {exc}"
                st.session_state["last_raw"] = raw_content

# ---------------------------------------------------------------------------
# Output layer — persisted via session_state
# ---------------------------------------------------------------------------
if st.session_state.get("last_error"):
    st.error(st.session_state["last_error"])
    raw = st.session_state.get("last_raw")
    if raw:
        with st.expander("Raw model response"):
            st.code(raw, language="text")

elif st.session_state.get("last_result"):
    data = st.session_state["last_result"]

    st.markdown(
        '<div class="result-header">'
        "<h2 style='margin:0'>Generated Scenario</h2>"
        '<span class="badge">Isaac Sim Ready</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    scene_name = data.get("scene_info", {}).get("name", "Scenario")
    num_assets = len(data.get("assets", []))
    light_type = data.get("lighting", {}).get("type", "—")

    mc1, mc2, mc3 = st.columns(3)
    mc1.markdown(
        f"<div class='dash-card'><h3 style='color:#64748b;font-size:12px;"
        f"text-transform:uppercase;letter-spacing:0.5px'>Scene Name</h3>"
        f"<p style='font-size:18px;font-weight:600;color:#f0f4f8;margin:0'>{scene_name}</p></div>",
        unsafe_allow_html=True,
    )
    mc2.markdown(
        f"<div class='dash-card'><h3 style='color:#64748b;font-size:12px;"
        f"text-transform:uppercase;letter-spacing:0.5px'>Assets</h3>"
        f"<p style='font-size:18px;font-weight:600;color:#f0f4f8;margin:0'>{num_assets}</p></div>",
        unsafe_allow_html=True,
    )
    mc3.markdown(
        f"<div class='dash-card'><h3 style='color:#64748b;font-size:12px;"
        f"text-transform:uppercase;letter-spacing:0.5px'>Lighting</h3>"
        f"<p style='font-size:18px;font-weight:600;color:#f0f4f8;margin:0'>{light_type}</p></div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.json(data)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.download_button(
        label="Download JSON",
        data=json.dumps(data, indent=2),
        file_name="isaac_sim_scenario.json",
        mime="application/json",
    )
