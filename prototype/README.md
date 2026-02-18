# AV Failure Triage Engine (Prototype)

Streamlit prototype that ingests unstructured AV/DMV failure logs, runs a simulated triage pipeline, and uses OpenAI to generate a plausible similar failure scenario as **Isaac Sim-compatible JSON**.

## Run

```bash
cd prototype
pip install -r requirements.txt
streamlit run app.py
```

## Requirements

- **OpenAI API Key**: Enter your key in the sidebar. It is never logged or displayed.
- Output JSON can be used with NVIDIA Isaac Sim (scene_info, assets, lighting).
