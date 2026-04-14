# Teammate Setup Guide (Brev GPU + Frontend + Backend)

This guide is the handoff runbook for launching the full video upload pipeline on a fresh Brev GPU workspace.

It covers:
- syncing the latest project files
- starting backend (`uvicorn`) and frontend (`next dev`) in separate terminals
- exposing the frontend locally
- common failure fixes

## Architecture at a glance

- Backend: `pipeline/remote_gpu_runner.py` (FastAPI, port `8001`)
- Frontend: `frontend/` Next.js app (port `3000`)
- Frontend uploads call backend when `REMOTE_GPU_RUN_URL` is set

## Prerequisites

- Brev workspace is running
- Brev CLI is installed and authenticated on local machine
- NVIDIA API key
- Hugging Face token with access to required models
- Local clone path to this folder (`relevant_video_debate_files`)
- Brev workspace name (shown in Brev UI)

## 0) Copy project to the VM

Run on local machine:

```bash
export WORKSPACE_NAME="<your-brev-workspace-name>"
export LOCAL_PROJECT_DIR="<absolute/local/path/to/relevant_video_debate_files>"

brev copy "$LOCAL_PROJECT_DIR" "$WORKSPACE_NAME":~/
```

Example:

```bash
export WORKSPACE_NAME="my-gpu-workspace"
export LOCAL_PROJECT_DIR="$HOME/dev/NVIDIAproject/relevant_video_debate_files"
```

Note: if you open a new local terminal, re-run `export WORKSPACE_NAME=...` before Brev commands.

## 1) Backend terminal (VM)

Open terminal 1:

```bash
brev shell "$WORKSPACE_NAME"
```

Then run:

```bash
cd ~/relevant_video_debate_files
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export NVIDIA_API_KEY="YOUR_NVIDIA_API_KEY"
export HUGGINGFACE_HUB_TOKEN="hf_..."
export HF_TOKEN="$HUGGINGFACE_HUB_TOKEN"

uvicorn pipeline.remote_gpu_runner:app --host 0.0.0.0 --port 8001
```

Keep this terminal open.

## 2) Frontend terminal (VM)

Open terminal 2:

```bash
brev shell "$WORKSPACE_NAME"
```

Then run:

```bash
cd ~/relevant_video_debate_files/frontend
```

Install Node only if needed (`npm: command not found`):

```bash
sudo apt-get update
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
```

Start frontend:

```bash
npm install
printf "REMOTE_GPU_RUN_URL=http://127.0.0.1:8001/run-video\n" > .env.local
npx next dev -H 0.0.0.0 -p 3000
```

Keep this terminal open.

## 3) Port-forward terminal (local machine)

Open terminal 3 (local machine):

```bash
brev port-forward "$WORKSPACE_NAME" -p 3001:3000
```

Then open:
- `http://localhost:3001`
- `http://localhost:3001/video-lab`

## Quick health checks

Backend health (run in any VM shell):

```bash
curl http://127.0.0.1:8001/health
```

Frontend health from VM:

```bash
curl -I http://127.0.0.1:3000
```

## Restart commands

Backend restart:

```bash
cd ~/relevant_video_debate_files
source .venv/bin/activate
pkill -f "uvicorn pipeline.remote_gpu_runner:app" || true
uvicorn pipeline.remote_gpu_runner:app --host 0.0.0.0 --port 8001
```

Frontend restart:

```bash
cd ~/relevant_video_debate_files/frontend
pkill -f "next dev" || true
npx next dev -H 0.0.0.0 -p 3000
```

## Troubleshooting

- `uvicorn: command not found`
  - venv not active. Run `source ~/relevant_video_debate_files/.venv/bin/activate`.

- `npm: command not found`
  - Node is missing. Install Node 20 with the commands in the frontend section.

- `address already in use` on port `8001`
  - backend already running. Stop old process or use `pkill -f "uvicorn pipeline.remote_gpu_runner:app"`.

- `local port 3000 is already in use` while forwarding
  - use a different local port, for example `3001:3000`.

- `Connection refused` from port-forward
  - frontend is not running on VM. Start `npx next dev -H 0.0.0.0 -p 3000` first.

- upload returns 500 with JSON parsing style errors from description stage
  - sync latest code to VM (`brev copy ...`) and restart backend.

## Security notes

- Do not commit tokens to git.
- Rotate any token that was accidentally pasted into logs/chat.
- Prefer setting secrets as ephemeral exports in active shell sessions.

