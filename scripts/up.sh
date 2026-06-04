#!/usr/bin/env bash
# Bring up the whole Verity service with one command.
#
#   NIM (Cosmos-Reason, :8081)  +  waymo_runner API (:8000)  +  judge_ui (:8001)
#   +  Next.js frontend (:3000, single origin — proxies all of the above)
#
# Design: the browser only ever talks to :3000; the frontend proxies /probe-path,
# /batches, /judge/*, etc. to the right backend (see frontend/next.config.mjs).
# So you only forward/expose ONE port (3000).
#
# Credential split (important — a single SA can't do both here):
#   * :8000 ingest reads the Waymo SOURCE bucket -> uses your USER ADC
#     (the signer SA gets 403 on the public Waymo dataset).
#   * :8001 judge reads the OUTPUT bucket and SIGNS video URLs -> uses the SA key
#     (a user refresh-token cannot sign v4 URLs).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ---- config (override via env) --------------------------------------------
PROJECT="${VERITY_GCP_PROJECT:-nvidia-adr}"
SA_KEY="${VERITY_SIGNER_KEY:-$ROOT/pipeline/nvidia-adr-ba6d55bc8013.json}"
BUCKET="${VERITY_BUCKET:-gs://nvidia-adr-waymo-segment-videos/verity}"
OUTPUT_DIR="${VERITY_OUTPUT_DIR:-$ROOT/outputs/session-1}"
LOGDIR="$ROOT/.run"; mkdir -p "$LOGDIR"

# shellcheck disable=SC1091
source .venv/bin/activate
set -a; source .env; set +a

echo "== Verity bring-up =="
[ -f "$SA_KEY" ] || { echo "FATAL: signer key not found at $SA_KEY (set VERITY_SIGNER_KEY)"; exit 1; }

start() {  # name port "ENV=… …" "cmd…"
  local name="$1" port="$2" envs="$3" cmd="$4"
  if [ -s "$LOGDIR/$name.pid" ] && kill -0 "$(cat "$LOGDIR/$name.pid")" 2>/dev/null; then
    echo "  $name already running (pid $(cat "$LOGDIR/$name.pid"))"; return
  fi
  echo "  starting $name on :$port"
  # setsid + nohup so the service outlives this script and its shell
  env $envs setsid nohup bash -c "$cmd" >"$LOGDIR/$name.log" 2>&1 < /dev/null &
  echo $! > "$LOGDIR/$name.pid"
}

wait_for() {  # url label timeout_s
  local url="$1" label="$2" timeout="${3:-180}" i=0
  while [ "$(curl -s -o /dev/null -w '%{http_code}' "$url" 2>/dev/null)" != "200" ]; do
    i=$((i+1)); [ "$i" -ge "$timeout" ] && { echo "  ✗ $label not ready after ${timeout}s"; return 1; }
    sleep 1
  done
  echo "  ✓ $label ready (${i}s)"
}

# 1) NIM container (encoder; only needed for `analyze`, but part of the stack)
echo "[1/4] NIM container"
docker compose --profile gpu up -d cosmos-reason2 >/dev/null 2>&1 || true

# 2) waymo_runner API :8000 — USER ADC for *reads* (Waymo source bucket), but a
#    signer KEY for *signing* the dest MP4 URLs (user ADC can't sign v4; the SA
#    can't read the public Waymo bucket — so reads use ADC, signing uses the key).
echo "[2/4] ingest/analysis API :8000"
start api-8000 8000 \
  "GOOGLE_CLOUD_PROJECT=$PROJECT GOOGLE_APPLICATION_CREDENTIALS= VERITY_SIGNER_KEY=$SA_KEY" \
  "exec uvicorn waymo_pipeline.waymo_runner:app --host 0.0.0.0 --port 8000"

# 3) judge_ui :8001 — SA key (reads output bucket + signs URLs)
echo "[3/4] judge API :8001"
start judge-8001 8001 \
  "GOOGLE_CLOUD_PROJECT=$PROJECT GOOGLE_APPLICATION_CREDENTIALS=$SA_KEY JUDGE_BUCKET_URI=$BUCKET JUDGE_PROPOSALS_PATH=$OUTPUT_DIR/scored.json" \
  "exec uvicorn pipeline.modules.judge_ui.server:app --host 0.0.0.0 --port 8001"

# 4) frontend :3000 — same-origin (empty base URLs -> calls go through the proxy)
echo "[4/4] frontend :3000"
start frontend-3000 3000 \
  "NEXT_PUBLIC_API_URL= NEXT_PUBLIC_JUDGE_API_URL=" \
  "cd frontend && exec npm run dev"

echo "== health =="
wait_for "http://localhost:8000/probe-path?uri=gs://x" "api-8000  (:8000)" 60 || true
wait_for "http://localhost:8001/judge/proposals"       "judge     (:8001)" 60 || true
wait_for "http://localhost:3000"                        "frontend  (:3000)" 90 || true
wait_for "http://localhost:3000/judge/proposals"        "proxy     (:3000 -> :8001)" 30 || true
echo "  (NIM :8081 loads for ~3-4 min; analyze waits on it, the UI does not.)"

echo ""
echo "== UP =="
echo "  Open http://localhost:3000  (forward ONLY port 3000)"
echo "  Logs: $LOGDIR/*.log   Stop: scripts/down.sh"
