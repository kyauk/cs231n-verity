#!/usr/bin/env bash
# Stop the Verity app servers started by up.sh (frontend + the two APIs).
# Leaves the NIM container running by default (it takes ~3-4 min to reload);
# pass --all to stop it too.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="$ROOT/.run"

for name in frontend-3000 judge-8001 api-8000; do
  pidf="$LOGDIR/$name.pid"
  if [ -s "$pidf" ]; then
    pid="$(cat "$pidf")"
    # kill the whole process group (setsid put each service in its own)
    if kill -0 "$pid" 2>/dev/null; then
      kill -- -"$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
      echo "stopped $name (pid $pid)"
    fi
    rm -f "$pidf"
  fi
done
# belt-and-suspenders: free the ports if anything lingers
for port in 3000 8000 8001; do
  fuser -k "${port}/tcp" 2>/dev/null && echo "freed :$port" || true
done

if [ "${1:-}" = "--all" ]; then
  (cd "$ROOT" && docker compose stop cosmos-reason2 >/dev/null 2>&1) && echo "stopped NIM container"
fi
echo "down."
