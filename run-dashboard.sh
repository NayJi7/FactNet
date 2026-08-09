#!/usr/bin/env bash
# Start the engine and the interface together, and stop both on Ctrl-C.
#
#   ./run-dashboard.sh
#
# The Bluesky credentials are optional: without them the live-URL mode is
# unavailable and everything else, including the pre-loaded cascades, still
# works.
set -euo pipefail
# Job control is deliberately left off, so the children stay in this script's
# process group and a real Ctrl-C in a terminal reaches them directly. The trap
# below covers the other cases: the script signalled on its own, or exiting for
# any other reason.
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"

API_PORT=8000
WEB_PORT=5173

[ -f "$HOME/.bsky/identifier" ]   && export BSKY_IDENTIFIER="$(cat "$HOME/.bsky/identifier")"
[ -f "$HOME/.bsky/app_password" ] && export BSKY_APP_PASSWORD="$(cat "$HOME/.bsky/app_password")"

if [ ! -d models/roberta-liar ] || [ ! -d models/graph ]; then
  echo "Models are missing. Build the local ones with:"
  echo "  uv run python -m factnet.serve.build"
  echo "The fine-tuned transformers come from the Kaggle run (see notebooks/)."
  exit 1
fi

listeners() { lsof -t -i ":$API_PORT" -i ":$WEB_PORT" -sTCP:LISTEN 2>/dev/null || true; }

# `uv run` and `bun run` are launchers: the process holding the port is a
# grandchild, so the whole subtree has to be collected. Signalling only the pid
# we started is what used to leave a listener behind.
descendants() {
  local pid=$1 child
  for child in $(pgrep -P "$pid" 2>/dev/null || true); do
    descendants "$child"
    printf '%s ' "$child"
  done
}

api_pid=""
web_pid=""
stopped=0

cleanup() {
  [ "$stopped" = 1 ] && return 0
  stopped=1
  local targets=""
  for pid in "$api_pid" "$web_pid"; do
    [ -n "$pid" ] || continue
    targets="$targets $(descendants "$pid") $pid"
  done
  # shellcheck disable=SC2086
  [ -n "${targets// /}" ] && kill -TERM $targets 2>/dev/null || true

  for _ in $(seq 1 20); do
    [ -z "$(listeners)" ] && return 0
    sleep 0.25
  done
  local left
  left=$(listeners)                       # still holding the port after five seconds
  # shellcheck disable=SC2086
  [ -n "$left" ] && kill -KILL $left 2>/dev/null || true
  return 0
}
trap cleanup EXIT INT TERM

# A previous run can leave a listener behind. Clear the ports rather than
# failing with "address already in use".
stale=$(listeners)
if [ -n "$stale" ]; then
  echo "releasing ports held by a previous run"
  # shellcheck disable=SC2086
  kill -TERM $stale 2>/dev/null || true
  for _ in $(seq 1 20); do [ -z "$(listeners)" ] && break; sleep 0.25; done
  stale=$(listeners)
  # shellcheck disable=SC2086
  [ -n "$stale" ] && kill -KILL $stale 2>/dev/null || true
  sleep 0.5
fi

uv run uvicorn factnet.serve.api:app \
  --host 127.0.0.1 --port "$API_PORT" --log-level warning &
api_pid=$!

# wait for the engine rather than racing it: the first model load takes a moment
for _ in $(seq 1 60); do
  curl -sf --max-time 2 "http://127.0.0.1:$API_PORT/api/health" >/dev/null && break
  sleep 0.5
done
if ! curl -sf --max-time 2 "http://127.0.0.1:$API_PORT/api/health" >/dev/null; then
  echo "the engine did not start"
  exit 1
fi
echo "engine    http://127.0.0.1:$API_PORT"

( cd web && exec bun run dev --host 127.0.0.1 --port "$WEB_PORT" >/dev/null 2>&1 ) &
web_pid=$!
echo "interface http://127.0.0.1:$WEB_PORT"
echo "Ctrl-C stops both."
wait
