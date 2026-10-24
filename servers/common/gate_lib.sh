# SPDX-License-Identifier: Apache-2.0
# Sourced by servers/*/tests/hpu/run_gates.sh: start a server on this card unless --url names a running one.
#   run_gates.sh [--url http://host:port] [gate options...]
# Without --url the server starts here with the environment you give it, its log goes to $GATE_LOG
# (default /tmp/<server>-gates.log), and it is stopped when the gates finish.

COMMON=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$COMMON${PYTHONPATH:+:$PYTHONPATH}"
URL=""
GATE_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --url) URL="$2"; shift 2 ;;
    *) GATE_ARGS+=("$1"); shift ;;
  esac
done

# start_server <server dir> <startup budget in seconds> [command...]: sets URL, stops the server on exit.
start_server() {
  local dir budget=$2
  dir=$(cd "$1" && pwd); shift 2
  [ -n "$URL" ] && return 0
  local port=${PORT:-8000}
  GATE_LOG=${GATE_LOG:-/tmp/$(basename "$dir")-gates.log}
  (cd "$dir" && exec "$@") >"$GATE_LOG" 2>&1 &
  SERVER_PID=$!
  trap 'kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null' EXIT
  URL="http://127.0.0.1:$port"
  local t0=$SECONDS
  until curl -sf "$URL/health" >/dev/null; do
    if ! kill -0 $SERVER_PID 2>/dev/null; then echo "FAIL startup: server exited, see $GATE_LOG"; tail -20 "$GATE_LOG"; exit 1; fi
    if [ $((SECONDS - t0)) -gt "$budget" ]; then echo "FAIL startup: not healthy after ${budget}s, see $GATE_LOG"; exit 1; fi
    sleep 5
  done
  echo "PASS startup: healthy after $((SECONDS - t0))s (log $GATE_LOG)"
}
