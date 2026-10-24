#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Device gates for the Muse-Glimmer-30B port. Start the server with the recipe's Launch block first
# (port image, fixes applied), then run this from anywhere that reaches it. One PASS or FAIL line per
# gate; the exit code is the number of failed gates.
#
#   ports/muse-glimmer-30b/tests/hpu/run_gates.sh [--base-url http://127.0.0.1:8000] [--quick]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$HERE/gates.py" "$@"
