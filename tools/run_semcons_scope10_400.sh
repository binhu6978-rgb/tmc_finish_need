#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash tools/run_full_only_sweep.sh --config configs/semcons_scope10_400.json "$@"
