#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash tools/run_full_only_sweep.sh --config configs/adapterfl_next10_m2_m3_400.json "$@"
