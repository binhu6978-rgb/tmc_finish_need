#!/usr/bin/env bash
# Bash >=4.3; fixed six-arm 300-round study. No retries or resume.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON="${PYTHON:-python3}"
DATA_ROOT="${DATA_ROOT:-$PWD/data/cifar100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/runs/hgkr7_300_seed1_$(date +%Y%m%d_%H%M%S)}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Refuse existing output, preserving failed and completed studies alike.
mkdir -p -- "$(dirname -- "$OUTPUT_ROOT")"
mkdir -- "$OUTPUT_ROOT"
pid1=""; pid2=""
cleanup() {
  local code=$?
  trap - EXIT INT TERM
  if [[ -n "$pid1" ]]; then kill -TERM "$pid1" 2>/dev/null || true; wait "$pid1" 2>/dev/null || true; fi
  if [[ -n "$pid2" ]]; then kill -TERM "$pid2" 2>/dev/null || true; wait "$pid2" 2>/dev/null || true; fi
  if (( code != 0 )); then
    printf '%s\n' '{"failed":true,"message":"Stopped; retained logs. No retries or later batches."}' > "$OUTPUT_ROOT/FAILED.json"
    printf 'Run failed. Inspect logs in %s\n' "$OUTPUT_ROOT" >&2
  fi
  exit "$code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Direct training on GPU 0; no package/GPU/memory preflight or preparation job.
gpu1=0; gpu2=0
printf 'Batch GPU assignment: %s / %s\n' "$gpu1" "$gpu2" | tee "$OUTPUT_ROOT/gpu_assignment.log"

finish_pair() {
  # Fail immediately when either child fails, rather than waiting for a long peer.
  if ! wait -n "$pid1" "$pid2"; then return 1; fi
  wait "$pid1"; wait "$pid2"
  pid1=""; pid2=""
}

run_pair() {
  local left="$1" right="$2"
  printf 'Starting %s + %s (300 rounds each)\n' "$left" "$right"
  "$PYTHON" -m experiments.run_hgkr7 --mode "$left" --config "configs/hgkr7/$left.json" --gpu "$gpu1" --dataset-root "$DATA_ROOT" --output-dir "$OUTPUT_ROOT/$left" > "$OUTPUT_ROOT/$left.stdout.log" 2> "$OUTPUT_ROOT/$left.stderr.log" &
  pid1=$!
  "$PYTHON" -m experiments.run_hgkr7 --mode "$right" --config "configs/hgkr7/$right.json" --gpu "$gpu2" --dataset-root "$DATA_ROOT" --output-dir "$OUTPUT_ROOT/$right" > "$OUTPUT_ROOT/$right.stdout.log" 2> "$OUTPUT_ROOT/$right.stderr.log" &
  pid2=$!
  finish_pair
}
run_pair C0 C1
run_pair C3 C4
run_pair C2 C5
"$PYTHON" -m experiments.summarize_hgkr7 "$OUTPUT_ROOT" > "$OUTPUT_ROOT/summary.stdout.log" 2> "$OUTPUT_ROOT/summary.stderr.log"
printf 'All six runs verified at 300 rounds. Report: %s/report.md\n' "$OUTPUT_ROOT"
