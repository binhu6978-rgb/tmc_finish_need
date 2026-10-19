#!/usr/bin/env bash
# Six bounded full-data diagnostics, never the S00-S09 formal400 queue.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-/home/hh/miniconda3/envs/hh/bin/python}
ROOT=${PREFLIGHT_ROOT:-runs/semcons_activation_full10_seed1}
if [[ ${1:-} != --execute || ${2:-} != --allow-shared-gpu0 ]]; then
  echo "Explicit invocation: bash tools/run_semcons_activation_preflight.sh --execute --allow-shared-gpu0"
  for mode in s03 s04 s06 s07 s08 s09; do
    printf 'CUDA_VISIBLE_DEVICES=0 %q -u tools/preflight_semcons_activation.py --mode %s --rounds 10 --gpu 0 --output-root %q\n' "$PYTHON" "$mode" "$ROOT"
  done
  exit 0
fi
mkdir -p "$ROOT"
# Preserve test evidence in root; never reuse a previous training invocation.
mkdir "$ROOT/launch_lock"
exec > >(tee -a "$ROOT/launch.log") 2>&1
trap 'echo "FAILED (line $LINENO): $(date -Is)" > "$ROOT/status.txt"' ERR
echo "RUNNING: $(date -Is) parent=$$" > "$ROOT/status.txt"
printf 'mode\tpid\texit_code\n' > "$ROOT/processes.tsv"
for pair in 's03 s04' 's06 s07' 's08 s09'; do
  nvidia-smi > "$ROOT/gpu_before_${pair// /_}.txt"
  free=$(( $(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits) ))
  if (( free < 12000 )); then
    echo "Insufficient GPU0 free memory for two unchanged models: ${free} MiB" >&2
    exit 1
  fi
  read -r a b <<< "$pair"
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u tools/preflight_semcons_activation.py --mode "$a" --rounds 10 --gpu 0 --output-root "$ROOT" > "$ROOT/$a.stdout.log" 2> "$ROOT/$a.stderr.log" &
  pa=$!
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u tools/preflight_semcons_activation.py --mode "$b" --rounds 10 --gpu 0 --output-root "$ROOT" > "$ROOT/$b.stdout.log" 2> "$ROOT/$b.stderr.log" &
  pb=$!
  printf '%s\t%s\tRUNNING\n%s\t%s\tRUNNING\n' "$a" "$pa" "$b" "$pb" >> "$ROOT/processes.tsv"
  ea=0; eb=0
  wait "$pa" || ea=$?
  wait "$pb" || eb=$?
  printf '%s\t%s\t%s\n%s\t%s\t%s\n' "$a" "$pa" "$ea" "$b" "$pb" "$eb" >> "$ROOT/processes.tsv"
  nvidia-smi > "$ROOT/gpu_after_${pair// /_}.txt"
  if (( ea != 0 || eb != 0 )); then
    echo "FAILED pair=$pair exit=$ea,$eb; no retry or subsequent batch" > "$ROOT/status.txt"
    exit 1
  fi
done
"$PYTHON" tools/report_semcons_activation.py "$ROOT" > "$ROOT/activation_report.stdout.json"
echo "COMPLETED_SHORT_PREFLIGHT: $(date -Is)" > "$ROOT/status.txt"
