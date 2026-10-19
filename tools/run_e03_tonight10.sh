#!/usr/bin/env bash
# One GPU0 child at a time: authorized C2 restart, then ten fixed hypotheses.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}
DATA_ROOT=${DATA_ROOT:-/home/hh/ICML_finish/data}
OUT=runs/e03_tonight10_400_seed1
ORDER=(c2 t01 t04 t05 t06 t09 t02 t03 t07 t08 t10)
mode=${1:---dry-run}
if [[ "$mode" != --dry-run && "$mode" != --execute ]]; then echo 'Use --dry-run or --execute' >&2;exit 2;fi
command_for() {
  local method=$1 name="${1^^}_full_only_400_seed1"
  if [[ "$method" == c2 ]];then name=C2_full_only_400_seed1_after_update;fi
  CMD=("$PYTHON" -u tools/run_evidence_candidate.py --dataset-root "$DATA_ROOT"
    --method "$method" --epochs 400 --seed 1 --gpu 0 --generate_data 0 --data_beta .3
    --eval-profiles full --conflict-lambda 4 --method-run-name "$name" --method-output-root "$OUT")
  RUN="$OUT/$method/$name"
}
if [[ "$mode" == --dry-run ]];then
  for method in "${ORDER[@]}";do command_for "$method";printf 'CUDA_VISIBLE_DEVICES=0 ';printf '%q ' "${CMD[@]}";printf '\n';done
  exit 0
fi
# Never overwrite experiments; no edits/retries of a failed run.
mkdir "$OUT"
export CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1
nvidia-smi > "$OUT/gpu_before.txt"
status() {
  "$PYTHON" - "$OUT/status.json" "$1" "$2" "$3" "${CMD[@]}" <<'PY'
import json,sys,time
from pathlib import Path
p=Path(sys.argv[1]);d=json.loads(p.read_text()) if p.exists() else {'completed':[],'failed':[],'order':['c2','t01','t04','t05','t06','t09','t02','t03','t07','t08','t10']}
d.update(phase=sys.argv[2],method=sys.argv[3],pid=int(sys.argv[4]),argv=sys.argv[5:],updated=time.time())
if sys.argv[2] in ('completed_method','failed_method'):d['completed' if sys.argv[2]=='completed_method' else 'failed'].append(sys.argv[3])
t=p.with_suffix('.pending');t.write_text(json.dumps(d,indent=2));t.replace(p)
PY
}
for method in "${ORDER[@]}";do
  command_for "$method"
  "${CMD[@]}" > "$OUT/$method.stdout.log" 2> "$OUT/$method.stderr.log" &
  pid=$!;status running "$method" "$pid"
  if wait "$pid";then
    if "$PYTHON" tools/report_tonight10.py --validate-run "$RUN" >> "$OUT/completion_validation.log" 2>&1;then
      status completed_method "$method" "$pid"
    else status failed_method "$method" "$pid";fi
  else status failed_method "$method" "$pid";fi
done
status training_finished t10 0
if "$PYTHON" tools/report_tonight10.py --root "$OUT" --references runs/historical \
  --c1 runs/e03_scope_components400_seed1/c1/C1_full_only_400_seed1 \
  > "$OUT/report_generator.log" 2>&1;then
  status finished t10 0
else status report_failed t10 0;exit 1;fi
