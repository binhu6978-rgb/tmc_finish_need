#!/usr/bin/env bash
# Exactly C1 then C2, GPU0, fixed protocol, failure stops; no automatic retry.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}
DATA_ROOT=${DATA_ROOT:-/home/hh/ICML_finish/data}
OUT=runs/e03_scope_components400_seed1
mkdir "$OUT"
export CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1
nvidia-smi > "$OUT/gpu_before.txt"
status() {
  "$PYTHON" - "$OUT/status.json" "$1" "$2" "$3" <<'PY'
import json,sys,time
from pathlib import Path
p=Path(sys.argv[1]); data=json.loads(p.read_text()) if p.exists() else {'order':['c1','c2'],'completed':[]}
data.update(phase=sys.argv[2],method=sys.argv[3],pid=int(sys.argv[4]),updated=time.time())
if sys.argv[2]=='completed_method':data['completed'].append(sys.argv[3])
t=p.with_suffix('.pending');t.write_text(json.dumps(data,indent=2));t.replace(p)
PY
}
for method in c1 c2; do
  name="${method^^}_full_only_400_seed1"
  "$PYTHON" -u tools/run_evidence_candidate.py --dataset-root "$DATA_ROOT" \
    --method "$method" --epochs 400 --seed 1 --gpu 0 --generate_data 0 --data_beta .3 \
    --eval-profiles full --conflict-lambda 4 --method-run-name "$name" --method-output-root "$OUT" \
    > "$OUT/${method}.stdout.log" 2> "$OUT/${method}.stderr.log" &
  pid=$!
  status running "$method" "$pid"
  if wait "$pid"; then
    if ! "$PYTHON" - "$OUT/$method/$name" <<'PY'
import json,sys,csv
from pathlib import Path
import torch
p=Path(sys.argv[1]);c=json.loads((p/'completed.json').read_text());s=torch.load(p/'latest.pt',map_location='cpu',weights_only=False)
with (p/'accuracy_each_round.csv').open() as f:r=list(csv.DictReader(f))
assert c['success'] and c['rounds_completed']==c['requested_epochs']==s['next_round']==len(s['rows'])==len(r)==400
assert [int(x['round']) for x in r]==list(range(1,401))
PY
    then
      status failed_validation "$method" "$pid";exit 1
    fi
    status completed_method "$method" "$pid"
  else
    status failed "$method" "$pid";exit 1
  fi
done
status completed_all c2 0
