# Paired500 launch audit (not a result report)

- Frozen training commit: `49b5b5f86fe61815aa47b09445b27a0c6db2fbee`.
- Branch: `feature/semconsfl-semantic-a07-v2`.
- Existing server project: `/home/hh/TMC_re_e03_scope_components400_20261019`.
- New output only: `runs/semantic_a07_pair500_seed1_20261022/`.
- Suite PID:216037; A07-only PID:216727; V2 PID:216728 (startup identities, status.json is authoritative).
- Physical GPU0, two processes; occupied GPU1 left alone.
- Local tests:289 passed in19.02s; Linux:289 passed in15.87s.
- True complete-data plain-A07 vs diagnosed-A07:2 rounds each, exact seven-model/slot/SGD RNG/Full-test regression passed.
- Diagnosed pilot duration27.281917s, peak allocated1292426240bytes, reserved1409286144bytes. Its complete recovery checkpoint is53MiB. These are preflight costs, not formal training costs.
- Both formal500 processes were started only after the exact pilot passed. This document does not claim they have completed.
- All V2 scoring, lambda0.2, history, original full50000 training partition and SGD remain frozen. CRD omitted by explicit user instruction.
- Main A07-only is measurement-observed lambda0: main-model updates exactly use frozen A07. The observer collects shadow U and V2 counterfactuals, without affecting SGD. Adapter remains auxiliary.
- Utility probes exclude semantic-scoring samples and same-identity senders, but remain SGD-training-side. They are not an independent validation set.

## Actual launcher

```bash
nohup /home/hh/miniconda3/envs/hh/bin/python -u tools/run_semantic_pair500_suite.py \
  --execute --root runs/semantic_a07_pair500_seed1_20261022 \
  --dataset-root /home/hh/ICML_finish/data \
  --prepared runs/capability_final40_400_seed1_20261021/prepared \
  > runs/semantic_a07_pair500_seed1_20261022_launcher.log 2>&1 < /dev/null &
```

The suite owns the GPU0 mapping and two-process limit. Do not run this command a second time against the existing directory. The success marker requires two real500-row results and paired schedule/RNG audit; report generation is automatic afterward. Failure keeps original logs and does not retry. One rolling recovery checkpoint per trajectory is written every50 rounds and at the final round; no test-oracle-best model is saved.

## Outputs, once actually complete

`neutral/training/a07/neutral_500_seed1/` and `active/training/a07/active_500_seed1/`: Full CSV, method diagnostics, per-round source weights, both memories, regression/RNG trace, fixed-window IV-E records, manifests and recovery checkpoint. The root `report/` contains two-trajectory performance comparison, Figure7/8 PNG/PDF, rawCSV and grouped mechanism statistics. Old evidence and unstarted120-round code remain intact.

Score association, same-input aggregation utility, and independent whole-trajectory performance must be assessed separately. No positive conclusion is made at launch. Completion analysis must retain adverse/neutral observations, report true source-round units, inspect PDFs, and push lightweight scalar evidence only, never datasets, images, credentials or checkpoints.
