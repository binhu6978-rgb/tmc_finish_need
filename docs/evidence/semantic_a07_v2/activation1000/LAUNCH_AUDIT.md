# One V2 seed1 run:1000 rounds (launch record)

This is the user's newly authorized mechanism-operation observation. It does not replace or reinterpret the completed paired500 evidence.

- Frozen source commit:`149c6713c339204ad23fe43952350ae2832b3cd0`.
- Server project:`/home/hh/TMC_re_e03_scope_components400_20261019`.
- Output:`runs/semantic_a07_v2_activation1000_seed1_20261023/`.
- Suite PID366263; formal worker PID366572 (status.json is authoritative).
- GPU0, one independent process. GPU1 is occupied and was not used.
- Source/Memory/scoring/aggregation/model/SGD/data are unchanged; fixed lambda0.2,seed1.
- Fresh1000-round run; no500-checkpoint identity migration or overwrite.
- Local295 tests passed in18.64s; Linux295 passed in15.98s.
- Real2-round complete-data smoke matched old V2's seven models, slots, local SGD/RNG and official Full evaluations exactly.
- Formal worker started after smoke_pass.json success. This launch record does not claim1000-round completion.
- Initial GPU0 free23473MiB; account usage69981MiB of100GiB; ample headroom for one rolling complete checkpoint and scalar logs.

## Actual command

```bash
nohup /home/hh/miniconda3/envs/hh/bin/python -u tools/run_semantic_activation1000.py \
  --execute --root runs/semantic_a07_v2_activation1000_seed1_20261023 \
  --dataset-root /home/hh/ICML_finish/data \
  --prepared runs/capability_final40_400_seed1_20261021/prepared \
  --reference500 runs/semantic_a07_pair500_seed1_20261022 \
  > runs/semantic_a07_v2_activation1000_seed1_20261023_launcher.log 2>&1 < /dev/null &
```

Do not repeat the command against an existing output. The reference500 path supplies the short exact-regression evidence, not a second trained comparator. One formal active run only. All-round Full Accuracy/CE and mechanisms are saved. Counterfactual read-only diagnostics are fixed at53 rounds; all outcomes retained. Every50 rounds and the final round update one complete rolling recovery checkpoint; no test-selected best model. No automatic retries or recovery.

On genuine completion, `active/activation_report/` will contain the new activation-only Figure7/8, raw scalar CSVs and a scoped report. Their role is to document evidence flow and actual source-share/parameter intervention, not demonstrate semantic utility or comparative performance. Original500 figures and conclusions remain intact. Completion-only follow-up is scheduled; no live per-round reporting or extra experiment is authorized.
