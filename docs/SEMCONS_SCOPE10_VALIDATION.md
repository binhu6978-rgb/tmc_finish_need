# Scope10 validation evidence

All AdapterFL and scope tests: **100 passed** on target Linux hh (11.35s), CUDA hidden. Test log: runs/semcons_scope_preflight/pytest_linux_final.log.
Additional cases verify native M2/S01 full-loop equality and uncovered-tail preservation in S03/S06/S07/S09. First suite had90 passes and one failing history test fixture: duplicate +1/-1 responses averaged to zero, correctly producing no valid history. The fixture was changed to nonzero valid duplicate observations; the neutral fallback behavior was retained.

Ten modes each completed2 real CIFAR-100 CPU smoke rounds. All Full parameters finite, Full-only CSV correct, all ten training RNG and slot schedules equal. These use explicitly labelled50-example/client and200-test-example subsets, not formal performance conditions. No400-round S run or queue was started.

S00/S02 final parameters bitwise equal: True; maximum difference0; training RNG equal: True.

|Mode|Rounds|Gate region-slot fraction r1/r2|Difference from native L2 r1/r2|
|---|---:|---|---|
|s00|2|0/0|0/0|
|s01|2|original M2 diagnostics|original M2 diagnostics|
|s02|2|0/0|0/0|
|s03|2|0.20512821/0.029850746|0.0049276412/0.00014218283|
|s04|2|0/0|0/0|
|s05|2|0/0|0/0|
|s06|2|0/0|0/0|
|s07|2|0/0|0/0|
|s08|2|0/0|0/0|
|s09|2|0/0|0/0|

## Interpretation of the short smoke

Functional measurement covered an average11.3% of class coordinates per slot on these small private subsets. S03 produced nonzero intervention in both rounds. S04-S09 semantic Gates remained neutral in this early two-round subset smoke; no semantic-intervention accuracy benefit or real S07/S09 strength separation is claimed. Positive nonneutral intervention/locality and normalization differences are verified by constructed unit cases instead of forcing Gates active.
S07/S08/S09 accumulated38 and61 unique valid client-region observation events in the two rounds; repeated slots did not create extra history events. S08 class-validity metadata and score distributions were preserved. S07/S09 Gates were equal in the two real smoke rounds, which were neutral. The nonneutral algebra test separately verifies native versus gated denominator behavior.

## Files and provenance

Source hashes in every smoke checkpoint match the frozen training source. The real profile geometry has9 disjoint regions and11,424,356 coordinates. The original models/Fed.py, models/resnet.py, models/Update.py, models/test.py and utils/get_datatset_new.py were not modified.
Evidence: smoke_checks.json, S00_S02_real_smoke_pairing.json, smoke_status.txt, commands_dry_run.txt and per-mode stdout/manifests/CSV/latest/best under runs/semcons_scope_preflight/. No dense private semantic matrices or model weights are committed.

## Ten complete formal commands (dry-run only)

```bash
# Batch 1
CUDA_VISIBLE_DEVICES=0 /home/hh/miniconda3/envs/hh/bin/python -u /home/hh/TMC_re_adapterfl_full_only_400_20261019/main_fed_ori.py --method s00 --method-run-name S00_full_only_400_seed1 --method-output-root /home/hh/TMC_re_adapterfl_full_only_400_20261019/runs/semcons_scope10_full400_seed1/experiments --epochs 400 --seed 1 --gpu 0 --generate_data 0 --data_beta 0.3 --eval-profiles full --conflict-lambda 4.0 > /home/hh/TMC_re_adapterfl_full_only_400_20261019/runs/semcons_scope10_full400_seed1/S00.stdout.log 2>&1
CUDA_VISIBLE_DEVICES=0 /home/hh/miniconda3/envs/hh/bin/python -u /home/hh/TMC_re_adapterfl_full_only_400_20261019/main_fed_ori.py --method s02 --method-run-name S02_full_only_400_seed1 --method-output-root /home/hh/TMC_re_adapterfl_full_only_400_20261019/runs/semcons_scope10_full400_seed1/experiments --epochs 400 --seed 1 --gpu 0 --generate_data 0 --data_beta 0.3 --eval-profiles full --conflict-lambda 4.0 > /home/hh/TMC_re_adapterfl_full_only_400_20261019/runs/semcons_scope10_full400_seed1/S02.stdout.log 2>&1
# Batch 2
CUDA_VISIBLE_DEVICES=0 /home/hh/miniconda3/envs/hh/bin/python -u /home/hh/TMC_re_adapterfl_full_only_400_20261019/main_fed_ori.py --method s01 --method-run-name S01_full_only_400_seed1 --method-output-root /home/hh/TMC_re_adapterfl_full_only_400_20261019/runs/semcons_scope10_full400_seed1/experiments --epochs 400 --seed 1 --gpu 0 --generate_data 0 --data_beta 0.3 --eval-profiles full --conflict-lambda 4.0 > /home/hh/TMC_re_adapterfl_full_only_400_20261019/runs/semcons_scope10_full400_seed1/S01.stdout.log 2>&1
CUDA_VISIBLE_DEVICES=0 /home/hh/miniconda3/envs/hh/bin/python -u /home/hh/TMC_re_adapterfl_full_only_400_20261019/main_fed_ori.py --method s03 --method-run-name S03_full_only_400_seed1 --method-output-root /home/hh/TMC_re_adapterfl_full_only_400_20261019/runs/semcons_scope10_full400_seed1/experiments --epochs 400 --seed 1 --gpu 0 --generate_data 0 --data_beta 0.3 --eval-profiles full --conflict-lambda 4.0 > /home/hh/TMC_re_adapterfl_full_only_400_20261019/runs/semcons_scope10_full400_seed1/S03.stdout.log 2>&1
# Batch 3
CUDA_VISIBLE_DEVICES=0 /home/hh/miniconda3/envs/hh/bin/python -u /home/hh/TMC_re_adapterfl_full_only_400_20261019/main_fed_ori.py --method s04 --method-run-name S04_full_only_400_seed1 --method-output-root /home/hh/TMC_re_adapterfl_full_only_400_20261019/runs/semcons_scope10_full400_seed1/experiments --epochs 400 --seed 1 --gpu 0 --generate_data 0 --data_beta 0.3 --eval-profiles full --conflict-lambda 4.0 > /home/hh/TMC_re_adapterfl_full_only_400_20261019/runs/semcons_scope10_full400_seed1/S04.stdout.log 2>&1
CUDA_VISIBLE_DEVICES=0 /home/hh/miniconda3/envs/hh/bin/python -u /home/hh/TMC_re_adapterfl_full_only_400_20261019/main_fed_ori.py --method s05 --method-run-name S05_full_only_400_seed1 --method-output-root /home/hh/TMC_re_adapterfl_full_only_400_20261019/runs/semcons_scope10_full400_seed1/experiments --epochs 400 --seed 1 --gpu 0 --generate_data 0 --data_beta 0.3 --eval-profiles full --conflict-lambda 4.0 > /home/hh/TMC_re_adapterfl_full_only_400_20261019/runs/semcons_scope10_full400_seed1/S05.stdout.log 2>&1
# Batch 4
CUDA_VISIBLE_DEVICES=0 /home/hh/miniconda3/envs/hh/bin/python -u /home/hh/TMC_re_adapterfl_full_only_400_20261019/main_fed_ori.py --method s06 --method-run-name S06_full_only_400_seed1 --method-output-root /home/hh/TMC_re_adapterfl_full_only_400_20261019/runs/semcons_scope10_full400_seed1/experiments --epochs 400 --seed 1 --gpu 0 --generate_data 0 --data_beta 0.3 --eval-profiles full --conflict-lambda 4.0 > /home/hh/TMC_re_adapterfl_full_only_400_20261019/runs/semcons_scope10_full400_seed1/S06.stdout.log 2>&1
CUDA_VISIBLE_DEVICES=0 /home/hh/miniconda3/envs/hh/bin/python -u /home/hh/TMC_re_adapterfl_full_only_400_20261019/main_fed_ori.py --method s07 --method-run-name S07_full_only_400_seed1 --method-output-root /home/hh/TMC_re_adapterfl_full_only_400_20261019/runs/semcons_scope10_full400_seed1/experiments --epochs 400 --seed 1 --gpu 0 --generate_data 0 --data_beta 0.3 --eval-profiles full --conflict-lambda 4.0 > /home/hh/TMC_re_adapterfl_full_only_400_20261019/runs/semcons_scope10_full400_seed1/S07.stdout.log 2>&1
# Batch 5
CUDA_VISIBLE_DEVICES=0 /home/hh/miniconda3/envs/hh/bin/python -u /home/hh/TMC_re_adapterfl_full_only_400_20261019/main_fed_ori.py --method s08 --method-run-name S08_full_only_400_seed1 --method-output-root /home/hh/TMC_re_adapterfl_full_only_400_20261019/runs/semcons_scope10_full400_seed1/experiments --epochs 400 --seed 1 --gpu 0 --generate_data 0 --data_beta 0.3 --eval-profiles full --conflict-lambda 4.0 > /home/hh/TMC_re_adapterfl_full_only_400_20261019/runs/semcons_scope10_full400_seed1/S08.stdout.log 2>&1
CUDA_VISIBLE_DEVICES=0 /home/hh/miniconda3/envs/hh/bin/python -u /home/hh/TMC_re_adapterfl_full_only_400_20261019/main_fed_ori.py --method s09 --method-run-name S09_full_only_400_seed1 --method-output-root /home/hh/TMC_re_adapterfl_full_only_400_20261019/runs/semcons_scope10_full400_seed1/experiments --epochs 400 --seed 1 --gpu 0 --generate_data 0 --data_beta 0.3 --eval-profiles full --conflict-lambda 4.0 > /home/hh/TMC_re_adapterfl_full_only_400_20261019/runs/semcons_scope10_full400_seed1/S09.stdout.log 2>&1
```
