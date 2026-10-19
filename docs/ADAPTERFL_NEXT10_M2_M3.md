# AdapterFL next10 M2/M3 exploratory sweep

Base: 89351d7. Developed directly in `/home/hh/TMC_re_adapterfl_full_only_400_20261019` after verifying all previous B00/E01-E09 experiments completed 400 rounds. Their results/checkpoints remain unchanged. No new formal training or queue was started for this delivery; review is required before execution.

## Fixed settings

All runs: 400 rounds, seed=1, CIFAR-100, the existing Dirichlet alpha=.3 partition, seven original ResNets, original sampling/SGD/local settings, `--eval-profiles full` with existing RNG isolation. No method combinations or formula changes. Old B00 and E03 are historical comparisons, not rerun or overwritten.

|ID|Method|Parameters|
|---|---|---|
|N01|m2|conflict_lambda=3.0|
|N02|m2|conflict_lambda=3.5|
|N03|m2|conflict_lambda=4.5|
|N04|m2|conflict_lambda=5.0|
|N05|m2|conflict_lambda=6.0|
|N06|m2|conflict_lambda=8.0|
|N07|m3|shared=1.0, remainder=.92|
|N08|m3|shared=1.0, remainder=.98|
|N09|m3|shared=.98, remainder=.95|
|N10|m3|shared=1.02, remainder=.95|

Batches: N03+N04, N01+N02, N05+N06, N07+N08, N09+N10. Both jobs must complete 400 rounds before the next batch. Failure stops later scheduling and preserves errors/checkpoints; no retries, early stopping, parameter changes or test-based reordering.

## Commands (not executed during preparation)

Use the existing hh environment and project directory:

```bash
cd /home/hh/TMC_re_adapterfl_full_only_400_20261019
conda activate hh
PYTHON=/home/hh/miniconda3/envs/hh/bin/python bash tools/run_next10_m2_m3_400.sh --dry-run --gpus 0,0
```

After GitHub review and explicit execution approval, the complete command is:

```bash
cd /home/hh/TMC_re_adapterfl_full_only_400_20261019
PYTHON=/home/hh/miniconda3/envs/hh/bin/python bash tools/run_next10_m2_m3_400.sh --execute --gpus 0,0 --allow-shared-gpu
```

This uses physical GPU0 for both processes, each with internal `--gpu 0`. At preparation GPU0 was idle (23,994 MiB free); GPU1 had an unrelated job using about22GB and was not touched. Prior two-process GPU0 runs completed successfully. The existing launcher rechecks nvidia-smi before each batch, refuses other compute occupants or insufficient free memory, and never kills existing jobs or changes batch size. Recheck resources when execution is approved. No real-time per-round monitoring is installed.

The wrapper reuses `tools/run_full_only_sweep.sh` and the existing Python queue, selecting `configs/adapterfl_next10_m2_m3_400.json` via the added optional `--config`. The previous default configuration remains unchanged. Actual dry-run commands are preserved in `runs/next10_preflight/commands_dry_run.txt`.

## Output and reports

Default root: `runs/adapterfl_next10_m2_m3_400_seed1/`. It does not exist after dry-run. Each job uses `experiments/<method>/<ID>_full_only_400_seed1/`; stdout/stderr are merged into `<ID>.stdout.log`. Existing root directories are rejected before launching; no old results are reused as new results.

Per-round CSV now also stores `best_full_accuracy` and `best_round`, preserving the earliest round for ties. Existing Full Accuracy/CE/timing, M2 conflict weights/negative fractions or M3 region diagnostics, `summary.json`, `best_full.pt` and `latest.pt` remain available. This logging addition does not draw RNG or change updates. New training source fingerprints differ, so do not resume old-source checkpoints with these files.

After all ten finish, the offline reporter creates:

- `comparison.csv`: peak/round, final accuracy, last20/last50 means, final CE, CE at the peak and the mean CE within peak +/-2 rounds (clipped to1..400), and differences from historical B00/E03 peaks in percentage points.
- `report.md`: comparison table and scientific interpretation.
- `full_accuracy_comparison.png`: ten new curves and historical B00/E03 curves.
- `accuracy_curves/N01.png` through `N10.png`: individual Full curves.
- `full_accuracy_all_rounds.csv`: raw values for all400 rounds and ten runs.
- `reference_metrics.json`: metrics read from the preserved B00/E03 CSVs (verified peaks36.40% and37.29%).

These are official-test **test-oracle peaks**. The next10 settings were chosen after reviewing previous official-test outcomes; a higher next10 peak is further exploration on one seed, not an independent validation-selected or unbiased performance claim. The reporter does not tune parameters or retrain anything.

## Checks and changed files

Linux hh, CPU with CUDA hidden: **68 passed in9.84s**. Existing65 checks plus next10 hook/parameter/path checks, dry-run protection and a synthetic-data-only offline report test. The synthetic report fixture checks ten PNGs, the comparison plot,400-row export, CE window boundaries, raw historical references and reference-file preservation; it is not a new CIFAR performance experiment. Existing peak tests additionally verify per-round cumulative peak columns and earliest tied round. Log: `runs/next10_preflight/pytest_linux.log`.

Changed: `adapterfl_methods.py` (two cumulative-peak CSV fields), `tools/run_full_only_sweep.py` (optional config/report dispatch), `tests/test_adapterfl_methods.py` (peak fields). Added: fixed next10 JSON, wrapper shell, offline reporter, next10 tests and this document. Original model, aggregation, sampler, data partition, local training and Full-only/RNG evaluation implementation are unchanged.
