# G01–G10 completed scalar evidence

- `raw/GXX/`: original 400-round CSV, diagnostics, manifest, run environment, resources, summary and completion marker, copied from the existing server run directories. No model weights or CIFAR images are included.
- `baseline/`: reused completed deterministic B00-D scalar records; the baseline was not rerun.
- `status.json`: terminal queue state, ten completed, zero failed.
- `checkpoint_verification.json`: direct server inspection of latest/best checkpoints, finite Full parameters, actual 400-round completion and identical training source fingerprints.
- `launch_evidence.json`, `linux93_tests.log`: startup evidence and original Linux correctness tests, not performance results.
- `report/`: original automatic end-of-suite report, all-round CSVs and plots.
- `analysis/`: final read-only analysis, reconstructed source shares, geometry, coverage, fixed contrasts and costs. G08/G10 region gains in the original diagnostics are **virtual**; the applied intervention is `global_scale`.
- `report_tests.log`: 31 final evidence/geometry, aggregation and queue tests passed.

The legacy `formal_300_success` field in original `completed.json` is false because these are 400-round runs. Success is determined by `success=true`, `rounds_completed=400`, complete per-round logs and actual latest checkpoint validation, not that historical 300-round field.

Frozen training commit: `36b84b858b6a03c038ad8a142ededf870c5b5ce6`.
Full Chinese interpretation: [NATIVE_GROUPS10_RESULTS.md](../../NATIVE_GROUPS10_RESULTS.md).
All peaks are official **test-oracle peaks** in single-seed exploration.
