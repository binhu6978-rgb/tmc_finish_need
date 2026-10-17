# FD-SemConsFL offline mechanism diagnosis

These tools are isolated from the formal trainer. They read the completed A2/A3 100-round seed=1 artifacts from `b546112e482c4dbdeb6ed49744cf9a8ae64452fc`; they do not train, evaluate models, modify A2/A3 or select candidates with official-test performance.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONIOENCODING='utf-8'
$PY = 'D:\software\minicoonda\envs\fd\python.exe'
& $PY -m pytest -q tests/test_risk_diagnostics.py
& $PY -m research.risk_diagnostics --a2 runs/pair100_A2_seed1 --a3 runs/pair100_A3_seed1 --output runs/semconsfl_risk_diagnosis_seed1
& $PY -m research.write_risk_report --output runs/semconsfl_risk_diagnosis_seed1
```

Use a new output directory to repeat the main analysis. Raw inputs and results under `runs/` remain local and ignored by Git; a fresh clone needs those actual experiment artifacts. The report writer asserts that it is rendering the audited seed=1 dataset, rather than silently reusing these specific findings for another run.

The primary replay reconstructs class evidence, old anchors and all Reliability decisions. It verifies both 1000-slot trajectories and final checkpoint memories exactly, and cross-checks the earlier derived slot CSV. Low-support probability rows were not logged, so their confidence telemetry is explicitly excluded from equality; no score-relevant input is excluded.

Important findings, recomputed from raw logs:

- A3: 1000 slots, 99 physical clients, 944 client-rounds and 56 duplicate slots.
- Of 906 calibrated slots, anomaly was strictly below/equal/above its old profile median in 882/0/24 cases. All 882 zero z values correspond to below-median scores; no IQR-floor hits.
- Mean anomaly declined from .12871 in valid rounds1–20 to .01840 in rounds81–100; 257/269 repeated client-profile pairs had negative temporal slopes.
- Among 699 history-supported slots, the `(z>0,r>0)` table is neither=628, r-only=61, z-only=7, both=3. The independent-marginal expected overlap is only .916, so the small overlap is not evidence of negative statistical association.
- Raw drift is not a harmful-update label. Formal histories and final models do not recover discarded per-round uploads. Separate first-round diagnosis tensors do not fix the later-round information gap.

Only three past-only scoring views are used, with no window grid or threshold search:

| View on fixed A3 trajectory | positive calibrated score | positive old persistence | nonneutral shadow Gate |
|---|---:|---:|---:|
| Formal | 24 | 64 | 3 |
| Empirical percentile midrank | 792 | 689 | 612 |
| Profile history restricted to past20 rounds | 189 | 290 | 118 |

Each view maintains its **own** chronological client history. The short profile horizon does not shorten the original physical-client history age/capacity. ECDF rank maps an all-tied pool to .5 and therefore can generate positive risk for statistically ordinary updates: activity is not validity. Shadows never feed back into the logged training trajectory.

The output includes all-slot CSVs, group funnels by round/profile/client/sample range, per-round old pool age/median/IQR, repeated-client indicators, physical-client associations, and old-z provenance. Temporal slopes use deduplicated client-profile-round units with physical-client bootstrap clusters; this is not a new FL seed experiment or a causal/performance confidence interval.

Local NumPy `lstsq` exits code1 on a 3x2 toy input. The analysis uses equivalent alternating group projections for additive round/profile fixed-effect residuals, validates the normal equations, and tests balanced/unbalanced cases. No environment or training code was changed to work around this analysis-only issue.

See [the unexecuted fixed-upload protocol](../docs/FD_SEMCONSFL_UPDATE_UTILITY_PROTOCOL.md). A minimal future pilot can collect a single original next dispatch from the saved final Full/Memory/RNG rather than rerunning100 rounds. It creates new conditional uploads, not recovered old uploads, and requires explicit user authorization. Probe indices are frozen from original training data without creating a held-out validation partition; no probe forward or SGD has been executed by these tools.
