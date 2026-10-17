# A3: remove only true-class confidence protection

`A0` remains native AdaptiveFL, `A1` remains observer-only, and `A2` remains the formal FD-SemConsFL definition at `8aeb65a4af5d6320b8db7d6046db5a8b4d18c417`.

`A3` is an explicitly separate research arm. Its only scoring change is
`candidate_c = max(0, signed_drift_c)`, regardless of the true-class confidence change.
The default `SemanticMemory.score(..., confidence_guard=True)` retains the formal A2 definition. Only the A3 trainer path passes `confidence_guard=False`.

Measurements, class validity/support, sample-count weighting, old anchors and EMA, old same-profile median/IQR, clipped z, old physical-client reliability, risk=z*r, gate_strength=.25, coordinate coverage denominator, and original training/evaluation remain unchanged. Round zero still forces Gate=1 and excludes memory writes. A3 is identified in its manifest and checkpoint mode; it cannot resume an A2 checkpoint.

A3 diagnostics report `confidence_protected=0` because the guard is disabled. `confidence_would_protect` records how many positive-drift class pairs the original A2 guard would have protected. A0/A1/A2 diagnostic output is unchanged.

Existing synthetic tests plus reference-commit regression tests cover the default scoring definition and the A0/A1/A2 round trajectories, state, memory, evidence, diagnostics and RNG. A3 integration tests check positive drift with improving confidence, unchanged history-based Gate and unchanged aggregation.

An offline precheck can be run without training:

```powershell
python -m experiments.replay_confidence_ablation --input runs/pair40_A2_seed1 --output runs/four100_offline_A3_seed1
```

The replay recomputes A3 anomalies from logged valid class drifts and sample counts, with its own strictly past Reliability Memory. It verifies the original A2 decisions exactly. The result predicts Gates only on the fixed A2 signatures/anchors/updates. It is not a closed-loop A3 accuracy estimate.

The completed seed=1 40-round replay produced 380/380 positive valid anomalies for A3 (A2: 8/380), but only 1/400 nonneutral Gate, first at round 18; minimum gamma=.98843303. Full had 0/58 nonneutral Gates. This is sufficient to distinguish A3 from a mathematically identical baseline, but the signal is sparse and does not establish effectiveness.

The authorized next stage is four new 100-round seed=1 runs: A0/A1 first, A2/A3 second, at most two concurrent GPU processes. Controlled actual-client ResNet diagnosis precedes formal training. Deterministic CUDA flags used for diagnosis must not enter formal runs. No 40-round checkpoint extension, tuning, fifth arm, or automatic 1000-round run is authorized.

The actual-data diagnosis ran the full original first-round 10-slot schedule, all original client examples, all five local epochs, and the actual seven ResNet-18 profiles, in sequential independent processes. A0 repeated without any observer first differed at slot 1 (zero-based), profile 3, SGD step 0, in gradients; initialization, incoming parameters, forward outputs, RNG and training sample order matched. Its aggregated Full maximum absolute difference was .0025087893 (relative L2 .0041675435). A0 versus A1 first differed at the same backward step; Full maximum absolute difference was .0026189594 (relative L2 .0041673914).

All 20 pre/post measurements preserved model state, buffers, gradients, module training flags and Python/NumPy/CPU/CUDA RNG. With diagnostic-only `CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic algorithms and cuDNN deterministic enabled, A0/A1 matched bitwise at every recorded forward, gradient, optimizer step, upload and final aggregation, with identical RNG/sample order. These results support CUDA backward numerical nondeterminism rather than a deterministic observer state mutation in the tested real round. They do not identify an individual CUDA kernel or guarantee identical independent-process formal trajectories. Formal runs retain the original nondeterministic backend settings.

Detailed traces and tensor comparisons are retained locally under `runs/four100_control_20261017/diagnosis`; the independent diagnostic script is adjacent. Regression tests: 55 passed. The skill search CLI could not execute within the local sandbox (Node EPERM); no extra package or skill was installed and no training recipe was changed.

Report official-test peaks as **test-oracle peak**, with the final round and a fixed last-20-round mean. A3 beating A2 or A0 would not by itself prove that semantic risk identifies harmful updates. Matched-norm scaling and permuted-Gate controls require a later authorization.
