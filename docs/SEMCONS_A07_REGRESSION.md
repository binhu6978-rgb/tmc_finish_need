# A07 real-training regression evidence

V1 frozen execution commit: `dd583b70859167a557b3583a88e8c87b3d388657`. Server project: `/home/hh/TMC_re_e03_scope_components400_20261019`; Python `/home/hh/miniconda3/envs/hh/bin/python`; physicalGPU0. Full real CIFAR100, original50000-example/100-client partition, original seven models/10 slots/5 epochs/batch50/SGD, deterministic CUDA, Full-only evaluation, no PT.

|Budget|Original A07 reference|A07 with read-only measurement|Exact comparison|
|---|---|---|---|
|2 rounds|Completed|Completed|Pass, no differing fields|
|20 rounds|Completed|Completed|Pass, no differing fields at any round|

Initialization and every round compare all seven parameter-state numerical digests, per-slot post-SGD state through round20, training RNG before/after every slot, round RNG, actual client/profile schedule, Full CE/Accuracy. The observer also asserts its operations preserve main weights, BN buffers and existing gradients. No model tensor/checkpoint is saved. Original-source SHA and configuration SHA are in both manifests and the comparison proof.

Twenty-round original reference wall time259.9110s, observed298.7311s; difference38.8201s (+14.94%) in this simultaneous paired run. Observed includes diagnostic probes and checksum/gradient assertions, so this is the total instrumented run comparison, not isolated Adapter arithmetic latency. Peak allocated memory: reference1,104,749,056bytes; observation1,289,812,992bytes. Peak reserved:1,308,622,848/1,503,657,984bytes. GPU0 also had an existing small training task; GPU1 was occupied by another project and was not used. No process was stopped.

Evidence: `docs/evidence/semcons_a07/regression_v1/semcons_a07_regression_v1_20261022/`, especially `equivalence_2.json`, `equivalence_20.json`, both `regression.jsonl`, `resources.json`, manifests, completion markers and actual stdout logs. The root status records parentPID4183663 and child PIDs.

Linux V1:230 targeted tests passed after restoring missing old Git objects used by four FD regression tests. The initial226-pass/4-failure result was an incomplete historical Git-object environment, not an algorithm failure. Linux V2 adds five statistical separation tests:235 passed. V2 changes observation sampling only; its fresh2/20 paired regression is separately required and is not inferred from the V1 result.

V2 real2/20 regression subsequently **passed**, execution commit `1c7699b` (full SHA in both manifests). It uses the corrected all-physical-pair sampling budget and separate fresh output directories. Its20-round reference took266.4905s, observed321.8865s (+20.79%). Peak allocated memory remains1,104,749,056/1,289,812,992bytes; reserved1,308,622,848/1,505,755,136bytes. Both proof files report exact_match=true and no mismatches. Evidence is in the adjacent `regression_v2` archive, including real Linux test logs. Sampling adjustment was frozen from opportunity counts, not utility values.

Passing regression establishes A07 preservation under this execution environment. It does not validate the semantics or predict Full performance gains. Stage C stays observation-only until the precommitted scientific decision.
