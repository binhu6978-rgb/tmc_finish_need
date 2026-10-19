# SemConsFL scope10: frozen mechanism design and review package

Base commit: bd64c2c1f76d927ce712b0751a3ae85ac192fb12. Developed in `/home/hh/TMC_re_adapterfl_full_only_400_20261019`. This package prepares S00-S09 for independent review. **No formal 400-round S experiment or waiting queue is started by this development task.**

The question is whether current functional compatibility evidence improves aggregation when restricted to the actual structural support where direction comparison is valid. S09 is a candidate, not a presumed winner. These controls do not implement the original paper's complete Semantic Memory or Reliability Memory.

## 1. Fixed protocol and ten controls

Original seven ResNets, 100 clients, 4:3:3 resource groups, 10 slots/round, original sampling, saved CIFAR-100 Dirichlet alpha=.3 partition, seed1, 400 rounds, original5 local epochs/batch50/SGD/lr/decay/momentum/weight decay, Full-only Accuracy and CE with isolated evaluation RNG. All seven synchronize after aggregation. Original aggregation, model shapes/slicing and models/Update.py remain unchanged.

|ID|Method|Functional measurement|Decision|
|---|---|---|---|
|S00|s00|No|Native coverage-normalized AdapterFL|
|S01|s01|No|Unchanged original M2, conflict_lambda=4|
|S02|s02|Yes|Measure/log only; native result retained exactly|
|S03|s03|Yes, logging only|Regional direction conflict only|
|S04|s04|Yes|Regional functional disagreement only|
|S05|s05|Yes|Joint regional evidence pooled into physical-client global Gates|
|S06|s06|Yes|Same base joint evidence as S05, Gates stay within compared regions|
|S07|s07|Yes|S06 plus prior valid-observation count confidence|
|S08|s08|Yes|S07 with shuffled semantic scores, unchanged evidence eligibility|
|S09|s09|Yes|S07 with native coverage denominator for gated learning increments|

S03 logs the same functional responses as S06 but ignores them in decisions, matching observation overhead. Defaults are fixed before S-series test results: gate strength4 reused from the direction reference, two examples per eligible class, one history pseudocount, epsilon1e-12. No threshold search, KD, local loss change, new model, or official-test feedback enters decisions.

## 2. Actual structural regions

For each parameter key, partition its first one/two dimensions at every real profile prefix boundary; keep convolution spatial dimensions whole. Group disjoint cells sharing the same supporting-profile set into a stable region. The real seven ResNets produce **9 regions**, exactly11,424,356 coordinates. Shapes, cell bounds, supporting profiles and geometry hash are saved in the manifest.

A region uses only slots whose actual profile covers it. At least two different physical clients are required for intervention. Single-client regions, including repeated slots of that client, are native passthrough. Uncovered coordinates stay unchanged. Multiple distinct Full holders may compare jointly covered Full-only coordinates; one Full holder is protected.

Project actual local increments d_i=U_i-S_i onto each region. Average repeated slots of a physical client for its evidence reference, counting its sample weight once; exclude all slots of the target client from its leave-one-client-out reference. Other physical-client references are sample-weighted. Define D_i,r=max(0,-cos(d_i,r,reference_i,r)). Zero-norm increments/references are missing evidence, not conflict.

Training and aggregation keep repeated upload slots separate with original sample counts. Pooling is only for evidence. Declared profile shapes are checked against actual uploaded shapes.

## 3. Class-conditioned functional response

For each physical client/class with at least2 local training examples, select2 deterministic hash-ranked indices using seed/client/index. Use only that client's training dataset. The observer accepts no test or validation dataset.

Before/after unchanged local SGD, evaluate a detached model copy with no_grad/eval on identical selected examples and replayed training augmentations. Views use isolated hash(seed,round,physical_client). Python, NumPy, Torch CPU and initialized CUDA RNG states are restored in finally; training parameters, buffers, gradients and module modes are preserved by copying. The loader has a dedicated generator.

P_pre[c,:] and P_post[c,:] are mean100-class probabilities conditioned on local true class c. Observation F_i,c=P_post[c,:]-P_pre[c,:] uses the common CIFAR-100 output coordinates, not unaligned hidden features. There are at most200 examples and two forward passes per slot. Missing classes stay missing.

Within a region's supporting cohort, average repeated observations within each physical client, class by class. Compare each class to other physical clients with that class. Q_client,r is the equal-class mean of max(0,-cos(F_client,c,mean_other_clients(F_other,c))) over nonzero valid comparisons. Empty/zero references give missing evidence. Q is a functional-disagreement proxy, not a harmfulness label or per-kernel localization.

Private-example variation, local overfitting and BatchNorm batch context can affect responses. Cross-client output-coordinate comparability alone does not establish update benefit. S04 and S03/S06 test that hypothesis. Logs retain class coverage, true-class probability changes, response norms/hash, observation seconds and a semantic payload-byte proxy; dense matrices are not dumped into every CSV.

## 4. Risks, Gates, confidence and global control

Valid regional risk is D for S03, Q for S04, and D*Q for S05/S06. Missing evidence gives risk0/Gate1. S07/S08/S09 multiply joint risk by c=n/(n+1), where n counts prior valid direction-and-semantic observations for this physical client/region. First observation has confidence0. After decisions, each physical-client/region count increments at most once per round, including repeated slots. Nonconflicting valid observations count too. No historic vector or semantic anchor is reused; confidence measures observation exposure. Counts/config/geometry/round progress are saved/restored.

Regional Gates are g=exp(-4*risk). S05 averages repeated-slot risks per physical-client/region, then averages valid regional risks weighted by region size into one physical-client score. Its Gate is applied across that client's multiply-covered upload, deliberately testing extrapolation beyond the evidence region. S06 uses the same base evidence with regional Gates. Normalized single-client regions remain native even for S05.

S08 permutes already-valid physical-client semantic scalar scores within each region using a dedicated deterministic NumPy Generator. Client membership, original valid-class metadata, direction evidence and history eligibility remain fixed. The regional score multiset is exactly preserved. Use a nonidentity client derangement for at least2 valid clients; constant scores remain unchanged numerically. Log donors and actual changed-score counts. Labels, masks, examples and training RNG draws are never permuted. Donor validity metadata is not reinterpreted as the recipient's class measurement.

## 5. Aggregation and dispatch anchor

For a region with supporting slots I, N=sum_i n_i and d_i=U_i-S_i, let native A be the unchanged original aggregation. Define the ungated dispatch-fusion anchor B=sum_i(n_i*S_i)/N and gated learning numerator T=sum_i(n_i*g_i*d_i).

- S03-S08: A_new=B+T/sum_i(n_i*g_i).
- S09: A_new=B+T/N.

Implementation computes A_new=A+[T/Z-sum_i(n_i*d_i)/N], preserving the native initialization-fusion anchor without reweighting independently initialized dispatched models. All-neutral regions return A directly, bitwise; normalized common Gates also return A directly. Single-client and uncovered regions return native results. Only S05 deliberately extrapolates a learned Gate to another multiply-covered region.

This anchoring is shared across S03-S09. S01 remains the original M2 absolute-upload reweighting, including its initialization behavior. Thus S01/S03 is not interpreted as isolating only region partitioning during initialization. The requested S03/S06 and S05/S06 contrasts use the same new anchor.

At identical inputs/evidence/history/Gates, S09 differs from S07 only by the denominator: its gated learning vector equals S07's vector times sum(n*g)/N. The native dispatch anchor is unchanged. This can suppress learning-update strength relative to normalized redistribution without importing a global coverage penalty. It need not reduce the server-step norm relative to native A, because suppressing opposing contributions also changes cancellation. Training trajectories may then diverge across rounds.

## 6. Logs and offline comparisons

Per-round CSV retains Full Accuracy/CE, cumulative peak/round and timings. Diagnostics retain stable region IDs/size/supporting profiles, actual slots/physical clients, valid direction and semantic risks/classes, confidence, donor map, Gate values, coverage mass, learning-update norms and native-difference L2/changed-coordinate fraction. Gate-target and actual-changed fractions are separate. Per-region fallback reasons and insufficient-history slots distinguish missing evidence, cold history and neutral decisions.

S01 retains the original M2 diagnostics; its region coverage is reconstructable from manifest geometry and logged slots/profiles, and the offline reporter exports S01_structural_coverage.jsonl. Best/latest, source hashes, configuration and existing output protection remain intact.

After all400 rounds, generate comparison.csv/report.md, all-ten and individual Accuracy curves, raw per-round export, and comparisons with historical B00/E03/N08. mechanism_summary.json and mechanism_report.md compare S00/S02 final weight/RNG equality, schedule/initialization pairing, conflict/validity/Gate/affected-region/fallback fractions, observation cost, and S07/S09 learning strengths. Test-oracle peaks remain exploratory official-test selections; no report feeds back into training.

## 7. Preflight, resource limits and smoke scope

CPU tests include original baseline/M2/M3 compatibility, real seven-profile disjoint geometry and mapping, neutral/single-client fallback, deliberate S05 extrapolation versus S06 locality, duplicate-slot evidence/history handling, S08 distribution/validity/RNG invariance, identical S07/S09 Gates with the prescribed denominator difference, exact S00/S02/old-baseline training trajectories, measurement model/RNG preservation, interrupted S07 restore, fixed commands and report generation.

The preflight uses the existing hh environment. GPU0/GPU1 had other users' compute processes, so tests and smoke set CUDA_VISIBLE_DEVICES empty and use two CPU threads per process. No other task is stopped or displaced.

Smoke calls the actual AdaptiveFL loop and actual seven ResNets on CIFAR-100, with100 clients,10 slots/round,5 local epochs and batch50, but explicitly restricts each original private client partition to its first50 examples and test evaluation to200 examples. This is a labelled implementation test, not a formal accuracy result or original full-partition experiment. Its manifest stores the original split hash, actual subset hash, runner hash and subset sizes. Each S mode runs2 rounds, at most two CPU smoke processes concurrently, only under runs/semcons_scope_preflight/smoke/. The smoke tool refuses more than3 rounds. Optional --full-data exists for later checks when resources allow.

Actual S00/S02 two-round CPU CIFAR smoke produced bitwise-identical Full parameters, maximum difference0 and equal training RNG states. Three-round synthetic augmented-data tests also match all seven synchronized models and old named baseline exactly. Formal GPU runs preserve the original nondeterministic CUDA backend settings; the package does not claim future400-round GPU bitwise identity from CPU checks. The completion reporter measures final S00/S02 weights and RNG directly; any discrepancy must be analyzed, not inferred away from similar Accuracy.

Formal400-round source/data/config are never replaced with smoke subsets. Old B00/E/N results and checkpoints remain untouched. The new source fingerprint intentionally differs; do not resume old-source checkpoints with this code.

## 8. Review and commands

Fixed config: configs/semcons_scope10_400.json. Output root: runs/semcons_scope10_full400_seed1/. Batches: S00+S02, S01+S03, S04+S05, S06+S07, S08+S09. Reuse the existing two-process queue, independent directories/logs, PIDs/status,400-round validation, failure-stop policy, no retries/tuning/reordering. nvidia-smi is checked at each batch boundary; external compute occupants are not preempted. Shared GPU0 requires explicit --allow-shared-gpu after checking resources.

Dry-run, already safe to execute:

```bash
cd /home/hh/TMC_re_adapterfl_full_only_400_20261019
PYTHON=/home/hh/miniconda3/envs/hh/bin/python bash tools/run_semcons_scope10_400.sh --dry-run --gpus 0,0
```

Only after independent GitHub review and new execution authorization:

```bash
cd /home/hh/TMC_re_adapterfl_full_only_400_20261019
PYTHON=/home/hh/miniconda3/envs/hh/bin/python bash tools/run_semcons_scope10_400.sh --execute --gpus 0,0 --allow-shared-gpu
```

No --execute call is made for the formal queue during development. No formal wait/launch automation is installed. No real-time per-round monitor is used.

## 9. File map

- semcons_scope.py: measurement, support geometry, evidence, Gates, history and aggregation.
- Algorithm/Training_AdaptiveFL.py: opt-in before/after observations, exact dispatch snapshots, original local training otherwise retained.
- adapterfl_methods.py: opt-in scope-hook factory, checkpoint/manifest geometry and source fingerprint; old methods retained.
- utils/options.py: add s00-s09 choices.
- configs/semcons_scope10_400.json, tools/run_semcons_scope10_400.sh: fixed modes and existing queue wrapper.
- tools/report_semcons_scope10.py: mechanism comparisons and S00/S02 final state check; tools/report_next10_m2_m3.py gains generic title/reference columns while keeping old reports compatible.
- tools/smoke_semcons_scope10.py: bounded, explicitly labelled smoke helper; never400 rounds.
- tests/test_semcons_scope.py: focused regressions and constructed intervention cases.
- docs/SEMCONS_SCOPE10_DESIGN.md and preflight evidence: audit specification and actual checks.

## 10. Actual seven-profile region inventory

|Region|Supporting profile IDs|Coordinates|
|---|---|---:|
|0|[0, 1, 2, 3, 4, 5, 6]|2120985|
|1|[1, 2, 3, 4, 5, 6]|185955|
|2|[1, 2, 4, 5, 6]|339416|
|3|[2, 3, 4, 5, 6]|671880|
|4|[2, 4, 5, 6]|29040|
|5|[2, 5, 6]|1300464|
|6|[3, 4, 5, 6]|2180801|
|7|[5, 6]|117040|
|8|[6]|4478775|

P0=(.40,2), P1=(.40,3), P2=(.40,4), P3=(.66,2), P4=(.66,3), P5=(.66,4), P6=(1,2). Regions are rebuilt from actual state shapes, not copied from the old three-tier implementation.
S08 preservation is checked conditional on each round's pre-shuffle input. Once different Gates change model weights, later functional observations may legitimately diverge across methods.
Observation/native-train timers are wall-time diagnostics. Formal CUDA asynchronous execution can shift time between sections; total elapsed time and paired profiling are needed for an overhead claim.

Metric denominators: Gate trigger and semantic-comparison fractions count region-slot pairs within cohorts with at least two distinct clients; local class coverage counts valid class coordinates per slot. Gate-target area is weighted by region size over covered coordinates, while actual-changed area is counted over Full coordinates. Neutral/fallback region fractions use the fixed nine region types, including uncovered types. Controls without measurement are identified by measurement_enabled rather than interpreted as failed observations.
