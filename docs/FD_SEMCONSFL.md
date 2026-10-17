# FD-SemConsFL implementation and review guide

This branch adds the finalized four-stage method to commit
`d6354d073ff4d5614a4e6b25fb542a9b43e0dfa8`. It does not edit any existing
Python source or partition. No CIFAR100 training or performance tuning was run.

The user's subsequent instruction keeps **all original experimental settings**.
It supersedes the implementation plan's proposed validation split: this code
retains the original 50,000 training examples, original client indices, original
10,000-example official test evaluation each round, and all seven profiles.
No new validation set or changed partition protocol is introduced. An official
test peak remains a historically exposed test peak; it is not independent
validation evidence. Test metrics never enter the Gate or Memory decisions.

## Complete file list

All changes are additions. No pre-existing tracked file is modified.

```text
.gitignore
.gitattributes
configs/fd_semconsfl.json
docs/FD_SEMCONSFL.md
docs/baseline_audit/baseline_manifest.json
experiments/__init__.py
experiments/baseline.py
experiments/run_semconsfl.py
semconsfl/__init__.py
semconsfl/aggregation.py
semconsfl/config.py
semconsfl/measurement.py
semconsfl/reliability_memory.py
semconsfl/semantic_memory.py
semconsfl/trainer.py
tests/test_fd_semconsfl.py
```

## Baseline boundary and actual execution

The historical entry `main_fed_ori.py`, `Algorithm/Training_AdaptiveFL.py`,
`models/Fed.py`, `models/Update.py`, `models/resnet.py`, `models/test.py`, and
`utils/options.py` remain byte-for-byte unchanged. Historical semantic adapters,
`train_with_proto`, auxiliary Semantic CE, and `models/aggregation_semanticfl.py`
are not used by the new method.

The independent entry reads the original argument defaults, reproduces the
otherwise-unused initial Full-model construction, then calls the original
`get_model_list`. The seven independent initializations are retained in round0;
each subsequent global state is distributed by the original `split_model`.
`slim_idx` controls where narrowing starts, not residual-layer deletion.

The default experiment is CIFAR100, ResNet18, Dirichlet alpha=.3, seed=1,
100 clients, 4:3:3 tiers, 1000 rounds, fraction=.1, five local epochs, batch=50,
SGD lr=.01, decay=.998 per round, momentum=.5, weight decay=.0001, test batch=128.
Profiles are `(width,slim_idx)` = `(.4,2/3/4), (.66,2/3/4), (1,2)`.
The original `available` scheduler draws profiles uniformly with NumPy and calls
the original `select_clients` using Python RNG; Small can use clients 0--99,
Medium 40--99, Full 70--99. Repeated physical clients retain all aggregation
slots. The new runner explicitly rejects other scheduling policies rather than
silently substituting them. It supports the requested 100-client 4:3:3 scenario.

Local SGD **directly calls** `LocalUpdate_AdaptiveFL.train`: fresh per-slot
optimizer, pure Task CE, shuffled original dataset order, `drop_last=True`.
The effective aggregation weight remains the original client's entire sample
count, including examples dropped in a particular SGD epoch. No augmentation,
optimizer, representation, or classifier change is made. The train/test transform
is precisely the original `ToTensor + Normalize`.

`generate_data=0` is enforced in this entry solely to read and preserve the
existing cached partition; it cannot regenerate or overwrite the cache. The old
entry's default `generate_data=1` and its behavior are untouched. The original
default JSON filename says `_bal`, but the actual metadata says
`dirichlet_unbalanced`; audited client counts range 262--757. This is reported,
not corrected. All three arms load the same indices in the same order.

## Four-stage correspondence

| Stage | Implementation | Invariants |
|---|---|---|
| Functional measurement | `semconsfl/measurement.py` | Original 100-class softmax, T=1; complete sorted unique local training indices; identical pre/post batches, label ordering and class counts; no gradients; restores every module mode, buffer and Python/NumPy/CPU/all-CUDA RNG state, including on exceptions. |
| Semantic Memory | `semconsfl/semantic_memory.py` | Wrong-class conditional distributions with sufficient support/mass; signed JS drift relative to old anchor; true-confidence improvement protects a class; support-weighted anomaly over comparable classes only. |
| Reliability Memory | `semconsfl/reliability_memory.py` | Old same-profile median/IQR calibration; clipped z; old physical-client mean shrunk by m/(m+nu); risk=z*r; one event per client per round after profile calibration; missing calibration never becomes a fabricated zero history. |
| Aggregation | `semconsfl/aggregation.py` | Original prefix masks and ungated coverage denominator; only `(local-old Full)` numerator gets gamma=1-lambda*risk; uncovered coordinates retain old values. |

Semantic reference update: average valid pre signatures within each physical
client/class, then take the coordinate median across clients, normalize its L1
mass, and apply EMA. A duplicate slot does not create another client vote.
At least two distinct clients are required for an anchor update. Round0 supplies
neither anchors nor reliability records, since its submodels have independent
initializations. The first possible anchor update is **after round1 aggregation**,
using round1's pre-training signatures of already synchronized models. It first
becomes usable for round2 scoring. Current post signatures never update anchors.

Profile pools store at most one averaged raw anomaly per client/profile/round.
After old-profile calibration, repeated client/profile z values are averaged;
profile-level means then form one physical-client observation. Valid zero
anomalies and calibrated zeros are retained; unavailable values are missing.
Clients with insufficient old observations contribute neutral risk, while their
valid current z may become a future observation. All slots read the same frozen
old snapshots, and all Memory writes occur after aggregation and synchronization.

Frozen engineering choices are in `configs/fd_semconsfl.json`: class support=5,
wrong mass >=1e-6, confidence tolerance=1e-7, measurement batch=50, EMA=.9,
at least two clients for an anchor, profile pool=200/minimum=10 records,
IQR floor=1e-6, client history=5/minimum=2 events, maximum age=100 rounds,
nu=2, lambda=.25. Stale anchors are excluded and are reinitialized only from a
fresh valid client consensus. These are explicit initial engineering choices,
not performance-selected values. There is no extra trust model or fifth module.

All-one gates and round0 call **the unchanged `Aggregation_AdaptiveFL` directly**,
yielding bitwise equality to that function. For non-one gates the formula is

`old[k] + sum(n_s*M_s[k]*gamma_s*(local_s[k]-old[k])) / sum(n_s*M_s[k])`.

This intentionally changes the original paper's global denominator, as mandated
by the final algorithm design. No Gate denominator, direction filter, trusted
subset, protective branch, BDRC, G5, Adapter, auxiliary CE or distillation is added.
Current ResNet states contain float32 weight/bias tensors and no BN counters;
unsupported integer/scalar states or inconsistent prefix shapes fail explicitly.

## Arms and commands

Run from the repository root using an environment with torch, torchvision,
numpy, tqdm, matplotlib and thop (the last two are imported by existing modules).
Use pytest for the synthetic suite. No new framework dependency is added.
The verified environment is Python 3.12, torch 2.7.1+cu128, torchvision 0.22.1+cu128.

| Mode | Local objective | Measurement/Memory | Aggregation |
|---|---|---|---|
| A0 | Original Task CE | Off | Original AdaptiveFL |
| A1 | Original Task CE | On | Original AdaptiveFL, gates fixed at one |
| A2 | Original Task CE | On | FD-SemConsFL; neutral fallback to original |

These commands are **instructions for future authorized runs**, not commands
executed for the current implementation. DATA must already contain the original
`cifar-100-python` files; the runner uses `download=False`.

```powershell
python -m experiments.run_semconsfl --mode A0 --dataset-root DATA --output-dir runs/A0_seed1 --seed 1
python -m experiments.run_semconsfl --mode A1 --dataset-root DATA --output-dir runs/A1_seed1 --seed 1 --semantic-config configs/fd_semconsfl.json
python -m experiments.run_semconsfl --mode A2 --dataset-root DATA --output-dir runs/A2_seed1 --seed 1 --semantic-config configs/fd_semconsfl.json
```

Additional entry flags: `--partition-path EXISTING_JSON`, `--gpu -1` for CPU,
`--audit-only`, and `--resume`. The original baseline flags are parsed by the
unchanged `utils.options.args_parser`. A different seed needs an already existing
matching partition; the runner does not create one. Use identical common flags,
data and device for all arms. No special A2 optimizer or data recipe exists.

Read-only audit (no local model training or evaluation):

```powershell
python -m experiments.run_semconsfl --mode A2 --audit-only --dataset-root DATA --output-dir audit_seed1 --gpu -1
```

The committed `docs/baseline_audit/baseline_manifest.json` records original source
SHA256 values, immutable partition hash and per-client class histograms, all
initial model hashes, actual seven-profile coverage, and 1000 rounds of original
scheduling. Audit initialized models and inspected labels, but did not perform
SGD or report accuracy. No private research documents or raw CIFAR files are
included in this branch.

Git attributes preserve stored file bytes across Windows/Linux checkouts, so
line-ending conversion cannot invalidate the audited source hashes.

## Logging and recovery

Each new run requires a new output directory. `manifest.json` records settings,
source/environment/data hashes and the expected original schedule. Scheduling
still makes the original RNG calls each round, and checks them against the audit;
it does not replace the scheduler with a different sampling rule.
`rounds.jsonl` records every real slot, counts, sparse supported-class B/A rows,
confidence and wrong mass, validity/missing reasons, signed drift, calibration,
historical support, r/R/gamma, all seven original test results, and timings.
It also records update norms with/without gates for shared, partially covered,
and Full-only regions. Norm ratios need not be <=1 when cancellation changes.
Per-class counts/signatures are additional research-side information and have
measurement/serialization cost; this implementation makes no claim about the old
paper's overhead numbers.

`latest.pt` atomically stores the synchronized global state, both Memories,
next round and Python/NumPy/CPU/CUDA RNG states after evaluation. Resume the same
command and directory with `--resume`. Only complete round boundaries are
supported. A changed source/config/data/environment/mode or a log/checkpoint
mismatch raises an error. A crash between log write and checkpoint replacement
requires inspection/restart in a new directory; it is not silently repaired.
No historical result file is appended to or overwritten.

## Verification and remaining scientific questions

```powershell
python -m pytest -q tests/test_fd_semconsfl.py
```

36 synthetic tests passed, including CPU and CUDA local-observer comparisons;
none were skipped on the verified RTX 4080 SUPER environment. Coverage includes:

- Bitwise all-one equivalence for independent/synchronized states and round0.
- Single-covered-coordinate attenuation, unequal sample weights, fixed ungated
  denominator, convolution prefixes and exact retention of uncovered coordinates.
- Invalid shapes/keys/nonfinite data/integer/scalar states fail explicitly.
- Unchanged local weights, gradients, batch order, mixed module modes, buffers,
  Python/NumPy/CPU/CUDA RNG, including exception restoration and BN tracking modes.
- Fixed BN batches repeat; changing BN batch membership changes measured behavior.
- Pre/post label-correspondence mismatch detection, JS bounds, true-confidence
  protection, low class support and unstable error-tail rejection.
- Old anchor/history snapshots, round0 exclusion, duplicate physical clients,
  profile-first calibration, history shrinkage and neutral missing/stale evidence.
- Three toy rounds of A0/A1 bitwise trajectory/RNG equality; A2 fallback equality;
  injected valid-risk A2 integration with the expected actual gate and Full-only
  attenuation; synthetic uninterrupted/resumed trajectory equality.
- Original default arguments, original dispatch calls, original unused-model
  initialization, and all seven real ResNet state shapes/coverage.
- Immutable cached partition, duplicate rejection, unique outputs and checkpoint
  round/config/source/mode boundary checks.

All 11 new Python files passed AST parsing. The five warnings are from existing
thop distutils comparisons and the existing options.py string escape.
No CIFAR100 local training, federated training, short performance experiment,
official-test tuning, or claimed accuracy improvement occurred.

The actual parameter coverage is 2,120,985 coordinates in all seven profiles,
4,824,596 in a proper subset, and 4,478,775 Full-only, total 11,424,356.
These are freshly calculated static properties, not measurements of beneficial
or harmful updates. Formal experiments are still needed to determine whether
functional drift predicts update utility, whether history adds information,
whether gates outperform matched generic step-size shrinkage, and whether
Full-only attenuation harms useful sparse learning. The original paper's
convergence result and overhead claims do not establish these for the new variant.
