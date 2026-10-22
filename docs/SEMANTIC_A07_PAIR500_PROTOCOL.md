# Frozen V2: two 500-round trajectories

This task supersedes the unstarted 120-round heldout diagnostic at cafbc12. Its files and all earlier evidence remain intact. Algorithm baseline is b06a3e7, including the V2 formula and lambda=0.2; no score, threshold, memory or optimizer change is permitted during this run.

## Training and evaluation

Two fresh seed1 trajectories: A07-only (V2 measurements/memories observed, applied lambda=0) and V2 (applied lambda=0.2). Both use the original full 50000 training examples, prepared alpha0.3 partition, 100 clients, original seven ResNets, 4:3:3 physical capability tiers, 10 slots, five local epochs, original SGD and deterministic execution. Both observe the same measurement instrument so observer overhead is reported separately from algorithm time. Only their prescribed aggregation differs. Same initialization/schedule/RNG must be checked, but model trajectories are expected to diverge.

Official Full test Accuracy/CE is measured once per round by the unchanged test function and isolated RNG. No test output enters U, memory, probe sampling or aggregation. Test-oracle peak is exploratory. The user explicitly removed CRD from this task; no CRD is computed or fabricated.

## Mechanism sampling, fixed before outcomes

Use rounds 1,2 plus all rounds in 8..24,242..258,484..500: 53 rounds per trajectory. These are early/middle/late windows in the actual 500-round trajectory. Include neutral rounds and all selected pairs regardless of U or utility. Actual dispatch/upload tensors are retained only in memory on these rounds. Both trajectories run same-input A07 and V2 counterfactuals plus matched-mask no-anchor/no-rho, cyclic shuffled U and global-L2-matched A07. Counterfactual outcomes never feed training.

Per source tier/real covered feature region, keep the existing hash-selected different physical receiver in each other covering tier. Probe indices exclude every fixed semantic measurement sample, take two per available class with no 16-class truncation, use identical fixed augmented view/order/batch50 before/after. All 50000 training samples are already used by SGD; therefore these are **training-side evaluation probes, not independent validation samples**. Removing them from SGD would change the requested full training protocol, so this run does not do so. Distinct receiver identity alone does not eliminate training-set bias.

Common semantic categories and all probe categories are recorded concurrently; missing is not zero. Source U is a tier-round variable: recipients and regions do not increase independent semantic sample count. Group receivers by the frozen identity hash from the earlier expanded design; confirmation is the primary descriptive cohort, development is secondary, neither is used to tune scores. Record source-own task-measurement gain and source-own evaluation-probe gain separately. Full counterfactual panel uses one nonparticipating client per tier/cohort (six probes). Actual Full official-test curves are a third, separate result.

## Logging, storage and recovery

Every round retains current category task evidence, historical anchor age/support/reliability, U, A07 and hypothetical V2 region shares, applied shares, change norms, exact fallback reasons, schedule and training-RNG traces. No images or per-round uploaded tensors are saved. One atomic rolling recovery checkpoint per trajectory every50 rounds and at completion contains synchronized Full weights, original method state, adapter bank, both complete memories, RNG streams, next round and accumulated log rows. No test-selected best checkpoint. A completed-round journal is required for manual recovery; the launcher does not automatically retry/resume after failure.

Checkpoints are CPU copies and written after memory commit and RNG capture. Probe caches are deterministic and regenerated. A two-round diagnosed-A07 versus otherwise identical undiagnosed-A07 preflight must match seven synchronized states, SGD/RNG, Full evaluation and slots exactly. A synthetic non-neutral test covers the V2 branch and checkpoint round-trip; prior V2 real short regressions remain preserved. A failed preflight prevents formal startup.

Use GPU0 only while GPU1 is occupied, at most two independent processes; do not change training settings. Failures retain evidence and stop; no automatic new algorithms, seeds or replacements. A07 and V2 must each genuinely finish500 finite contiguous rows. Statistics use fixed ridge=1 and source-round/physical-identity grouping; bootstrap uncertainty is exploratory under one temporally dependent trajectory. Same-input counterfactual utility, score association, and actual two-trajectory performance are reported separately.
