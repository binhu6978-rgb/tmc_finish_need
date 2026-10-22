# A07 integration source audit

Audit baseline: server HEAD `24e79dbb26ce718b533e7bc2c440a91c628caa25`; original development branch HEAD `398d44a44fe16358b6fdd756160c40d07ffbfbf2`. The latter adds launch evidence; a direct diff of the A07, SGD, aggregation and training-entry files is empty. Integration branch starts from that descendant. Completed forty-run evidence is archived in `docs/evidence/final40_results`; it contains forty completed, finite, consecutive 400-round records, rather than launch-only claims. A07 achieved test-oracle peak 39.94%, Last20 38.6235% in that single-seed exploration.

## Frozen executable baseline

| Source | Actual behavior | Integration consequence |
|---|---|---|
| `models/resnet.py` | Custom stem64 and stages96/128/256/512. Width changes begin at `slim_idx`; representation is a 4-D final feature map. BN uses the model's actual `track_running_stats=False` configuration. Classifier has actual class count. | Pool features spatially; preserve prefix geometry and BN batch composition. This is not torchvision ResNet. |
| `models/Fed.py`: `get_model_list`, `split_model`, `Aggregation_AdaptiveFL` | Seven independently initialized prefix models; native covered-coordinate sample normalization, uncovered Full coordinates retained. | Preserve first-round fusion and all seven synchronization. Never zero-pad an absent update as a real observation. |
| `semcons_scope.py`: `build_regions` | Rectangular cells from real parameter shapes, grouped by identical profile support; nine regions partition the Full state. | Reuse geometry, without reusing S03/Q decisions. |
| `capability_final40.py`: `source_weights`, `Final40Hooks.aggregate` | Physical capabilities use cid boundaries40/70. In each covered region, group mass is sample-slot mass. P04 target is proportional to mass^-1/2; within group sample weighting. A07 excludes classifier keys from rewriting. One-tier regions retain Native. Repeated slots remain distinct. First call is exact Native. | Call this frozen implementation directly for gamma=1. Semantic identities must be deduplicated separately from upload weights. |
| `Algorithm/Training_AdaptiveFL.py` | Original local SGD per actual slot; aggregation followed by `split_model` for all seven; Full-only evaluation uses isolated RNG. | Wrap observation around local `train`, not a new FL loop. |
| `models/Update.py`: `LocalUpdate_AdaptiveFL.train` | Task CE, original SGD, original shuffled/drop-last local loader, five epochs. | Call it unchanged. |
| `tools/run_final40_job.py` | Prepared partition; deterministic CUDA; seed1, including unused Full construction before seven models; no PT. | Reuse setup and that initialization order, with explicit short regression budgets. |

## Earlier paper-style implementation

`models/Update.py:train_with_proto` feeds **non-detached** representations through Adapter and frozen projection, optimizes Task CE **plus a private auxiliary CE**, and adds Adapter/private classifier parameters to the optimizer. The auxiliary CE reaches the Backbone. Post-training prototypes use the shuffled training loader; no fixed paired pre/post views exist. It changes the A07 gradient and RNG trajectory.

`models/aggregation_semanticfl.py:SemanticFLServer` keeps Adapter by `(client_id,input_dimension)`, not full profile, and private classifier by client. Its shared random projection is frozen, but private trainable decoders and Adapters do not establish cross-client embedding alignment. Global prototypes initially contain random normalized vectors, including classes without observations. Current semantic scores enter client history before reliability is calculated; duplicate slots may update it repeatedly. Global sample weights and padded missing coordinates do not reproduce native coordinate coverage normalization. The non-key update can shrink coordinates without an upload. Top-K/trusted/protective branches also differ from A07. Warmup beta=1 does not repair those differences.

## Later FD implementation and scope experiments

`semconsfl/measurement.py` measures class-conditioned task probabilities, not Adapter embeddings. It isolates RNG and restores modes/buffers. `semconsfl/semantic_memory.py` snapshots past anchors, uses signed pre/post JS changes with a confidence guard, and updates after aggregation from valid pre-signatures, pooling duplicate physical identities. Round0 is excluded; insufficient classes are missing. `semconsfl/reliability_memory.py` uses past profile anomaly calibration and client history, with separate missing/age/support protection. Its profile pool is not personal reliability. These are useful timing safeguards, not evidence that these scores predict utility.

`semconsfl/aggregation.py` falls back to **Native**, not A07, when neutral; nonneutral updates use old Full rather than actual dispatch, and do not preserve A07 classifier/source rules. `semconsfl/trainer.py` is a separate loop. Neither is reused as the integration training engine.

`semcons_scope.py` supplies nine-region geometry and earlier probability-response Q. Previous receiver diagnostics found no stable incremental Q information; the integration does not reactivate it. `tools/receiver_ce_probe.py` supplies examples of paired fixed training probes and native injection, but its PT saving is not reused.

## Required scientific distinctions

Shared embedding dimension, frozen projection and supervised class labels do not prove embedding comparability. Class-coordinate probabilities are comparable as defined outputs, while their predictive usefulness remains empirical. Updating an Adapter changes the measuring instrument across rounds; record the frozen pre/post Adapter version and its separate calibration change. Distinguish current prototype disagreement, historical anchor drift, sender self-learning, direction, amplitude and recipient CE impact. Missing evidence is not a measured zero. Old convergence statements do not cover the proposed observational or potential gated system.

The frozen A07 files and old experiment results are preserved. New instrumentation, its configuration digest, actual source commit, partition digest and command are recorded separately. Scientific Go is decided from train-side held-out utility, never Full test accuracy.

## Final empirical disposition

The new observer completed its real2/20 exact regressions and the fixed400-round observational trajectory. All400 Full accuracy/CE values and schedules match frozen A07. The train-side utility analysis returned No-Go; semantic gating and conditional formal experiments were stopped. Prototype comparability and Memory utility are not inferred from this source audit. Final evidence, limited positive/negative findings and unresolved reviewer issues are in SEMCONS_A07_MECHANISM_VALIDATION.md and SEMCONS_A07_FINAL_AUDIT.md. No legacy or A07 training/aggregation source was changed to obtain the result.
