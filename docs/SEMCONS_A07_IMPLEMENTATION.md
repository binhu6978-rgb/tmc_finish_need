# Executable four-stage observation and mathematical scope

## Implemented path

`tools/run_semcons_a07_observation.py` calls the original `Algorithm/Training_AdaptiveFL.py` loop. A scoped context replaces only its client-selection recorder and logging/hook factory for this process; those bindings are restored on exit. Original SGD, source files and A07 arithmetic are unchanged. `--mode reference` calls the frozen A07 with numerical/RNG journals; `--mode observe` adds detached client measurement. Both are deterministic and checkpoint-free.

```text
initialize exactly as frozen A07, including the unused initial Full construction
for each communication round:
    snapshot committed Adapter/Semantic/Reliability states
    select original ten client/profile slots
    for each slot independently:
        fixed train-probe pre features, no_grad on model copy
        ORIGINAL complete local TaskCE SGD
        same train-probe post features, same frozen Adapter/head
        compute class prototypes and past-only scores
        fit next Adapter candidate using detached post features
    ORIGINAL A07 aggregation (round1 exact Native)
    if a pre-fixed diagnostic round:
        recipient train-probe CE before/after actual sender regional SGD injection
    ORIGINAL seven-model prefix synchronization and Full-only evaluation
    commit deduplicated measurement candidates and both Memories
```

Observation keeps current-round evidence out of the anchors/history used for scoring. Same physical/profile/seed slots read the same prior Adapter and commit one averaged candidate. Memories pool physical identities for evidence; A07 keeps every real training upload slot and its original sample mass.

## Frozen A07 mathematics

For each actual coverage region r, let I_r be uploading slots and I_g,r those from physical capability tier g. M_g,r=sum_i in I_g,r n_i. For active feature and BN cells, pi_g,r is proportional to M_g,r^-1/2 and a_i,r=pi_g,r*n_i/M_g,r, summing to1 over covered slots. Single-tier cells use original Native arithmetic. Classifier weight and bias use Native sample coverage weights. Uncovered coordinates retain old Full. Round1 uses exact Native independently initialized state fusion. The implementation reuses existing float operations rather than reconstructing this formula.

The detached measurement/score/Memory definitions are frozen in `SEMCONS_A07_DESIGN.md`. Affine Adapter loss is measurement-only CE(D H A stopgrad(h),y); H/D are frozen and the task classifier is separate. Category labels align probability coordinates; embedding comparability requires evidence. The Adapter-drift counter distinguishes changing instruments from a Backbone response change.

## Conditional semantic correction

If scientific Go is obtained later, a permissible correction to an already computed A07 result for qualifying shared feature cells is sum_i a_i,r*(gamma_i-1)*(upload_i-actual_dispatch_i). This preserves A07 fusion and lets gamma reduce actual learning contribution without renormalization. Actual dispatch is essential, including independent initialization. Classifier/Full-only/single-physical/uncovered cells and first round are protected. gamma=1 returns the original A07 directly.

At the current observation checkpoint **no nonneutral gamma formula is implemented or enabled**. `semcons_a07/aggregation.py` rejects it. This follows the required Stage C stop condition; a functioning observation stack is not presented as a validated semantic aggregation algorithm.

## Cost and privacy interpretation

Counters distinguish original local SGD duration, extra Backbone forward examples, detached Adapter backward examples, measurement wall time, train-probe utility forward examples and Adapter-bank storage. `logical_upload_bytes` counts actually constructed post-prototype tensor payloads in their actual dtype; metadata, encoding and network transport overhead are not included. This simulator executes no physical network transfer. Real deployment would perform measurement on clients, upload prototypes/support metadata, and retain private Adapter state locally. The simulator's central bank represents those client states.

Recipient utility is an offline research oracle. Deploying it online would require sending regional increments/identities and coefficients to receivers, additional receiver forwards, returning per-class CE/support, additional message rounds and privacy assessment. The server cannot assume possession of private client images. Probe images and raw feature/update tensors are not saved or pushed.

Storage/compute depend on actual observed classes and client/profile identities. Existing counters at100 classes/100 clients are measured evidence only at that scale; no extrapolated latency is labelled a measured scalability result.
