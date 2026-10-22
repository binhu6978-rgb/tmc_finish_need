# Frozen A07 observational design v1

Frozen before new GPU observations. Machine-readable constants: `configs/semcons_a07_observation.json`. Stage B/C are read-only; gamma is exactly1 and actual aggregation calls the unchanged `Final40Hooks.aggregate`. No formal Gate is authorized by an implementation test alone.

## Budget and separation

Two fresh seed1 A07 runs of20 rounds: original reference and instrumented observation. Compare all seven states, slot identities, Full test CE/accuracy and training RNG at initialization and every round, including round2. No checkpoint is written. Any mismatch stops further observation runs pending diagnosis.

After successful equivalence, one fresh400-round **observation-only A07 trajectory** generates actual uploads in memory. Fixed diagnostic rounds: early2/5/10/20, middle180/190/200/210, late370/380/390/400. Up to two senders per covered multi-physical region, two other physical receivers each, hash ordering fixed independently of scores/CE/direction. The first round is excluded. No additional trajectories or cases are added to remedy insufficient samples. A short20-round observed run is the utility pipeline pilot, not held-out efficacy evidence. No historical upload is reconstructed from CSV.

Development: early-stage pairs whose sender and receiver IDs are both <70. Select at most one of S0/S1/S2 on development grouped cross-validation error (tie order S0,S1,S2). Confirmation: middle/late pairs whose sender and receiver IDs are both >=70. The remaining records are descriptive only. This separates both stages and physical participants. The partition can yield too few pairs; that is an explicit inconclusive/No-Go outcome, not permission to relax it.

## Detached measurement instrument

Fixed head seeds17/29/43 are measurement sensitivity seeds, not FL seeds. For each seed, a frozen 128x128 projection H and 100x128 class decoder D are deterministically initialized. Class decoder row y always denotes true class y. Per `(physical_client,profile,head_seed)` trainable affine Adapter A maps pooled Backbone features to128. Outputs are z=normalize(H A h) and p=softmax(D H A h). Backbone task probabilities q are also recorded. Adapter initialization uses an identity-specific isolated seed. Same-round slots read the same committed state; their candidate Adapter states are averaged once on commit. No slot consumes another slot's uncommitted measurement state.

Each client probe contains hash-selected two training samples for every class with at least two samples. Input views are materialized under an isolated client-specific seed and reused pre/post, preserving order and batch50. Feature extraction is eval/no_grad on a model copy (BN still depends on the fixed batch). Pre/post use exactly the same Adapter snapshot and frozen head. Candidate Adapter fitting uses detached post features, one fixed-order epoch, CE(D H A h,y), SGD lr0.01, no momentum/decay; only measurement parameters change. Fitting is for the **next** committed observation. Measure old/new Adapter responses on identical post features separately to expose instrument drift. No auxiliary loss reaches the main model. Report forward examples, Adapter backward examples, bytes, elapsed time and state size.

## Prototype, references and scores

For class c with support>=2: mean normalized embedding followed by L2 normalization; mean probability in true100-class order, renormalized to unit sum. Missing classes remain absent. At round start take immutable previous-commit anchors. The first round contributes no anchor. Per class/representation/head, average duplicate slots and profiles within each physical client, coordinate median over >=2 different clients, normalize, then EMA: anchor=normalize(0.9 old+0.1 current), resetting after age100. Valid support and timestamp are stored. Mean prototype aggregation is an observation definition, not proof of aligned embeddings.

Scores are signed harmfulness predictions, averaged uniformly over valid receiver probe classes shared by sender pre/post and past anchor:

- S0 = mean[1-cos(z_post,c,anchor_z,c)]. Higher means disagreement, an unvalidated risk hypothesis.
- S1 = mean[(1-cos(z_post,c,anchor_z,c))-(1-cos(z_pre,c,anchor_z,c))]. Positive means movement away from old anchor, allowing beneficial negative values.
- S2 = mean[JS(q_post,c,anchor_q,c)-JS(q_pre,c,anchor_q,c)]. Task output control, independent of Adapter/head.

Also record Adapter probability JS changes, class support, anchor age and instrument drift. S0/S1 are evaluated for all three head seeds, without choosing a head by test score. Scores describe client functional observations; no claim of per-kernel semantics is made.

Semantic Memory contains category references. Independent Reliability Memory contains past per-client and per-client/profile valid-evidence counts, age and dispersion; it is read before scoring and committed afterwards, with one observation per physical identity per round. Group summaries remain group summaries. At least two prior valid rounds and age<=100 are required for readiness. It does not infer correctness from repeated disagreement. Stage C explicitly tests whether past support/stability adds held-out utility information; no reliability Gate is enabled here.

## Real utility and controls

For region r supported by different physical sender s and receiver j, alpha_s,r = n_s/sum_{slots covering r} n_i. On a copy of j's **actual dispatched model**, inject alpha_s,r*(actual_upload_s-actual_dispatch_s) only into r, preserving all other coordinates. Evaluate fixed receiver training probe (16 hash-selected supported classes, two examples/class), identical before/after BN batches. Positive per-class CE difference means recipient damage. Record sender/receiver/profile/tier/region/slot, duplicates, alpha, raw/scaled delta L2, region cosine to other-physical-client LOO mean, sender own pre/post CE gain, class support, all scores, past reliability and anchor metadata. Single-source/uncovered regions are excluded, not filled with fabricated utility. Selected pairs include conflicting and nonconflicting cases by outcome-independent hash sampling; report actual counts, including zero negative cases.

## Precommitted scientific Go rule

Use positive utility delta as harmful label (zero is its own category), no threshold fitting. Controls: signed direction cosine, log scaled L2, alpha, pre-injection CE, sender own gain, sample mass, profile and tier indicators. Fit standardization and ridge-free least-squares only on development; compare controls with controls+selected score and controls+score+past reliability on disjoint confirmation. Use physical sender and receiver cluster bootstraps separately (500 fixed-seed replications), report the wider interval. Permute scores within stage/profile/tier/support-validity strata, preserving missingness and participant identities. No changed validity masks.

Go requires >=100 valid pairs, >=10 senders and>=10 receivers, >=20 harmful and20 beneficial labels **in each split**; selected score harmfulness AUC>=0.60 with both physical cluster95% lower bounds>0.5; positive association in both confirmation stages; >=2% held-out MSE reduction beyond controls with cluster95% lower bound>0; superior to permuted score; consistent association across all three frozen head seeds for Adapter scores; evidence beyond profile/mass. Memory evidence must independently improve held-out prediction beyond the current functional change. Failure, insufficient data, unstable head sensitivity or unidentifiable memory increments yields No-Go for a formal four-stage Gate. Descriptive positive associations alone are insufficient.

## Conditional later stages

Only after Go, freeze one score and one conservative nonnegative gamma before any semantic performance experiment. For qualifying shared feature cells, preserve A07 state fusion and add a correction sum a07_i,r*(gamma_i-1)*(upload_i-actual_dispatch_i); do not renormalize gated weights. Classifier remains Native, first round Native, missing/single-physical/Full-only/uncovered cells unchanged. All gamma=1 must return the original A07 result object path, avoiding an alternative floating-point computation. No active gamma rule is supplied or enabled in Stage B/C. A scientific No-Go ends this integration at observation and evidence delivery.

Formal six-arm400 and seed1/2/3 evidence remain conditional. Software implementation, mechanism Go, formal performance and paper qualification are distinct statuses.
