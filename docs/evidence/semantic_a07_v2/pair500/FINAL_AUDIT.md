# Pair500 audit and evidence status

## Frozen execution

Training commit `49b5b5f86fe61815aa47b09445b27a0c6db2fbee`; both processes exited0 after500 real rounds. Original suite status is retained verbatim: it says failed because the *report process* exceeded Python CSV's128KiB field limit. The training completion markers,500-row finite journals,53-round diagnostics, final model checks and checkpoint audit are authoritative for training completion.

Original partition SHA256: `41ab2d3486663ce0100e183ab00565de137f91af07632c236c2ddf60d890ba41`.

Configuration identities differ as intended by applied lambda: neutral `ff372cb77c1c7f0adff1bd001cfb335fb76eb8944af676cc80ea7881fc7b1e9c`; active `68a92160eecb4f8eb74623ad88ddea21b5c714b7714974eec5ec2c47b08f9632`.

## Verified

- Two-round real complete-data diagnostic regression: exact seven models, slots, local SGD RNG and official Full evaluation. Local and Linux startup tests289 passed.
- Formal500: equal initialization, first-round models, all500 schedules and training/evaluation RNG streams; contiguous finite Accuracy/CE and genuine completion for both.
- New A07 first400 Accuracy/CE exact against original historical A07 CSV; historical file SHA256 `4d10eda3727348666404eae1be86785ce106f3ece6f6c0e427394b5136ff1fd0`.
- 53 preset diagnostic rounds per trajectory;1326 source-region-recipient records and1908 Full-counterfactual evaluations per trajectory. No additional sampling after outcomes.
- All37 original scalar journal/log/manifest files matched their archive index. Archive SHA256 `08f15cf26f0be4dc51131cc6ddc44019cd72d631618136eaa99f0001d4e42361`.
- Both complete500-round Semantic/ Reliability histories replayed to exact U and tier commits using only previous committed state; source fusion shares match frozen formula.
- Final checkpoints:500 rows, finite Full, exact final logged Full hash,1530 Adapter states,300 tier/class anchors and histories, all Python/NumPy/Torch/CUDA RNG streams.
- Figure7/8 PDF one page each,7.2in width, rendered via Poppler and visually inspected. Every outcome retained, including large common-class outlier and adverse utility.
- Postprocessing local291 tests passed in18.69s; Linux291 passed in14.79s. Independent Linux/local report computations agree over1185 numeric fields, maximum absolute difference5.68e-14, nonnumeric structure identical. See results/cross_platform_verification.json and post_tests_linux.log.

## Repairs after training, not algorithm changes

Raise offline CSV field limit to64MiB; make output path explicit to preserve failed partial report; add fixed-bin means/uncertainty, prespecified stage grouping, raw prediction CSVs and separate actual-trajectory curves. Original source-score, mixture lambda, classification/BN scope, SGD, data and historical results are unchanged.

Recovery-helper review additionally found that the generic CPU copy loses `defaultdict` behavior. Explicit reconstruction for clients/profiles/tiers is now tested for previously unseen keys. This code path was never used in the completed runs. No real resumed training was performed. The source/config identity guard remains strict; applying a recovery-only patch to an old frozen checkpoint requires an explicit reviewed version migration, not silent resumption.

## Scientific status

|Question|Status|
|---|---|
|Read-only measurement and original A07 preservation|Verified|
|Actual nonneutral V2 fusion|Verified,304/500 rounds|
|Adapter participates in applied score|No; auxiliary only|
|Both tier Memories participate in U|Verified|
|Prototype/embedding cross-client utility guarantee|Unverified|
|U pooled association on all probe classes|Positive descriptive signal|
|Stable common-class/phase/grouped incremental information|Insufficient support|
|Independent receiver utility gain over same-input A07|Unconfirmed; interval spans0|
|Distinct Semantic Memory and Reliability benefit|Unconfirmed across controls/partitions|
|Full V2 trajectory beats A07|No in this seed1 pair|
|Final semantic algorithm scientific Go|No-Go for current frozen V2|

Reviewer-facing evidence: R1/R3 receive actual source-update utility records, effect/negative-control statistics and the explicit source-level attribution limit. R2 receives measured simulator costs, logical communication proxies, test-oracle labeling and the lack of matching full-algorithm convergence proof. Multi-seed generalization, SGD-independent utility data, causal Adapter benefit and robust double-Memory advantage remain unresolved. No additional seed, dataset, mechanism or training is scheduled.

See [Chinese research report](../../../SEMANTIC_A07_PAIR500_RESULTS.md), [scalar results](results/), [original failed report log](results/original_report_failure.log), and [full source evidence index](results/scalar_evidence_index.json). Checkpoints and83.5MB original scalar archive remain on the authorized server, with a verified local scalar copy; they are excluded from Git.
