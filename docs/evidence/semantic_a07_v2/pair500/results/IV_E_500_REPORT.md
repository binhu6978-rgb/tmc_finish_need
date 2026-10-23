# Frozen A07 / V2: 500-round mechanism evidence

Single seed1, official-test exploratory trajectories; training-side utility probes are disjoint from semantic measurement but NOT from SGD. CRD omitted by user.

|Method|test-oracle peak|Round|Final|Last20|Last50|Final CE|
|---|---:|---:|---:|---:|---:|---:|
|A07-only|39.9400|353|38.7100|38.8385|38.8360|3.542496|
|Semantic-A07 V2|39.1600|452|38.1200|38.2085|38.2054|3.566545|

Three separate questions must be read separately: score association (source-tier/round units), same-input counterfactual utility, and whole-trajectory Accuracy. CSVs preserve neutral and adverse cases. Memory ablations retain the full-V2 validity mask, so they do not establish the effect of removing all history. Adapter remains auxiliary. No score was tuned.

## neutral
Distinct semantic source-round units: 74 across 29 rounds; temporally and physically dependent. Matched-class Spearman=0.2861818728225851, clustered95=[-0.140660576988337, 0.6323657823943618]. All-class Spearman=0.5012736980568864.
Same-input V2 active-round Full CE utility=3.375559893242564e-07, clustered95=[-0.0001305245702602502, 0.00010849831239317383]. Applied nonneutral rounds=0/500. See diagnostic_summary.json for current/memory/reliability ablations and held-out-round/receiver-group prediction errors.

## active
Distinct semantic source-round units: 74 across 29 rounds; temporally and physically dependent. Matched-class Spearman=0.24405017560281167, clustered95=[-0.13220259522651767, 0.6144087593621022]. All-class Spearman=0.45274943372495635.
Same-input V2 active-round Full CE utility=-7.684787410974493e-06, clustered95=[-0.000158722447515189, 8.719991413029121e-05]. Applied nonneutral rounds=304/500. See diagnostic_summary.json for current/memory/reliability ablations and held-out-round/receiver-group prediction errors.

These observations do not establish multi-seed generalization. Temporal and physical-client dependence, training-side evaluation bias and source-versus-region attribution remain explicit limitations. No further run is scheduled.