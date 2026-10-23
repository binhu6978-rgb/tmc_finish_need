# IV-E complete500 mechanism evidence

The complete Chinese research report is [SEMANTIC_A07_PAIR500_RESULTS](../../../../SEMANTIC_A07_PAIR500_RESULTS.md). The compact automatic table is [IV_E_500_REPORT](IV_E_500_REPORT.md).

Both real trajectories completed500 rounds. A07 test-oracle peak39.94%, Last20=38.8385%; V2 peak39.16%, Last20=38.2085%. The current frozen V2 receives a scientific No-Go: source-score association, same-input utility and full-trajectory performance are evaluated separately, with all adverse results retained.

- [Figure7 PDF](figure7_semantic_utility.pdf): source-round observations, both category scopes, fixed-bin means and clustered uncertainty.
- [Figure8 PDF](figure8_aggregation_behavior.pdf): regional source-share changes, same-input controls, actual Full learning curves and fixed-window utility.
- [Full Accuracy/CE curves](accuracy_ce.png), [performance table](performance_comparison.csv).
- Raw figure CSVs and prediction CSVs are under [A07 trajectory](neutral/) and [V2 trajectory](active/). Both include500-round Accuracy/CE, source provenance and committed evidence identities.
- [Statistics](summary.json), [past-only replay and cost audit](offline_audit.json), [training completion](completion_audit.json), [checkpoint and historical-A07 audit](recovery_and_historical_audit.json).
- [Original scalar index](scalar_evidence_index.json) references the full server archive. [Original report failure](original_report_failure.log) is preserved; it is a CSV parser failure after training completion.

Replot only, with no training:

```bash
python tools/plot_semantic_pair500.py docs/evidence/semantic_a07_v2/pair500/results
```

Primary confirmation uses74 distinct tier-round observations over29 rounds. Repeated recipients/regions are dependent responses. Utility probes are disjoint from semantic scoring but still SGD training-side. Only seed1 was run. No next experiment is scheduled.
