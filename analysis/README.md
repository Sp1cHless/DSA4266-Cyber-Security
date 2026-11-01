# Research Diagnostics

These scripts keep their existing locations so imports, recorded commands and
source provenance remain stable. Browse findings through the
[research index](../docs/research/README.md) and every implementation resource
through the [code index](../docs/code/README.md).

| Study | Script | Role |
|---|---|---|
| Model selection | [compare_circular_shift_results.py](compare_circular_shift_results.py) | Compare completed paired rotation results |
| Web evidence | [diagnose_web_temporal_structure.py](diagnose_web_temporal_structure.py) | Occupancy, complementarity and temporal support |
| Attention/context | [diagnose_lstm_context.py](diagnose_lstm_context.py) | Frozen oracle context interventions |
| Distribution statistics | [audit_infilteration_features.py](audit_infilteration_features.py) | Training-only feature reconstruction and probe screening |
| Probe repeatability | [recheck_infilteration_feature_probes.py](recheck_infilteration_feature_probes.py) | Forest-seed rechecks on the saved internal partitions |
| Distribution comparison | [review_distribution_feature_results.py](review_distribution_feature_results.py) | Completed sequence/subgroup comparisons |
| Infilteration mixing | [inspect_distribution_infilteration_cases.py](inspect_distribution_infilteration_cases.py) | Changed-case flow families and oracle responses |

The feature audit has a dedicated [protocol](INFILTERATION_FEATURE_AUDIT.md).
Additional raw-flow audits and representation exports live under `train/` and
are routed by [the code guide](../docs/code/README.md). These are research
diagnostics with specific data scopes; their outputs are not all held-out test
metrics. Full generated runs remain local and ignored.
