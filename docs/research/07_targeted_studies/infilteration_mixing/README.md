# Infilteration Mixing Diagnostics

## Finding

The largest post hoc strict-threshold loss groups have different responses: TCP/3389 cases have low median attack-flow share and largely cross the threshold after benign-flow removal, while attack-heavy UDP/53 cases largely remain below threshold under the extension. Similar overall AP conceals different responses to attack evidence.

## Evidence scope

Post hoc groups among 23 losses after the distribution extension. Oracle deletions use flow labels and unchanged per-pipeline thresholds; intervention FPR is not recalibrated. These are diagnostic responses, not detection gains.

## Compact results

- [flow_family_profiles.csv](results/flow_family_profiles.csv)
- [group_profiles.csv](results/group_profiles.csv)

[Shared single-seed comparison](../results/table_v1_targeted_comparison.csv).

## Code and protocol entry points

- [analysis/inspect_distribution_infilteration_cases.py](<../../../../analysis/inspect_distribution_infilteration_cases.py>)
- [analysis/review_distribution_feature_results.py](<../../../../analysis/review_distribution_feature_results.py>)

## Local evidence

Completed runs and historical working material are consolidated in the ignored
`archive/` directory. Restore the matching inputs before rerunning a diagnostic
that requires saved arrays or checkpoints. See the [research index](../../README.md) for the result catalog and evidence boundaries.
