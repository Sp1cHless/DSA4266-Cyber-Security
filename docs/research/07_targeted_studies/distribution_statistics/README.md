# Distribution Statistics and Feature Probes

## Finding

Training-only feature probes motivate a fifteen-statistic extension. The resulting refitted pipeline gives similar overall validation AP (61.50% versus 61.53%), improves Web recall and reduces Infilteration recall. At the strict budget, 23 former Infilteration detections are lost and five are recovered.

## Evidence scope

Training-only probe development is distinct from the one-seed validation detector comparison. The feature extension includes a new bucket encoder, so the comparison evaluates the combined pipeline change.

## Compact results

- [attack_comparison.csv](results/attack_comparison.csv)
- [probe_results.csv](results/probe_results.csv)
- [probe_seed_aggregate.csv](results/probe_seed_aggregate.csv)

[Shared single-seed comparison](../results/table_v1_targeted_comparison.csv).

## Code and protocol entry points

- [analysis/INFILTERATION_FEATURE_AUDIT.md](<../../../../analysis/INFILTERATION_FEATURE_AUDIT.md>)
- [analysis/audit_infilteration_features.py](<../../../../analysis/audit_infilteration_features.py>)
- [analysis/recheck_infilteration_feature_probes.py](<../../../../analysis/recheck_infilteration_feature_probes.py>)
- [train/DISTRIBUTION_FEATURE_ABLATION.md](<../../../../train/DISTRIBUTION_FEATURE_ABLATION.md>)
- [train/run_distribution_feature_ablation.py](<../../../../train/run_distribution_feature_ablation.py>)
- [CIC_new/prepare_distribution_features.py](<../../../../CIC_new/prepare_distribution_features.py>)

## Local evidence

Completed runs and historical working material are consolidated in the ignored
`archive/` directory. Restore the matching inputs before rerunning a diagnostic
that requires saved arrays or checkpoints. See the [research index](../../README.md) for the result catalog and evidence boundaries.
