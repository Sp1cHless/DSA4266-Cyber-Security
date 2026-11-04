# Infilteration Input Ambiguity

## Finding

The historical LSTM detects 104 and misses 34 pure Infilteration validation sequences. Ten misses and ten detections have exact benign counterparts already in pre-encoder aggregates. Identical deterministic inputs share scores, giving a direct detection/false-alarm tradeoff for these pairs; other cases retain architecture-dependent recovery.

## Evidence scope

Post-selection raw-flow and exact-counterpart audit on inspected validation. Pure sequences contain only attack-labeled flows. No model or threshold is refitted; this is not independent test evaluation.

## Compact results

- [collision_summary.csv](results/collision_summary.csv)
- [neighbor_summary.csv](results/neighbor_summary.csv)
- [raw_behavior_summary.csv](results/raw_behavior_summary.csv)

[Shared single-seed comparison](../results/table_v1_targeted_comparison.csv).

## Code and protocol entry points

- [train/audit_pure_infilteration.py](<../../../../train/audit_pure_infilteration.py>)
- [train/audit_infilteration_label_conflicts.py](<../../../../train/audit_infilteration_label_conflicts.py>)
- [train/inspect_infilteration_space.py](<../../../../train/inspect_infilteration_space.py>)

## Local evidence

Completed runs and historical working material are consolidated in the ignored
`archive/` directory. Restore the matching inputs before rerunning a diagnostic
that requires saved arrays or checkpoints. See the [research index](../../README.md) for the result catalog and evidence boundaries.
