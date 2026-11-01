# Infilteration Input Ambiguity

Report sections: 6.2. Organized on 1 November 2026 from completed project records.

## Finding

The historical LSTM detects 104 and misses 34 pure Infilteration validation sequences. Ten misses and ten detections have exact benign counterparts already in pre-encoder aggregates. Identical deterministic inputs share scores, giving a direct detection/false-alarm tradeoff for these pairs; other cases retain architecture-dependent recovery.

## Evidence scope

Post-selection raw-flow and exact-counterpart audit on inspected validation. Pure sequences contain only attack-labeled flows. No model or threshold is refitted; this is not independent test evaluation.

## Presentation

The current report uses concise prose or a consolidated table for this study. Earlier standalone figures and insertion drafts remain local rather than duplicating the main evidence.

The shared intervention comparison is [Table V1](<../results/table_v1_targeted_comparison.csv>).

## Code and protocol entry points

- [train/audit_pure_infilteration.py](<../../../../train/audit_pure_infilteration.py>)
- [train/audit_infilteration_label_conflicts.py](<../../../../train/audit_infilteration_label_conflicts.py>)
- [train/inspect_infilteration_space.py](<../../../../train/inspect_infilteration_space.py>)

## Local supporting records

`local/` preserves the original dated insertion drafts, plotting workspaces, source screenshots and superseded displays for this topic. It is ignored by Git. These files are historical support; the current DOCX determines final section placement. Full experiment runs, arrays and checkpoints retain their original paths under the ignored runtime directories. Source paths and hashes in captured provenance describe the original records, not bundled datasets.

## Saved evidence locations

These original records retain their existing paths; full runs and private evidence maps are intentionally local.

- `train/runs/20261024_122800_754357_pure_infilteration_raw_flow_audit_6146f6/audit.json`
- `train/runs/20261024_122902_891375_infilteration_training_signature_audit_fe6d6d`
