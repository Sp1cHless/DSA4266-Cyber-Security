# Auxiliary Bucket Supervision

## Finding

Existing bucket labels support learnable localization: the top-scoring bucket hits an attack position in 116/135 context-bearing Infilteration cases (85.9%), versus 54.4% expected under uniform selection. Active-bucket validation AP is 82.88%; low-FPR sequence gains remain modest.

## Evidence scope

One seed and fixed auxiliary loss weight 0.5, with the original inputs and frozen encoder. Bucket labels are targets only and bucket metrics do not select the sequence checkpoint. The localization count was checked from saved prediction arrays in the 30 October audit.

## Compact results

- [attack_comparison.csv](results/attack_comparison.csv)

[Shared single-seed comparison](../results/table_v1_targeted_comparison.csv).

## Code and protocol entry points

- [train/LSTM_BUCKET_SUPERVISION.md](<../../../../train/LSTM_BUCKET_SUPERVISION.md>)
- [train/run_lstm_bucket_ablation.py](<../../../../train/run_lstm_bucket_ablation.py>)
- [train/lstm_bucket/model.py](<../../../../train/lstm_bucket/model.py>)

## Local evidence

Completed runs and historical working material are consolidated in the ignored
`archive/` directory. Restore the matching inputs before rerunning a diagnostic
that requires saved arrays or checkpoints. See the [research index](../../README.md) for the result catalog and evidence boundaries.
