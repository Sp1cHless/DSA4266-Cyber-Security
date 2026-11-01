# Auxiliary Bucket Supervision

Report sections: 6.3. Organized on 1 November 2026 from completed project records.

## Finding

Existing bucket labels support learnable localization: the top-scoring bucket hits an attack position in 116/135 context-bearing Infilteration cases (85.9%), versus 54.4% expected under uniform selection. Active-bucket validation AP is 82.88%; low-FPR sequence gains remain modest.

## Evidence scope

One seed and fixed auxiliary loss weight 0.5, with the original inputs and frozen encoder. Bucket labels are targets only and bucket metrics do not select the sequence checkpoint. The localization count was checked from saved prediction arrays in the 30 October audit.

## Presentation

The current report uses concise prose or a consolidated table for this study. Earlier standalone figures and insertion drafts remain local rather than duplicating the main evidence.

## Compact results

- [figure_provenance.json](<results/figure_provenance.json>)

The shared intervention comparison is [Table V1](<../results/table_v1_targeted_comparison.csv>).

## Code and protocol entry points

- [train/LSTM_BUCKET_SUPERVISION.md](<../../../../train/LSTM_BUCKET_SUPERVISION.md>)
- [train/run_lstm_bucket_ablation.py](<../../../../train/run_lstm_bucket_ablation.py>)
- [train/lstm_bucket/model.py](<../../../../train/lstm_bucket/model.py>)

## Local supporting records

`local/` preserves the original dated insertion drafts, plotting workspaces, source screenshots and superseded displays for this topic. It is ignored by Git. These files are historical support; the current DOCX determines final section placement. Full experiment runs, arrays and checkpoints retain their original paths under the ignored runtime directories. Source paths and hashes in captured provenance describe the original records, not bundled datasets.

## Saved evidence locations

These original records retain their existing paths; full runs and private evidence maps are intentionally local.

- `docs/RESEARCH_FINDINGS.md`
- `train/runs/20261025_221955_112858_lstm_bucket_supervision_seed4266_b1f2de/attack_comparison.csv`
- `train/runs/20261025_221955_112858_lstm_bucket_supervision_seed4266_b1f2de/bucket_supervised/bucket_metrics.json`
- `train/runs/20261025_221955_112858_lstm_bucket_supervision_seed4266_b1f2de/comparison.json`
