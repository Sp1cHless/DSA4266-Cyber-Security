# Past Context and Temporal Scale

Report sections: 6.1. Organized on 1 November 2026 from completed project records.

## Finding

Longer history adds activity for some anchors, while a sparse Web subgroup persists. Rebuilt temporal geometries change subtype coverage and operating-point behavior. The 400/40 configuration has 117 Web and only 45 Infilteration sequences, compared with 181 and 366 in the historical 100/10 reference.

## Evidence scope

Common-anchor raw-flow availability is distinct from detector results on rebuilt cohorts. Four new single-seed LSTM fits use refitted preprocessing and encoders; the historical reference uses a different runtime. Keep both FPR budgets and sample counts visible.

## Report figures

- T1: [PNG](<figures/fig_temporal_context_resolution.png>) and [SVG](<figures/fig_temporal_context_resolution.svg>).

## Compact results

- [common_anchor_activity.csv](<results/common_anchor_activity.csv>)
- [figure_provenance.json](<results/figure_provenance.json>)
- [temporal_results.csv](<results/temporal_results.csv>)

## Code and protocol entry points

- [train/TEMPORAL_LSTM.md](<../../../../train/TEMPORAL_LSTM.md>)
- [train/run_temporal_lstm.py](<../../../../train/run_temporal_lstm.py>)
- [train/diagnose_window_context.py](<../../../../train/diagnose_window_context.py>)
- [CIC_new/TEMPORAL_ABLATION.md](<../../../../CIC_new/TEMPORAL_ABLATION.md>)
- [train/TEMPORAL_CNN.md](<../../../../train/TEMPORAL_CNN.md>)

## Local supporting records

`local/` preserves the original dated insertion drafts, plotting workspaces, source screenshots and superseded displays for this topic. It is ignored by Git. These files are historical support; the current DOCX determines final section placement. Full experiment runs, arrays and checkpoints retain their original paths under the ignored runtime directories. Source paths and hashes in captured provenance describe the original records, not bundled datasets.

## Saved evidence locations

These original records retain their existing paths; full runs and private evidence maps are intentionally local.

- `train/TEMPORAL_LSTM.md`
- `train/runs/20261024_124830_784309_window_context_availability_0ce90b/paired_summary.csv`
- `train/runs/20261025_013446_294043_web_temporal_support_diagnostic_7c620b/past_context_comparison.csv`
- `train/runs/20261025_100701_778931_lstm_temporal_seed4266_d278c9/attack_comparison.csv`
- `train/runs/20261025_100701_778931_lstm_temporal_seed4266_d278c9/validation_comparison.csv`
