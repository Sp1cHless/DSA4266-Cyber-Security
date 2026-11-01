# Detector Comparison and Circular Rotation

Report sections: 5.2. Organized on 1 November 2026 from completed project records.

## Finding

Five-seed neural AP means are close. The original-order LSTM is retained as the working baseline, considering recall and seed variation. Five-bucket circular rotation gives small overall AP changes, while LSTM Web recall at the 1% FPR budget falls in every paired seed by 5.75 percentage points on average.

## Evidence scope

Four detector families on the fixed real validation cohort, seeds 4266-4270. Rotated models are refitted; channels and masks move together. Rotation preserves most local adjacency. It is not a proof that temporal order is irrelevant.

## Presentation

The current report uses concise prose or a consolidated table for this study. Earlier standalone figures and insertion drafts remain local rather than duplicating the main evidence.

## Compact results

- [five_seed_baseline.csv](<results/five_seed_baseline.csv>)
- [paired_shift_summary.csv](<results/paired_shift_summary.csv>)
- [provenance.json](<results/provenance.json>)

## Code and protocol entry points

- [train/README.md](<../../../train/README.md>)
- [train/CIRCULAR_SHIFT.md](<../../../train/CIRCULAR_SHIFT.md>)
- [train/CIRCULAR_SHIFT_MULTISEED.md](<../../../train/CIRCULAR_SHIFT_MULTISEED.md>)
- [train/run_circular_shift_multiseed.py](<../../../train/run_circular_shift_multiseed.py>)
- [analysis/compare_circular_shift_results.py](<../../../analysis/compare_circular_shift_results.py>)

## Local supporting records

`local/` preserves the original dated insertion drafts, plotting workspaces, source screenshots and superseded displays for this topic. It is ignored by Git. These files are historical support; the current DOCX determines final section placement. Full experiment runs, arrays and checkpoints retain their original paths under the ignored runtime directories. Source paths and hashes in captured provenance describe the original records, not bundled datasets.

## Saved evidence locations

These original records retain their existing paths; full runs and private evidence maps are intentionally local.

- `train/runs/20261024_104026_976396_projector_four_models_validation_ce128b/PROJECTOR_VIEWS.md`
- `train/runs/20261024_223102_671768_circular_multiseed_start5_ef34ba/reports/20261025_003115_565779_paired_multiseed_comparison_f43f00/aggregate.csv`
- `train/runs/20261024_223102_671768_circular_multiseed_start5_ef34ba/reports/20261025_003115_565779_paired_multiseed_comparison_f43f00/per_attack_aggregate.csv`
- `train/runs/20261026_holdout_lstm_projector_comparison/README.md`
- `train/runs/20261029_192945_155514_flooding_full_training_neighbors_56c0cf/FINDINGS.md`
