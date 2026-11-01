# Transfer After Training Exclusions

Report sections: 6.4. Organized on 1 November 2026 from completed project records.

## Finding

Set A jointly excludes XSS, SlowHTTPTest and HOIC and retains about 99% pooled recall. Set B excludes all seven flooding primary labels and reduces pooled recall by 26.57-31.68 percentage points across model families. Separate one-seed geometry diagnostics distinguish the remaining subtype responses.

## Evidence scope

Two fixed primary-label exclusion sets, not an exhaustive leave-one-out rotation. Validation retains excluded types for selection; the historical test has been inspected before. Table H1 reproduces the committed team three-seed summary; its per-seed prediction arrays are not present locally. H2/H3 are separate one-seed analyses, and the full training-neighbor reference includes synthetic rows.

## Report figures

- H2: [PNG](<figures/fig_set_a_projector_comparison.png>) and [SVG](<figures/fig_set_a_projector_comparison.svg>).
- H3: [PNG](<figures/fig_flooding_projector_diagnostics.png>) and [SVG](<figures/fig_flooding_projector_diagnostics.svg>).

## Compact results

- [figure_provenance.json](<results/figure_provenance.json>)
- [table_h1_three_seed_holdout.csv](<results/table_h1_three_seed_holdout.csv>)

## Code and protocol entry points

- [Training exclusion code and version boundaries](../../../train/TRAINING_EXCLUSIONS.md)
- [Restored team seed-report helper](../../../train/seed_report.py)
- [train/export_holdout_lstm_projector.py](<../../../train/export_holdout_lstm_projector.py>)
- [train/export_flooding_projector.py](<../../../train/export_flooding_projector.py>)
- [train/diagnose_flooding_projector.py](<../../../train/diagnose_flooding_projector.py>)

## Local supporting records

`local/` preserves the original dated insertion drafts, plotting workspaces, source screenshots and superseded displays for this topic. It is ignored by Git. These files are historical support; the current DOCX determines final section placement. Full experiment runs, arrays and checkpoints retain their original paths under the ignored runtime directories. Source paths and hashes in captured provenance describe the original records, not bundled datasets.

## Saved evidence locations

These original records retain their existing paths; full runs and private evidence maps are intentionally local.

- `docs/research_map_20261029/team_snapshot/HOLDOUT_PROTOCOL.md`
- `train/runs/20261026_holdout_lstm_projector_comparison/manifest.json`
- `train/runs/20261029_192411_050197_all_flooding_test_projector_699e44/manifest.json`
- `train/runs/20261029_192411_050197_all_flooding_test_projector_699e44/projector_flooding_selected.png`
- `train/runs/20261029_192945_155514_flooding_full_training_neighbors_56c0cf/projector_loic_http_fn.png`
- `train/runs/20261029_192945_155514_flooding_full_training_neighbors_56c0cf/summary.json`
