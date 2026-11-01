# Attack Type Distributions

Report sections: 5.3. Organized on 1 November 2026 from completed project records.

## Finding

Web and Infilteration occupy broader, locally mixed baseline representation regions than the compact DDoS groups. Mean benign shares among ten neighbors in unit-normalized native 64D space are 20.72% and 24.81%, versus 0% for each of the three DDoS types.

## Evidence scope

Matched attack-enriched Projector cohort of 4,917 validation samples. Screenshot PCA and native-dimensional neighbor statistics are separate measurements. The full-source address counts motivate the host study; counts alone do not establish the cause of detection performance.

## Report figures

- D1: [PNG](<figures/fig_attack_distributions.png>) and [SVG](<figures/fig_attack_distributions.svg>).

## Compact results

- [figure_provenance.json](<results/figure_provenance.json>)
- [native_neighbor_summary.csv](<results/native_neighbor_summary.csv>)

## Code and protocol entry points

- [train/diagnose_lstm_attack_overlap.py](<../../../train/diagnose_lstm_attack_overlap.py>)
- [train/report_attack_types.py](<../../../train/report_attack_types.py>)
- [train/export_projector.py](<../../../train/export_projector.py>)

## Local supporting records

`local/` preserves the original dated insertion drafts, plotting workspaces, source screenshots and superseded displays for this topic. It is ignored by Git. These files are historical support; the current DOCX determines final section placement. Full experiment runs, arrays and checkpoints retain their original paths under the ignored runtime directories. Source paths and hashes in captured provenance describe the original records, not bundled datasets.

## Saved evidence locations

These original records retain their existing paths; full runs and private evidence maps are intentionally local.

- `train/runs/20261024_104026_976396_projector_four_models_validation_ce128b/PROJECTOR_VIEWS.md`
- `train/runs/20261024_104026_976396_projector_four_models_validation_ce128b/lstm/metadata.tsv`
- `train/runs/20261024_104026_976396_projector_four_models_validation_ce128b/lstm/vectors.tsv`
- `train/runs/20261024_114328_990393_lstm_attack_overlap_diagnostics_219c9c/DIAGNOSTICS.md`
- `train/runs/20261024_114328_990393_lstm_attack_overlap_diagnostics_219c9c/attack_profiles.csv`
- `train/runs/20261024_114328_990393_lstm_attack_overlap_diagnostics_219c9c/equal_n50_resampling.csv`
