# Data Exploration

Report sections: 3. Organized on 1 November 2026 from completed project records.

## Finding

Full-source EDA supports source grouping, a sparse 100-second baseline with ten 10-second buckets, explicit masks and the study of natural attack/background mixtures. The source contains 20,115,529 flows; flow-level prevalence and attack-bearing-window prevalence use different denominators.

## Evidence scope

Descriptive full-source analysis before splitting and augmentation. These are data properties, not held-out detection results. Historical EDA window alignment is documented in the source provenance.

## Report figures

- EDA-1: [PNG](<figures/fig_eda_1_attack_support.png>) and [SVG](<figures/fig_eda_1_attack_support.svg>).
- EDA-2: [PNG](<figures/fig_eda_2_perspectives_and_dates.png>) and [SVG](<figures/fig_eda_2_perspectives_and_dates.svg>).
- EDA-3: [PNG](<figures/fig_eda_3_sparsity_and_scale.png>) and [SVG](<figures/fig_eda_3_sparsity_and_scale.svg>).
- EDA-4: [PNG](<figures/fig_eda_4_natural_background.png>) and [SVG](<figures/fig_eda_4_natural_background.svg>).

## Compact results

- [attack_type_inventory.csv](<results/attack_type_inventory.csv>)
- [daily_source_flows.csv](<results/daily_source_flows.csv>)
- [figure_provenance.json](<results/figure_provenance.json>)
- [occupancy_100s.csv](<results/occupancy_100s.csv>)
- [split_feasibility.csv](<results/split_feasibility.csv>)
- [verified_summary.json](<results/verified_summary.json>)
- [window_scale_summary.csv](<results/window_scale_summary.csv>)

## Code and protocol entry points

- [CIC_new/README.md](<../../../CIC_new/README.md>)
- [CIC_new/preprocessing/README.md](<../../../CIC_new/preprocessing/README.md>)
- [CIC_new/SCHEME2_DATASET.md](<../../../CIC_new/SCHEME2_DATASET.md>)

## Local supporting records

`local/` preserves the original dated insertion drafts, plotting workspaces, source screenshots and superseded displays for this topic. It is ignored by Git. These files are historical support; the current DOCX determines final section placement. Full experiment runs, arrays and checkpoints retain their original paths under the ignored runtime directories. Source paths and hashes in captured provenance describe the original records, not bundled datasets.

## Saved evidence locations

These original records retain their existing paths; full runs and private evidence maps are intentionally local.

- `_local_archive/cleanup_20261024_122355/retired/CIC_new/EDA/outputs/01_perspectives_20261018/results.json`
- `_local_archive/cleanup_20261024_122355/retired/CIC_new/EDA/outputs/02_sequence_lengths_20261018/results.json`
- `_local_archive/cleanup_20261024_122355/retired/CIC_new/EDA/outputs/03_features_20261018/results.json`
- `_local_archive/cleanup_20261024_122355/retired/CIC_new/reports/EDA/04_splits_20261018/results.json`
