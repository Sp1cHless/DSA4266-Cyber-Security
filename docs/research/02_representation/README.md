# Representation and Scheme Selection

Report sections: 4 and 5.1. Organized on 1 November 2026 from completed project records.

## Finding

Bucket L2 improves Scheme 1 validation AP from 3.44% to 10.40%; Scheme 2 reaches 19.65% in the temporal-autoencoder/linear-readout comparison. Scheme 2 uses 64 learned bucket coordinates plus a separate intensity channel and becomes the shared supervised input.

## Evidence scope

Complete representation pipelines at seed 4266, with paired original IDs, splits and donor plans. The autoencoder/readout comparison has a different objective from later supervised detector training. Historical test results use validation-selected thresholds.

## Report figures

- R1: [PNG](<figures/fig_repr_1_scheme2_pipeline.png>) and [SVG](<figures/fig_repr_1_scheme2_pipeline.svg>).
- R2: [PNG](<figures/fig_repr_2_normalization_diagnostic.png>) and [SVG](<figures/fig_repr_2_normalization_diagnostic.svg>).

## Compact results

- [figure_provenance.json](<results/figure_provenance.json>)
- [reconstruction_error_concentration.csv](<results/reconstruction_error_concentration.csv>)
- [representation_comparison.csv](<results/representation_comparison.csv>)

## Code and protocol entry points

- [CIC_new/SCHEME2_DATASET.md](<../../../CIC_new/SCHEME2_DATASET.md>)
- [CIC_new/representation_learning/README.md](<../../../CIC_new/representation_learning/README.md>)
- [CIC_new/prepare_scheme2_dataset.py](<../../../CIC_new/prepare_scheme2_dataset.py>)

## Local supporting records

`local/` preserves the original dated insertion drafts, plotting workspaces, source screenshots and superseded displays for this topic. It is ignored by Git. These files are historical support; the current DOCX determines final section placement. Full experiment runs, arrays and checkpoints retain their original paths under the ignored runtime directories. Source paths and hashes in captured provenance describe the original records, not bundled datasets.

## Saved evidence locations

These original records retain their existing paths; full runs and private evidence maps are intentionally local.

- `CIC_new/SCHEME2_DATASET.md`
- `CIC_new/representation_learning/README.md`
- `_local_archive/cleanup_20261024_122355/retired/CIC_new/reports/superimposition_comparison_20261022/evidence/ae_runs/20261022_110605_010212_scheme1_ae_aug_e500_b128_seed4266_ihye4k0n/evaluations/20261022_113351_validation_diagnostics_oxa2qrqr/report.json`
- `_local_archive/cleanup_20261024_122355/retired/CIC_new/reports/superimposition_comparison_20261022/evidence/ae_runs/20261022_170206_140989_scheme1_l2_ae_aug_e500_b128_seed4266_iu6xj7xz/evaluations/20261022_173556_scheme1_l2_comparison_wcxasexx/report.json`
- `_local_archive/cleanup_20261024_122355/retired/CIC_new/reports/superimposition_comparison_20261022/evidence/ae_runs/20261022_170206_140989_scheme1_l2_ae_aug_e500_b128_seed4266_iu6xj7xz/evaluations/20261022_173556_scheme1_l2_comparison_wcxasexx/three_way_comparison.json`
- `_local_archive/cleanup_20261024_122355/retired/CIC_new/reports/superimposition_comparison_20261022/test_metrics.json`
- `datasets/dataset/validation.json`
