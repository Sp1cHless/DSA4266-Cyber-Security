# Data Exploration

## Finding

Full-source EDA supports source grouping, a sparse 100-second baseline with ten 10-second buckets, explicit masks and the study of natural attack/background mixtures. The source contains 20,115,529 flows; flow-level prevalence and attack-bearing-window prevalence use different denominators.

## Evidence scope

Descriptive full-source analysis before splitting and augmentation. These are data properties, not held-out detection results. Historical EDA window alignment is documented in the source provenance.

## Compact results

- [attack_type_inventory.csv](results/attack_type_inventory.csv)
- [daily_source_flows.csv](results/daily_source_flows.csv)
- [occupancy_100s.csv](results/occupancy_100s.csv)
- [split_feasibility.csv](results/split_feasibility.csv)
- [verified_summary.json](results/verified_summary.json)
- [window_scale_summary.csv](results/window_scale_summary.csv)

## Code and protocol entry points

- [CIC_new/README.md](<../../../CIC_new/README.md>)
- [CIC_new/preprocessing/README.md](<../../../CIC_new/preprocessing/README.md>)
- [CIC_new/SCHEME2_DATASET.md](<../../../CIC_new/SCHEME2_DATASET.md>)

## Local evidence

Completed runs and historical working material are consolidated in the ignored
`archive/` directory. Restore the matching inputs before rerunning a diagnostic
that requires saved arrays or checkpoints. See the [research index](../README.md) for the result catalog and evidence boundaries.
