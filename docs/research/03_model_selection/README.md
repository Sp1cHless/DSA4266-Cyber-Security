# Detector Comparison and Circular Rotation

## Finding

Five-seed neural AP means are close. The original-order LSTM is retained as the working baseline, considering recall and seed variation. Five-bucket circular rotation gives small overall AP changes, while LSTM Web recall at the 1% FPR budget falls in every paired seed by 5.75 percentage points on average.

## Evidence scope

Four detector families on the fixed real validation cohort, seeds 4266-4270. Rotated models are refitted; channels and masks move together. Rotation preserves most local adjacency. It is not a proof that temporal order is irrelevant.

## Compact results

- [aggregate.csv](results/aggregate.csv)
- [five_seed_baseline.csv](results/five_seed_baseline.csv)
- [paired_shift_summary.csv](results/paired_shift_summary.csv)
- [per_attack_aggregate.csv](results/per_attack_aggregate.csv)
- [per_seed.csv](results/per_seed.csv)
- [provenance.json](results/provenance.json)
- [single_seed_model_comparison.csv](results/single_seed_model_comparison.csv)

## Code and protocol entry points

- [train/README.md](<../../../train/README.md>)
- [train/CIRCULAR_SHIFT.md](<../../../train/CIRCULAR_SHIFT.md>)
- [train/CIRCULAR_SHIFT_MULTISEED.md](<../../../train/CIRCULAR_SHIFT_MULTISEED.md>)
- [train/run_circular_shift_multiseed.py](<../../../train/run_circular_shift_multiseed.py>)
- [analysis/compare_circular_shift_results.py](<../../../analysis/compare_circular_shift_results.py>)

## Local evidence

Completed runs and historical working material are consolidated in the ignored
`archive/` directory. Restore the matching inputs before rerunning a diagnostic
that requires saved arrays or checkpoints. See the [research index](../README.md) for the result catalog and evidence boundaries.
