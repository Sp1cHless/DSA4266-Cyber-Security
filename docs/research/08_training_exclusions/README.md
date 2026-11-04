# Transfer After Training Exclusions

## Finding

Set A jointly excludes XSS, SlowHTTPTest and HOIC and retains about 99% pooled recall. Set B excludes all seven flooding primary labels and reduces pooled recall by 26.57-31.68 percentage points across model families. Separate one-seed geometry diagnostics distinguish the remaining subtype responses.

## Evidence scope

Two fixed primary-label exclusion sets, not an exhaustive leave-one-out rotation. Validation retains excluded types for selection; the historical test has been inspected before. Table H1 reproduces the committed team three-seed summary; its per-seed prediction arrays are not present locally. H2/H3 are separate one-seed analyses, and the full training-neighbor reference includes synthetic rows.

## Compact results

- [local_single_seed_attack_recall.csv](results/local_single_seed_attack_recall.csv)
- [local_single_seed_test.csv](results/local_single_seed_test.csv)
- [single_seed_flooding_geometry.csv](results/single_seed_flooding_geometry.csv)
- [single_seed_flooding_test.csv](results/single_seed_flooding_test.csv)
- [table_h1_three_seed_holdout.csv](results/table_h1_three_seed_holdout.csv)

## Code and protocol entry points

- [Training exclusion code and version boundaries](../../../train/TRAINING_EXCLUSIONS.md)
- [Restored team seed-report helper](../../../train/seed_report.py)
- [train/export_holdout_lstm_projector.py](<../../../train/export_holdout_lstm_projector.py>)
- [train/export_flooding_projector.py](<../../../train/export_flooding_projector.py>)
- [train/diagnose_flooding_projector.py](<../../../train/diagnose_flooding_projector.py>)

## Local evidence

Completed runs and historical working material are consolidated in the ignored
`archive/` directory. Restore the matching inputs before rerunning a diagnostic
that requires saved arrays or checkpoints. See the [research index](../README.md) for the result catalog and evidence boundaries.

The inherited `table_h1_three_seed_holdout.csv` covers the historical three-label
Set A and flooding Set B. It predates the later four-label Set A and Web-family
exclusion. Files beginning `local_single_seed_` are separate seed-4266 test
diagnostics for the supplied later datasets; they do not replace three-seed means.
