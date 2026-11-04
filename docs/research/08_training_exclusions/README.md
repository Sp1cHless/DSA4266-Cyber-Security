# Phase II: Transfer After Training Exclusions

## Final report finding

LSTM pooled held-out recall at validation-calibrated 1% FPR thresholds is
98.57% for Set A, 72.94% for B-1 and 12.12% for B-2. Set A excludes XSS,
SlowHTTPTest, HOIC and SSH while their related types remain; B-1 excludes all
flooding types and B-2 all Web types. The final report treats FTP and SSH as
brute-force siblings. Results use seeds 4266-4268 on fixed historical-test samples.

## Canonical implementation

- [Independent holdout workflow](../../../experiments/holdout/README.md).
- [Protocol and exact label sets](../../../experiments/holdout/CIC_new/HOLDOUT_PROTOCOL.md).
- [Builder](../../../experiments/holdout/CIC_new/prepare_scheme2_dataset.py).
- [Single-pair report](../../../experiments/holdout/train/unseen_subtype_report.py).
- [Three-seed report](../../../experiments/holdout/train/seed_report.py).

This implementation is retained separately from root Phase I/temporal code.
Validation retains excluded types for selection. Sequence exclusion uses primary
labels; non-primary residual flows are audited separately.

## Final report tables

- [Table VI summary](results/final_report_holdout_summary.csv).
- [Tables VII-IX subtype results](results/final_report_holdout_subtypes.csv).
- [Source hash and transcription scope](results/final_report_provenance.json).

Values preserve the final PDF's printed precision. They are report transcriptions,
not independent rescoring from raw predictions.

## Separate historical support

- [Earlier three-subtype/four-model summary](results/table_h1_three_seed_holdout.csv): predates the final Set A; keep its population distinct.
- [Later local single-seed test metrics](results/local_single_seed_test.csv).
- [Later local single-seed subtype recalls](results/local_single_seed_attack_recall.csv).
- [Historical single-seed flooding test](results/single_seed_flooding_test.csv).
- [Historical single-seed flooding geometry](results/single_seed_flooding_geometry.csv).

Root `train/export_holdout_lstm_projector.py`, `export_flooding_projector.py` and
`diagnose_flooding_projector.py` are separate geometry diagnostics. Local complete
runs and arrays are archived. No real-data model or threshold was refitted during
repository integration.
