# NB15 source-host holdout control

This is a paired **CNN autoencoder**, not a supervised classifier. It is a
diagnostic reproduction of the team pipeline, not a rerun of the temporal-bucket
control or the CIC-IDS2018 experiment. Existing project files are not replaced.

## What stays fixed

- Team source pinned at `origin/AE` commit
  `91f5ecb1d093435e0211277afa7ae09a5157428c`; unchanged copies are in `team_snapshot/`.
- Raw data: `datasets/NF-UNSW-NB15-v3.csv`; all rows, no new deduplication.
- Current `--drop-shortcuts` policy: six shortcuts removed, corresponding to
  the current code / `noshort2` configuration, not the older four-column variant.
- Original relative-time features, categorical encoding and destination-port
  indicators. **No new port removal and no IP address added as a feature.**
- Per-source-IP, start-time-ordered windows: 20 flows, stride 10, no padding;
  original one-hour maximum-span filter. Stride 10 is the code default; the
  exact stride used for the team's older saved metrics cannot be established
  from those metric tables alone.
- Original CNN architecture, approximately 250k parameters, latent size 32;
  seed 0, Adam 0.001, batch 256, at most 50 epochs, patience 5. Benign-only
  training and benign-validation early stopping; original masked MSE loss.
- Anomaly score is reconstruction error. Thresholds are frozen from benign
  validation, never from test. No test-driven tuning or feature deletion.

## What changes between the two arms

1. `time`: original chronological 60/20/20 window split and boundary guard;
   original benign-flow time-cutoff rule for scaler/imputation fitting.
2. `host`: disjoint **source IPs** in train, validation and test, and benign
   training-host-only scaler/imputation fitting. Allocation is declared before
   scoring: fixed seed, stratified by source /24 and homogeneous host label,
   approximately 60/20/20 hosts. Strata with fewer than three hosts stay in
   train. Four attack sources are allocated 2/1/1; training attacks are ignored
   by the AE as in the original pipeline. Host fractions need not equal flow
   or window fractions.

Destination endpoints can overlap; this does **not** claim that every device
appearing on either side of a connection is held out. `split_audit.json` records
the actual source-host intersections, fitting hosts and absent attack types.

For faithful comparison, full-data **unlabeled** vocabulary, dead-bit and
missing-indicator selection are retained in both arms, as are the original
relative-time definitions. Thus the host arm is a model/scaler source-host
holdout, **not fully inductive preprocessing**. Changing these policies as well
would require a separate control. A lazy window adapter avoids a multi-GB
expanded tensor; a self-test checks exact agreement with the original window
builder and anomaly scorer.

## Run from the repository root (PowerShell)

Use the existing `.venv-cnn` environment. This wrapper needs numpy, pandas,
scikit-learn, joblib and torch; it does not require parquet/pyarrow.

```powershell
.\.venv-cnn\Scripts\python.exe NB15_host_control\run_control.py --self-test
.\.venv-cnn\Scripts\python.exe -u NB15_host_control\run_control.py
.\.venv-cnn\Scripts\python.exe NB15_host_control\summarize_results.py
```

The default output is `NB15_host_control/artifacts/seed0_w20s10/`.
To repeat, select a fresh output directory (existing output is never overwritten):

```powershell
.\.venv-cnn\Scripts\python.exe -u NB15_host_control\run_control.py --out NB15_host_control/artifacts/repeat_seed0
```

`comparison.csv` contains both validation and independent test metrics.
`RESULTS.md` includes an audited comparison and metrics on the exact same
test-window intersection; this fixes evaluation composition, not differences
in training populations or time coverage.
Each arm also saves the checkpoint, scaler, full split metadata, individual
scores, per-host statistics, per-attack recall/intervals and subgroup AUCs.
The reported `pr_auc` is sklearn **average precision**, matching the original
team evaluator, not trapezoidal integration of the PR curve.

## Interpretation boundary

Previously saved team ablation numbers were validation-selection results;
do not describe them as independent test scores. This paired reproduction
reports both explicitly. Its source-host split also changes time coverage,
attack-source composition and sample counts. A single seed can establish a
failure or robustness **in this split**, but cannot prove causal memorization
of IPs, universal unseen-host generalization, or cross-dataset performance.
Inspect 59.* versus 149.* benign-family metrics, actual test FPR and missing
attack types rather than overall ROC-AUC alone. Family-specific AP has a
different attack prevalence and is not directly comparable to overall AP.
