# LSTM context and attention-pooling ablation

Use the existing environment from the repository root. No installation is needed.

```powershell
.\.venv-cnn\Scripts\python.exe -u train/run_lstm_context_ablation.py --dry-run
.\.venv-cnn\Scripts\python.exe -u train/run_lstm_context_ablation.py
```

The default reference is the completed chronological LSTM at seed 4266 from the
matched five-seed circular-shift study. It uses the current installed runtime,
unlike the older team checkpoint. Use `--reference-run PATH` to select another
completed chronological LSTM reference, including a different window setting.

Only the uniform masked mean of LSTM states changes to learned attention.
Masked max pooling, final hidden state, input channels, bucket encoder, training
groups, optimizer, batch size, seed, validation, and checkpoint/threshold policies
are retained. The original maximum of 30 epochs and patience 5 are copied from
the reference. No smaller trial budget is imposed. No preprocessing is run.

Attention uses a linear 64-to-1 scoring layer: 65 additional parameters. Its
weights start at zero, so the initial readout equals uniform mean pooling.
The LSTM and classifier train from scratch. Attention uses no bucket labels.

New outputs receive unique directories under `train/runs/`. Existing datasets
and weights are read only. `attack_comparison.csv` contains Infilteration and
Brute Force Web TP/FN and recall at both original global FPR budgets;
`comparison.json` also contains AP and parameter counts. Every model selects
its own global validation thresholds under the same policy.

This is an exploratory comparison reusing the original training and validation
data. It does not score test data. A single seed does not establish a general
improvement, and historical project work had already inspected the test split.

## Frozen diagnostics

```powershell
.\.venv-cnn\Scripts\python.exe -u analysis/diagnose_lstm_context.py
```

This separately checks existing weights across the historical baseline, ten
paired circular-shift fits and four new window settings. Oracle interventions
use true bucket labels to restrict pooling or remove pure-benign buckets.
They are mechanism stress diagnostics, never deployable detector results.
Only the last-active-state intervention is label free. Benign subsampling is
disclosed and does not provide a calibrated full-validation FPR estimate.

The frozen diagnostic was executed. The attention-pooling model has only been
checked on artificial data and has not been trained on the real dataset yet.
