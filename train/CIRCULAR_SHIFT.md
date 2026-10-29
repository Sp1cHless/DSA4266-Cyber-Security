# Four-model circular-shift retraining

For repeated paired trials, checked reuse of completed fits, restart support and
Web sparsity analysis, use the [multi-seed guide](CIRCULAR_SHIFT_MULTISEED.md).

This experiment compares XGBoost, LSTM, temporal CNN and CNN + LSTM under the
same fixed bucket rotation. A zero-based start of 5 changes the input order from
`[0,1,2,3,4,5,6,7,8,9]` to `[5,6,7,8,9,0,1,2,3,4]`. All 65 channels and the mask
move together, including empty buckets. Sequence labels, IDs, split ownership,
frozen embeddings, scaling and four-group training weights are preserved.

## Recommended paired batch

From the repository root, with the existing environment:

```powershell
.\.venv-cnn\Scripts\python.exe train/run_circular_shift.py
```

This runs **eight independent fits**, sequentially: chronological start 0 and
shifted start 5 for each family. Every fit starts from scratch. Default seed is
4266; each family retains its published architecture and training budget. The
same environment and device are used within each pair. Default `auto` uses CPU
for XGBoost and available CUDA for neural models, otherwise CPU. Use `--device
cpu` for an entirely CPU batch or `--device cuda` to request CUDA for all four.
With `--device mps`, neural models request MPS and XGBoost uses CPU; actual MPS
execution remains unverified on this Windows development machine.

The default batch deliberately trains new controls instead of mixing historical
runs from different PyTorch/CUDA environments. Completed experiments and model
weights are never overwritten. A fresh batch directory is created under
`train/runs/<unique_four_models_circular_start5_seed4266>/`, with:

```text
plan.json
status.json
xgboost/start0/   xgboost/start5/
lstm/start0/      lstm/start5/
cnn/start0/       cnn/start5/
hybrid/start0/    hybrid/start5/
COMPARISON.md
comparison.csv
comparison.json
per_attack_comparison.csv
```

The terminal prints the exact directory. Each model subdirectory contains the
usual model, configuration, history, validation scores, metrics and frozen
validation thresholds. Checkpoints and thresholds are selected separately per
run under the same validation policy; test shards are never opened in training
or batch summarization. The CSVs show AP, ROC-AUC, low-FPR recall, realized FPR,
macro attack recall and per-attack changes. Deltas are shifted minus control;
rate deltas are percentage points, while ROC-AUC delta is on its original scale.
No statistical significance test is implied.

## Alternatives

```powershell
# Four shifted fits only; baseline columns and paired deltas will be N/A.
python train/run_circular_shift.py --shift-only
# Select families or a different fixed start; runs fresh pairs by default.
python train/run_circular_shift.py --models cnn hybrid --circular-shift-start 3
# Portable CPU batch or a relocated identical dataset.
python train/run_circular_shift.py --device cpu --dataset "D:/shared/dataset"
# Regenerate a report from an existing completed batch, without any training.
python train/run_circular_shift.py --summarize "<batch-directory>"
```

`--out` must name a new directory. Summarization creates a fresh `reports/`
directory inside the batch by default. A failed fit leaves completed runs and
failure details in place; it does not silently skip failures or overwrite runs.
`--epochs`, `--batch-size`, `--rounds` and `--patience` override budgets only when
explicitly provided. Epoch/batch overrides apply to neural models, rounds to
XGBoost, and patience to all selected families. Defaults are recommended for the
experiment; artificial fixture checks use reduced budgets only to verify code.

Each family also supports an independent shifted run:

```powershell
python train/xgboost/train.py --circular-shift-start 5
python train/lstm/train.py --circular-shift-start 5
python train/cnn/train.py --circular-shift-start 5
python train/hybrid/train.py --circular-shift-start 5
```

Use start 0 for an independent chronological control. Omitted flags retain the
baseline unless an explicit config supplies a shift. Neural configs save the
start in their architecture; XGBoost saves it in the input-transform metadata
and in the Booster's native model attributes. Frozen `train/evaluate.py` replay
uses the stored rule automatically, including for existing shifted LSTM runs.
Always supply the original unrotated dataset to avoid a second rotation.

## Interpretation

This is a temporal perturbation diagnostic on existing real validation data.
All fits reuse the same original training cohort; rotation creates no new
independent examples. Neural models visit each training example once per epoch;
tree fitting uses the same weighted training population for every fit.

Training and validation both use the rotation, allowing each model to adapt.
Circular shifting preserves most adjacent pairs but changes the first/last
positions, removes one internal adjacency and adds the artificial `b9 -> b0`
adjacency. Similar scores do not prove that temporal order is irrelevant and do
not establish prediction invariance of a frozen chronological model.

For XGBoost, shifting flattened buckets is a fixed permutation of columns; it
does not remove the information available to a retrained tree model. Column
subsampling and split tie handling can still change a fitted result. CNN
padding boundaries and recurrent start/final states respond differently to the
same rotation. Inspect changes within each family and attack type before
comparing families. Use repeated seeds for stability claims. Validation selects
checkpoints and thresholds, and historical project work has already inspected
the existing test split; these are not fresh final test results.
