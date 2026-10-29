# LSTM baseline

Input: frozen Scheme2 `[N,10,65]` sequences plus the explicit bucket mask.
One 64-unit LSTM followed by masked mean/max pooling, its final hidden state,
and a 64-unit binary classification head. Dropout 0.2. Empty time bins remain
on the original grid; the recurrent state can evolve through them.

From the repository root:

```bash
python train/lstm/train.py
# Explicit CPU mode works on macOS, Windows and Linux.
python train/lstm/train.py --device cpu
```

Run only this assigned model on your own machine. Default `auto` uses CUDA only
when available and otherwise CPU, including on a Mac. Optional `--device mps`
requests an available Apple GPU; actual MPS training needs verification on the Mac.
Share the completed run folder with the coordinator; local caches are unnecessary.

Defaults: `datasets/dataset`, batch 512, AdamW lr 0.001, at most 30 epochs,
early stopping patience 5 on complete real-validation AP. `config.json` contains
the architecture and optimization settings. Runs go to a fresh `runs/` directory
under this folder. See [the shared guide](../README.md) for installation, weighting,
metrics, output files, all-model comparison and frozen evaluation.

For the main-baseline comparison against four new sequence/bucket settings, use
[`run_temporal_lstm.py`](../TEMPORAL_LSTM.md). It reuses completed preparations,
retains this LSTM's archived training settings, and highlights Infilteration and
Brute Force Web validation recall at 1% and 0.1% FPR without scoring test.

## Fixed circular-shift experiment

The same operation is supported for all four families. See the
[four-model paired batch guide](../CIRCULAR_SHIFT.md) to run eight matched fits
and produce AP, low-FPR recall and per-attack delta tables automatically.

To test sensitivity to the sequence start/end boundary, retrain the same LSTM
with a fixed rotation of all ten buckets. The start index is **zero-based**:
`--circular-shift-start 5` uses the middle bucket (the sixth bucket) as the new start:

```text
Original: b0 b1 b2 b3 b4 b5 b6 b7 b8 b9
Shifted:  b5 b6 b7 b8 b9 b0 b1 b2 b3 b4
```

From the repository root, using the same environment and device for both runs:

```powershell
# Chronological control, freshly trained with the same seed and settings.
python train/lstm/train.py --circular-shift-start 0 --seed 4266
# Fixed middle-start circular shift, trained from scratch.
python train/lstm/train.py --circular-shift-start 5 --seed 4266
```

Append `--device cpu` for CPU or `--device cuda` for an available NVIDIA GPU.
Other starts from 0 through 9 are supported; omitting the flag keeps the default
chronological baseline, unless an explicit `--config` supplies an architecture
`circular_shift_start` value. The command-line flag overrides that value.
Do not choose the best offset after inspecting validation results and call it
an independent estimate of temporal importance.

All 65 channels and the bucket mask rotate together, including empty buckets.
No bucket is dropped, interpolated, shuffled internally or moved across samples.
Sequence labels, sample IDs, train/validation ownership and four-group weights
stay the same. No new samples are added: this experiment explicitly reuses the
original training cohort, visiting each sample once per epoch. The fixed rotation
is identical for every sample and epoch and for train and real validation.

The bucket embeddings remain frozen. LSTM initialization, architecture size,
batch order, loss and optimization defaults match the control for a given seed.
Checkpoint selection and threshold fitting still use real validation AP and
scores only. Training never reads test shards. Both commands create fresh run
folders; shifted folder names contain `circular_start5` (or the requested offset).
The input order is recorded in `config.json`, `data_audit.json` and `summary.json`.
The checkpoint architecture stores the start index, so `train/evaluate.py --run
<shifted-run> --split validation` automatically repeats the saved rotation.
Pass original dataset shards to evaluation; do not pre-rotate them a second time.

Compare the two runs' `metrics.json` / `summary.json`: AP, ROC-AUC, recall at
validation FPR budgets 1% and 0.1%, and per-attack recall. Each run selects its own
checkpoint and validation thresholds under the same policy. `train/compare.py`
is for architecture screening and rejects mixed input transforms; it is not a
same-LSTM ablation comparator.

These are temporal perturbation diagnostics on the existing real validation
cohort, not chronological-input performance or final test results. A circular
shift retains most adjacent bucket pairs but removes one original internal
adjacency, introduces the artificial `b9 -> b0` adjacency, and changes the LSTM's
initial/final positions. A difference measures sensitivity to these effects
together; a small difference does not prove that temporal order is irrelevant.
One seed is exploratory and does not establish statistical significance.
