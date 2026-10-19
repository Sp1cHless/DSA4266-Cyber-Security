# Optional downstream models

Preprocessing is complete before running anything in this folder. The shared
input is an exported dataset containing X, masks, labels and provenance. A CNN,
LSTM, classifier or another model can use the same arrays. No classifier or LSTM
implementation is claimed here; the temporal AE is the supplied example.

`autoencoder.py` defines only the temporal CNN AE and masked reconstruction loss.
`train_autoencoder.py` loads the **already exported dataset**, samples batches,
fits that AE and evaluates real validation examples. It does not construct raw
sequences, aggregate buckets, split data, fit input embeddings or create synthetic
samples. Those are separate steps and modules.

```bash
python -u -m CIC_new.modelling.train_autoencoder \
  --dataset CIC_new/outputs/model_inputs \
  --out CIC_new/outputs/temporal_ae_augmented \
  --device cuda --epochs 10 --steps-per-epoch 200 --batch-size 32
```

For a smoke test use `--epochs 1 --steps-per-epoch 10 --val-limit 128`.
The example uses four-way balanced batches; changing the class sampling prior is
a modelling decision. A real-only control uses the same exported dataset with
`--real-only`, a fresh output directory, the same seed and update budget. It samples
only the real groups. Thus both models share the same embeddings, scaler and
validation selection. Train for the same number of optimiser updates when comparing.

To compare zero-shift and circular-shift augmentation, create different exported
datasets with `export_augmented_dataset --shift-mode none/circular`, using the
same seed. The trainer itself does not transform or rotate samples.

An epoch here is a fixed update budget, not a complete original-data pass. Both
classes are reconstructed; error is not a benign-only anomaly score. Loss is
feature-averaged MSE over observed buckets, with each sequence equally weighted.
The fixed real validation sample keeps natural class proportions; per-class MSE
is reported because overall validation may be dominated by benign traffic.
Test is not read for training or checkpoint selection.

Outputs: `history.csv`, `best.pt`, `last.pt`, input manifest/transform copies,
`sample_draws.jsonl`, fixed validation IDs and validation Projector TSVs. Sample
construction provenance is already in the dataset; `sample_draws.jsonl` records
which prepared examples were selected for each update. Checkpoints do not provide
exact optimiser/sampler resume.

Keep comparison budgets fixed, then assess embeddings with downstream detection
metrics, temporal shuffle controls and appropriate holdouts. Reconstruction error
and cluster appearance alone do not establish temporal or detection performance.
Losses on learned unit vectors and on the historical raw feature inputs are on
different scales and should not be compared as equivalent numerical scores.
