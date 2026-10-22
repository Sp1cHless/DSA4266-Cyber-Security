# Optional downstream models

For the selected Scheme2 training and frozen validation evaluation,
run `python CIC_new/run_scheme2_experiment.py`; see
[SCHEME2_EXPERIMENT.md](../SCHEME2_EXPERIMENT.md). It uses the established training
settings and creates a separately labelled run and evaluation folder.
`--stage evaluate` evaluates an existing completed scheme2 AE without retraining it.

Preprocessing is complete before running anything in this folder. The shared
input is an exported dataset containing X, masks, labels and provenance. A CNN,
LSTM, classifier or another model can use the same arrays. No classifier or LSTM
implementation is claimed here; the temporal AE is the supplied example.

`autoencoder.py` defines only the temporal CNN AE and masked reconstruction loss.
`train_autoencoder.py` loads the **already exported dataset**, samples batches,
fits that AE and evaluates real validation examples. It does not construct raw
sequences, aggregate buckets, split data, fit input embeddings or create synthetic
samples. Those are separate steps and modules.

For the latest completed Scheme2 dataset, run this direct entry point from the project root:

```powershell
python CIC_new\train_autoencoder.py
```

Defaults: automatic CUDA selection, input dimensions from the dataset manifest,
latent width 32, constant Adam learning rate 0.001, batch size 128, at most 500
epochs of 500 updates each. Each batch contains 32 examples
from each of real benign, synthetic benign, real malicious and synthetic malicious.
The entire real validation split selects the checkpoint by overall reconstruction
MSE, with benign/malicious MSE reported separately. Input embeddings stay frozen.
This is a convolutional AE reconstruction baseline; detection classification
metrics require a separate classifier/evaluation experiment.

Early stopping defaults to `--patience 20 --min-delta 0.001`. After each epoch,
full validation MSE is compared with the last significant reference. A relative
decrease greater than 0.1% updates that reference and resets the counter; smaller
decreases can accumulate toward the threshold. Twenty epochs without such an
improvement stop training. `best.pt` still records EVERY new absolute minimum,
including improvements smaller than the stopping threshold. The best model is
restored for final validation and Projector export. `--patience 0` disables
early stopping. No learning-rate scheduler is used.

`summary.json` records epochs completed, best epoch, restored per-class errors,
and `stopping_reason` (`early_stopping` or `max_epochs`). Only the former meets
the configured plateau criterion; exhausting the budget does not prove convergence.
This is one seeded run, not a measurement of stability across random seeds.

The first use builds train/validation disk caches for the 65-feature Scheme2
export. Disk requirements also include metadata and sampling logs; reserve space
for the dataset and each training run. Every direct invocation automatically reserves a new directory
under `CIC_new/outputs/ae_runs`, for example:

```text
<timestamp>_scheme2_ae_aug_e500_b128_seed4266_<unique suffix>/
```

The name includes local time, augmented/real-only sampling, maximum epochs, batch
size, seed and a collision-safe suffix. The absolute path is printed before data
loading. Each run keeps its own config, curves, checkpoints and sample audit.
`run_status.json` distinguishes `running`, `complete`, `interrupted` and `failed`;
an interrupted run is not a completed training result. Shared read-only data and
its reusable mmap caches remain under the dataset directory.
An explicit `--out` overrides the automatic path; existing nonempty directories
are refused so results cannot be appended to another run.
For a short compatibility run:

```powershell
python CIC_new\train_autoencoder.py --epochs 1 --steps-per-epoch 10 --val-limit 4096 --log-every 5
```

The direct entry point defaults to full validation (`--val-limit 0`). The older
module entry point below retains its original defaults and explicit paths.

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

## Independent frozen-model evaluation

For historical reproduction of the Scheme1 per-bucket L2 ablation and three-way
evaluation, see [SCHEME1_L2_EXPERIMENT.md](../SCHEME1_L2_EXPERIMENT.md).

```powershell
python CIC_new\evaluate_autoencoder.py
python CIC_new\evaluate_autoencoder.py --run CIC_new\outputs\ae_runs\<completed-run>
```

This reads the latest completed run by default and reserves a unique directory
under that run's `evaluations/`. It never restarts AE training, changes AE weights,
or opens test. Full train/validation inference reports per-group, primary-attack,
active-bucket and feature errors, tail loss concentration and latent variation.
Every line of the training draw audit is checked against exported train membership
and the per-update balanced quotas. Small validation subsets used for checkpoint
selection are reproduced separately from the full validation diagnostics.

A separate linear probe standardizes the frozen latent vectors using real train
only, then fits one class-balanced logistic model on real train (L2 0.001; at most
100 L-BFGS iterations). AE weights remain frozen. The fixed operating thresholds
are balanced-prior probability 0.5 and the real-training benign score quantiles
99%/99.9%; validation is not used to fit or choose these thresholds. ROC-AUC and
non-interpolated average precision group tied scores together. Primary-attack and
natural mixed-flow recall are diagnostic only; small subtype counts are reported.

Outputs include `report.json`, `linear_probe.json`, `validation_scores.npz`,
`training_draw_counts.npz`, `reconstruction.json` and `evaluation_status.json`.
The probe is neither the planned CNN nor an AE/CNN hybrid. Validation already
selected the AE checkpoint, so these are exploratory validation results. A
real-only AE control and multiple seeds remain needed to measure augmentation
benefit and stability. The completed frozen real-test comparison is recorded in
[the comparison summary](../reports/superimposition_comparison_20261022/SUMMARY.md).

## Real held-out test evaluation

**Scheme2 has completed a full temporal-AE training run and real held-out test
evaluation.** The comparison experiment supplies the current Scheme2 baseline:
test ROC-AUC 0.84948 and average precision 14.93%, using a frozen AE and a
real-train-fitted linear readout. See [the experiment record](../SCHEME2_EXPERIMENT.md)
for completion status, thresholds and artifact locations.

The completed Scheme1 L2 versus Scheme2 real-test comparison used frozen AE
checkpoints, train-fitted probes and thresholds fixed before test inference.
Results are in [the comparison archive](../reports/superimposition_comparison_20261022/SUMMARY.md).
[TEST_EVALUATION.md](../TEST_EVALUATION.md) documents the historical paired tool;
if Scheme1 data/checkpoints have been removed, rebuild them before rerunning it.
The active Scheme2 `--stage evaluate` command evaluates validation only.
