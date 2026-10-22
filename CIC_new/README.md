# CIC-IDS2018 workflows

**Active method: Scheme2, with the existing representation and port rules.**

Sequence construction and bucket aggregation precede the approximate 6:2:2
time-block split. Transforms are fitted on real training data; synthetic examples
are added only to training. Frozen 64-dimensional whole-bucket vectors are added
and L2-normalized; a separate scaled log-flow-count channel gives shape `[N,10,65]`.
Displacement is zero.

## Completed Scheme2 experiment

Scheme2 has already completed one full temporal-AE training run, validation and
evaluation on all 263125 original held-out test sequences (851 malicious).
Training completed through early stopping at epoch 257; the best checkpoint was
epoch 249. A linear readout fitted on real training latent vectors was then
evaluated with frozen weights and thresholds.

Test ROC-AUC was 0.84948 and average precision was 14.93%. At the training-benign
99th-percentile threshold, recall was 39.37% with actual test FPR 0.93%.
These completed results are the current Scheme2 baseline, including when used
independently of the original Scheme1 comparison. See
[the experiment record](SCHEME2_EXPERIMENT.md) for artifact locations and scope.

## Active commands

First follow the [environment and data setup](../README.md) for a fresh checkout.
The commands below run from the repository root with that Python environment:

```powershell
# Generate a fresh dataset from SQLite. No Scheme1 output is required.
.\.venv-cnn\Scripts\python.exe -u CIC_new\prepare_scheme2_dataset.py
# Verify the latest completed dataset; does not fit a model.
.\.venv-cnn\Scripts\python.exe CIC_new\run_scheme2_experiment.py --stage check
# Train and then perform frozen validation evaluation in a new run directory.
.\.venv-cnn\Scripts\python.exe -u CIC_new\run_scheme2_experiment.py
# Training and evaluation can also be run separately.
.\.venv-cnn\Scripts\python.exe CIC_new\run_scheme2_experiment.py --stage train
.\.venv-cnn\Scripts\python.exe CIC_new\run_scheme2_experiment.py --stage evaluate --run <completed-scheme2-run>
```

See [data construction](SCHEME2_DATASET.md) and [training/evaluation](SCHEME2_EXPERIMENT.md).
Reuse the existing completed dataset/model for downstream work. New commands do
not resume an interrupted optimizer or overwrite completed runs.
The generic `train_autoencoder.py` also defaults to Scheme2; pass `--dataset`
explicitly when reproducing another representation.

The default selects the latest completed build registered under
`outputs/scheme2_runs/`. For an existing dataset copied from another member, use
`--dataset <path-to-final-dataset>` for check/training. The final path ends in
`dataset/`, not `preprocessed/` or `encoded_sequences/`. For existing AE runs,
evaluation uses the dataset path saved in config.json; check that path before
reusing a training run on another machine.

## Directory map

| Path | Role |
|---|---|
| `preprocessing/` | Shared sequences, buckets, split, scaling, augmentation and loader |
| `representation_learning/` | Scheme2 bucket encoder and frozen sequence encoding |
| `modelling/` | Downstream temporal AE architecture and trainer |
| `outputs/scheme2_runs/` | Active Scheme2 preprocessing, encoder, encoded originals and dataset |
| `outputs/ae_runs/` | Active Scheme2 detector checkpoints, histories and evaluations |
| `outputs/test_evaluations/` | Completed frozen real-test evaluation and paired scores |
| `reports/superimposition_comparison_20261022/` | Shared summary/test reports; full evidence and cleanup audits stay local |
| `scheme1/` | Retired implementation, retained for reproduction |
| `tests/` | Small fixtures and regression checks |
| `EDA/`, other `reports/` subfolders | Historical analyses |

## Retired comparison tools

`prepare_dataset.py`, `train_embeddings.py`, `validate_dataset.py` and `scheme1/`
can rebuild Scheme1 from the original database. Shared preprocessing modules are
also required; copying only the entry-point files is insufficient. See
[the Scheme1 reproduction guide](scheme1/README.md).

`run_scheme1_l2_experiment.py`, `diagnose_superimposition.py`,
`diagnose_controlled_dilution.py` and `evaluate_test_set.py` are historical
comparison tools. If the old Scheme1 outputs have been removed, rerunning these
comparisons requires rebuilding the datasets/models and supplying their paths.
Completed conclusions remain readable without rebuilding.
`run_scheme2_experiment.py --baseline-run ...` supports historical paired
comparison explicitly, while its default operates independently.

Train-derived dilution diagnostics are distinct from the real held-out test.
The final real-test comparison favored Scheme2 overall; see
[the recorded decision and limitations](reports/superimposition_comparison_20261022/SUMMARY.md).

## Code checks

```powershell
python -m unittest discover -s CIC_new/tests -t . -v
```

Fixtures validate implementation behavior. They do not establish full-corpus
runtime, cross-seed stability or new detection performance.
