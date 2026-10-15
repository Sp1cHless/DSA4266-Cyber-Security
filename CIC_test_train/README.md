# CICIDS2018 CNN-AE pilot

This folder contains an independent **CIC-only** experiment. It reuses the
existing `datasets/cic_2018_traffic.db` and frozen 88-feature schema; it does not
import the CSV, rebuild buckets, or mix UNSW training sequences into this run.

## Run from the repository root (Windows PowerShell)

Use the existing working environment:

```powershell
.\.venv-cnn\Scripts\Activate.ps1
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

CUDA must print `True`. If PowerShell blocks activation, call
`.\.venv-cnn\Scripts\python.exe` instead of `python` in every command below.

Optional input check (no generated data):

```powershell
python .\CIC_test_train\prepare_data.py --check-only
```

Build windows with the default stride of 5:

```powershell
python .\CIC_test_train\prepare_data.py
```

Train one CNN autoencoder, freeze thresholds, and evaluate:

```powershell
python .\CIC_test_train\train_cnn_ae.py
```

Both programs also work from other directories because default paths are
relative to the script folder. Explicit relative-path arguments are relative
to your terminal's working directory.

If the environment needs dependencies, use a CUDA-enabled PyTorch build and
install `numpy` and `scikit-learn`. This folder does not require pandas/joblib.

## Data and experiment definition

- 10-second source-IP bins, 10 bins per window (100 seconds), stride 5 (50 seconds).
- Missing bins retain their original temporal positions. At least one observed
  bin is required; entirely empty windows are not enumerated.
- Window label = 1 if any observed bucket contains an attack. A source IP can
  have benign and attack windows. Labels never determine window boundaries.
- Actual observed 30-minute NETWORK time blocks are grouped by UTC capture day
  and allocated approximately 60/10/10/20% to train/validation/calibration/test
  using seed 0 and no labels. Days with fewer than four active blocks go to test.
- Windows crossing a block/day boundary are discarded. No raw bucket or global
  time block is shared between splits; both programs verify this.
- Train, validation and calibration use only benign windows. Test retains ALL
  nonempty in-block windows of both classes, without balancing or undersampling.
- Benign sample budgets: 100,000 train / 20,000 validation / 20,000 calibration.
  Sampling cycles across days, hosts and blocks. A single host contributes at
  most 1% of the requested split budget; shortages are reported in the manifest.
- This is same-environment detection on unseen time blocks. Hosts can recur
  across splits. It does not measure unseen-host or future-day generalization.
- Raw attack names are scanned once to annotate test windows; labels/names do
  not enter the network input or model/threshold selection.

## Model and scoring

- 89 encoder inputs = 88 behavioural features + observed-bin mask.
- 88 reconstruction outputs; the mask is not a reconstruction target.
- CNN layout follows the existing UNSW control; parameter budget is about 250k,
  latent size 32, Adam lr 0.001, batch 256, maximum 30 epochs, patience 5.
- Log1p + StandardScaler for counts/volumes/durations/distinct counts; ratio and
  entropy features remain in [0,1]. The scaler fits UNIQUE observed buckets from
  the selected benign training windows, once per bucket, without padding.
- Padding becomes zero AFTER scaling and is explicitly marked as unobserved.
- Each sequence loss/score is the mean feature MSE over its observed bins.
  Training averages these sequence losses; sparse windows are not diluted by
  empty bins. CNN inputs still see the pattern of observed positions.
- Benign validation selects the checkpoint. Separate benign calibration fixes
  0.1%, 1%, and 5% FPR quantile thresholds BEFORE test inference.
- Overlapping windows are correlated; the 0.1% tail has limited calibration
  support. One-active-bin windows provide little temporal context, so occupancy
  subgroup results are essential. No independent-window confidence interval is claimed.
- Bucket features are allocated by flow start and can describe a flow that ends
  later. This is an offline experiment, not a causal real-time latency benchmark.

## Outputs

Default prepared dataset: `CIC_test_train/data_stride5/`

- `manifest.json`: counts, prevalence, occupancy, missing attack types, sample shortages, source fingerprint.
- `time_blocks.csv`: exact frozen network time-block split.
- `bucket_features.npy`: raw 88-feature bucket matrix, disk-mapped (about 1.20 GiB for the current DB).
- `*_windows.npy`: compact window indices, labels, attack annotations and active counts.
- `*_metadata.csv`: window identities and attack types aligned with saved arrays.
- `categorical_schema.json`, `hosts.json`.

Default experiment: `CIC_test_train/runs/cnn_ae_stride5_seed0/`

- `test_metrics.json`: ROC-AUC, PR-AUC/AP, prevalence, detection rate, actual FPR, precision, confusion counts.
- `metrics_by_attack.csv`: each attack type vs all benign test windows; multi-type windows can appear in multiple rows.
- `metrics_by_day.csv`, `metrics_by_occupancy.csv`: temporal/sparsity diagnostics; frozen global thresholds.
- `test_scores.csv`, `calibration_scores.csv`: per-window scores/flags; IDs align with prepared metadata.
- `cnn_ae_checkpoint.pt`, `scaler.npz`, `feature_transform.json`, `thresholds.json`.
- `training_history.csv`, `environment.json`, `run_config.json`.

Generated data and runs are ignored by Git. Both programs require an empty/new
output directory. Interrupted runs leave `complete/completed=false`; choose a
new output folder instead of overwriting it. For a repeated model run, reuse the
prepared data and supply only a new `--outdir`.

```powershell
python .\CIC_test_train\train_cnn_ae.py --outdir .\CIC_test_train\runs\cnn_ae_stride5_seed0_retry
```

If GPU memory is insufficient, add `--batch-size 128` (or 64). Changing the model
seed is a new experiment; retain all runs instead of selecting using test AUC.

## Verify the implementation on tiny synthetic data

```powershell
python .\CIC_test_train\test_pipeline.py
```

Checks sparse time geometry, labels, split isolation, padding, scaler fitting,
sampling and score normalization; then builds a disposable small SQLite dataset
and runs two tiny CUDA epochs if a GPU is available. Synthetic metrics validate
the plumbing and are not CIC experimental results.
