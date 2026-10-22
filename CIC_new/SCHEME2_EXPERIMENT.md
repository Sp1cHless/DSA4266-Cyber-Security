# Scheme2 AE training, validation and completed test results

## Completed baseline

**One complete Scheme2 training/validation/test experiment has already been run.**
It was part of the Scheme1/Scheme2 comparison and is retained as the current
Scheme2 experimental baseline for reporting and subsequent model comparisons.
The detection pipeline is a temporal AE plus a linear readout fitted on real
training latent vectors. This experiment does not evaluate the planned CNN.

Training completed normally through early stopping at epoch 257, using the best
validation checkpoint from epoch 249. Frozen evaluation then covered the entire
real held-out test split: 263125 original sequences, including 851 malicious and
262274 benign. Test data did not fit weights, select checkpoints or choose
thresholds.

| Recorded result | Scheme2 |
|---|---:|
| Validation ROC-AUC | 0.87922 |
| Validation average precision | 19.65% |
| Test ROC-AUC | 0.84948 |
| Test average precision | 14.93% |
| Test recall at training-benign 99th-percentile threshold | 39.37% |
| Actual test FPR at that threshold | 0.93% |

Training artifacts are under
`CIC_new/outputs/ae_runs/20261022_131924_794173_scheme2_ae_aug_e500_b128_seed4266_ze7iu70m/`.
The completed test report, fixed evaluation plan, scores and slices are under
`CIC_new/outputs/test_evaluations/20261022_194436_scheme1_l2_vs_scheme2_zm94li19/`.
The training summary's `test_used: false` describes training/checkpoint selection;
the separate test report records the subsequent frozen test evaluation.
The [shared test report](reports/superimposition_comparison_20261022/TEST_RESULTS.md)
remains readable without the large local dataset or model files.

This is meaningful empirical evidence for the completed Scheme2 pipeline at one
seed. Additional seeds and controls can assess robustness and augmentation benefit;
they do not change the completed status of this experiment.

## Running a new experiment

The selected method uses the completed Scheme2 dataset with shape [N,10,65].
The default entry point runs independently of retired Scheme1 outputs:

```powershell
.\.venv-cnn\Scripts\python.exe -u CIC_new\run_scheme2_experiment.py
```

It selects the latest completed Scheme2 dataset, checks its validation record
and donor-plan checksum, trains a new temporal AE, restores the best checkpoint,
then performs frozen validation diagnostics. Existing completed runs are unchanged.
This default command performs training and validation; it does not automatically
repeat the completed held-out test evaluation described above.

| Setting | Value |
|---|---:|
| Maximum epochs | 500 |
| Updates per epoch | 500 |
| Batch | 128; 32 per group, with replacement |
| Seed | 4266 |
| Hidden width / latent | 32 / 32 |
| Adam lr | 0.001 |
| Patience | 20 |
| Relative minimum improvement | 0.001, or 0.1% |
| Checkpoint validation | Entire real validation split |

A batch entry is one ten-bucket sequence. Each epoch samples 64000 entries with
replacement, rather than sweeping all training data. Real benign, real malicious,
synthetic benign and synthetic malicious receive equal sampling weight.

Evaluation infers on full train/validation and reports reconstruction errors,
tails, latent variation and sampling coverage. A separate fixed class-balanced
linear readout is fitted only on real training latent vectors. It is not the
planned CNN or AE/CNN hybrid. Thresholds use balanced-prior 0.5 or training-benign
score quantiles. Validation is used for AE checkpoint selection, so these are
exploratory validation results, not another final test evaluation.

Compact flow_metadata_validation.npz and flow_metadata_test.npz sidecars contain
original sequence counts with exact ID/total/label checks and recorded hashes.
They remove the old evaluation dependency on Scheme1 arrays. Existing frozen
dataset/training manifests and learned weights were not rewritten during cleanup.

```powershell
# Check every exported shard checksum, without fitting.
python CIC_new\run_scheme2_experiment.py --stage check
# Train only, in a new identifiable run directory.
python CIC_new\run_scheme2_experiment.py --stage train
# Evaluate an existing completed run without retraining.
python CIC_new\run_scheme2_experiment.py --stage evaluate --run <completed-scheme2-run>
```

Pin data using --dataset; use --out only for a fresh training directory.
Default new runs go to outputs/ae_runs/<timestamp>_scheme2_ae_aug_e500_b128_seed4266_<suffix>/,
with config, history, checkpoints, summary, sampling audit and evaluations/.
The generic train_autoencoder.py also defaults to the latest Scheme2 dataset and
allows individual training controls. Nonempty output directories are refused.

Optional --baseline-run <rebuilt-scheme1-run> enables the historical matched
comparison, with optional --baseline-evaluation <report.json>. Those artifacts
were intentionally retired and must be rebuilt before repeating that comparison.
The completed real-test and diagnostic conclusions are preserved in
[the comparison archive](reports/superimposition_comparison_20261022/SUMMARY.md).
Raw reconstruction MSE across different target representations does not rank
attack-detection quality.
