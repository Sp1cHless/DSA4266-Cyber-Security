# LSTM window comparison: Infilteration and Brute Force Web

Use `run_temporal_lstm.py` for the selected main model. It references the original
100-second / 10-second LSTM result and trains four fresh LSTMs on the already
completed new-window preparations. No CNN is trained by this entry point.

| Sequence seconds | Bucket seconds | Buckets |
|---:|---:|---:|
| 200 | 10 | 20 |
| 200 | 20 | 10 |
| 400 | 10 | 40 |
| 400 | 40 | 10 |

## Historical reference and unchanged training policy

The reference is `train/lstm/runs/20261023_023651_471618_lstm_seed4266_b121bc/`.
Its config, summary, checkpoint and thresholds are read and checked; its weights
and completed results are never replaced, and the original configuration is not
trained again. The same archived bucket-encoder config used by Scheme2 supplies
the representation-fitting budget.

- One 64-unit LSTM, 64-unit classification head and dropout 0.2. Explicit masks,
  masked mean/max pooling and final hidden state are unchanged. Empty time bins
  remain in the sequence. T=20/40 inputs are not cropped or resized to ten bins.
- Original LSTM budget: at most 30 epochs, validation-AP patience 5, batch 512,
  AdamW lr 0.001, weight decay 0.0001, gradient norm cap 1.0, seed 4266 and four
  CPU threads. These values come from the archived LSTM config, not the CNN.
- All training samples are visited once per shuffled epoch. The four real/
  synthetic benign/malicious groups each contribute 25% of the objective under
  the unchanged weight formula.
- Checkpoints are selected on full real-validation average precision, including
  all attack types. Checkpoints are not selected to maximize the two focus types.
- Global operating thresholds are selected separately for each run by the same
  validation FPR 1% and 0.1% rules. The historical numeric threshold is not applied
  to a newly trained model, and no attack-specific threshold is fitted.
- The input encoder retains the original 5,000-step budget, batch 512, lr 0.001,
  category dimension 16, bucket dimension 64, validation every 200 steps, fixed
  validation limit 4,096 and two CPU threads. Zero-shift donor sampling, P
  synthetic examples per class, vector addition/L2 and flow-count scaling follow
  the existing Scheme2 policy.

Thirty epochs is an inherited comparison budget, not a claim of convergence.
This experiment does not simultaneously change the optimizer or early stopping.

## Run from the repository root

Use the computer's existing `.venv-cnn` environment directly. Do not create a new
environment or install/downgrade packages for this experiment. The driver records
current versions and any differences from the historical reference without
blocking training. No additional runtime-override flag is required. Model,
optimization, checkpoint selection and validation rules still come from the
original LSTM reference.

Preview, then run:

```powershell
.\.venv-cnn\Scripts\python.exe -u train/run_temporal_lstm.py --dry-run
.\.venv-cnn\Scripts\python.exe -u train/run_temporal_lstm.py
```

The driver automatically finds the latest matching batch containing all four
completed preparations and prints its path. An explicit selection is supported:

```powershell
.\.venv-cnn\Scripts\python.exe -u train/run_temporal_lstm.py --prepared-batch CIC_new/outputs/temporal_ablation_runs/20261024_184722_957599_b919ae
```

The raw buckets, vocabularies, scalers and splits are reused without rebuilding
them. Each case fits/freezes its bucket encoder, encodes/exports train and
validation only, then runs the existing LSTM training and validation entry point.
Test feature shards are not read, encoded, cached or evaluated. Original manifest
and SQLite catalog metadata still contain IDs/labels for all original splits as
required by unchanged synthetic-ID construction; that is not test-feature fitting.
Training data are reused for fitting and are not reported as held-out evidence.

## Focused outputs

A new `train/runs/<unique-lstm-temporal-run>/` contains:

- `lstm_config.json`, `batch.json`, per-case representation/LSTM folders and logs.
- `ATTACK_COMPARISON.md`: the main table, with the original LSTM plus each completed
  new-window LSTM for both focus types. It shows N, recall at 1% and 0.1% FPR, and
  descriptive changes in percentage points against the original LSTM.
- `attack_comparison.csv`: TP, FN, sample/host counts, recall, actual global FPR,
  thresholds and baseline references for both operating points. Missing types
  have null recall rather than zero.
- `validation_comparison.csv` and `.json`: overall AP, ROC-AUC, prevalence and
  global operating metrics, alongside the focus-type values.

The original LSTM values, recomputed from its saved validation predictions, are:

| Primary attack | N | TP / FN at 1% | Recall at 1% | TP / FN at 0.1% | Recall at 0.1% |
|---|---:|---:|---:|---:|---:|
| Infilteration | 366 | 254 / 112 | 69.40% | 128 / 238 | 34.97% |
| Brute_Force_-Web | 181 | 86 / 95 | 47.51% | 10 / 171 | 5.52% |

The goal is to inspect improvement in these two binary-detection slices while
tracking realized benign FPR and overall discrimination. This is not multiclass
attack classification. Longer sequence durations change populations, prevalence,
split assignments and class coverage; recall changes are descriptive validation
comparisons, not paired causal estimates. In particular the current 400-second
preparation has few validation Infilteration examples, so denominators matter.

## Resume and small checks

Resume an existing LSTM training batch with the same settings/references:

```powershell
.\.venv-cnn\Scripts\python.exe -u train/run_temporal_lstm.py --out <lstm-training-batch-directory> --resume
```

Completed stages are hash-checked and skipped; incomplete stages restart in fresh
numbered attempt directories. Old data and weights are preserved. Do not point
the LSTM runner at an older CNN training batch. After a forced kill, verify the
process has stopped before removing its stale `RUNNING.lock`.

The runner now limits disposable storage to the current case. After a verified
representation export, it removes train/validation preprocessing mmap caches and
encoded intermediate arrays. After verified detector completion, it removes that
attempt's loading cache. Final datasets, split records, source data, encoder and
detector checkpoints, predictions, metrics and logs remain. Embedded encoder
copies and JSON provenance from removed caches are preserved under `cache_cleanup/`.
Interrupted train/validation `.partial` caches are removed before retry; failed
stage directories and logs remain and new attempts get fresh numbered folders.

Each new stage checks an allocation estimate plus a 2 GiB reserve on its target
drive before starting. The representation estimate includes missing dense input
caches, frozen encoding and an uncompressed export bound. No model, training,
split, augmentation or numeric precision setting is changed to save space.

An existing failed batch made by the reviewed original driver can adopt only
this storage change: all scientific source hashes, references, data, runtime and
parameters must still match. The original batch metadata is snapshotted and the
storage migration is recorded. Completed stages remain hash-checked and reused.
To reclaim completed-case caches without starting another training stage:

```powershell
.\.venv-cnn\Scripts\python.exe -u train/run_temporal_lstm.py --out <existing-batch-directory> --resume --cleanup-only
```

The LSTM and optional CNN entry points share `common/temporal_experiment.py` for
orchestration/reporting and call the original common training loop. The CNN
entry point remains available as a separate experiment, not the main baseline.

```powershell
.venv-cnn\Scripts\python.exe -m unittest train.tests.test_temporal_lstm train.tests.test_temporal_cnn -v
```

These artificial fixtures use explicitly short reference budgets, hide test
features, and exercise all four geometries, matching settings, equal weights,
prediction reload, focused recall arithmetic and completed-stage reuse. They do
not establish real-data accuracy or convergence. The real-data experiment is
validation-only, with validation selecting checkpoints and thresholds; no fresh
independent test result is claimed.
