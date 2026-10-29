# Scheme2 supervised model screening

Compare XGBoost, LSTM, temporal CNN, and CNN + LSTM on the same frozen Scheme2
inputs. This round selects a main model family using fixed reasonable baselines;
it does not perform a hyperparameter search. No complete real-data model-screening
run is bundled with this code. Artificial fixture checks are not accuracy evidence.
Different team members run their assigned model independently on their own machines,
then share the completed run folders for one common comparison. Windows, macOS and
Linux use the same Python entry points; CUDA is optional.

See [provisional LSTM and Web findings](LSTM_WEB_FINDINGS.md) for the current
baseline decision, circular-shift results and interpretation boundaries.

## Input and common training policy

The default input is `<repository>/datasets/dataset/`, containing the completed
Scheme2 `manifest.json`, `validation.json`, and train/validation/test NPZ shards.
Keep all other delivered dataset files for provenance and frozen representation
reuse. The original database and bucket encoder training are unnecessary here.
Historical absolute paths inside the dataset manifest are provenance, not runtime
dependencies. Relocating the dataset is supported with `--dataset`.

Each sample contains ten 10-second buckets and 65 float32 channels: a frozen
64-dimensional unit bucket representation plus one scaled log-flow-count channel.
No new embedding fitting, scaling, donor generation, resplitting or augmentation
runs here. The explicit bucket mask is available to every model. Labels, attack
names, source IPs and donor IDs never enter model inputs.

| Split | Real benign | Real malicious | Synthetic benign | Synthetic malicious |
|---|---:|---:|---:|---:|
| Train | 757559 | 8093 | 8093 | 8093 |
| Validation | 250229 | 917 | 0 | 0 |
| Test | 262274 | 851 | 0 | 0 |

All four training groups contribute 25% of the supervised binary classification
objective. Every example receives weight `N / (4 * group_count)`, with mean weight
one. XGBoost receives these weights directly; neural models visit every training
example once per shuffled epoch and use the same weights in binary cross-entropy.
There is no additional class weight or positive-class multiplier. Weighting changes
the training prior, so outputs are ranking scores, not calibrated natural-prior
attack probabilities. Validation and test metrics are unweighted at natural prevalence.

## Models and fixed defaults

| Directory | Model input / architecture | Training budget |
|---|---|---|
| `xgboost/` | Flatten all ten buckets (650 values), append ten mask values; histogram boosted trees, depth 5, learning rate 0.05, row/column subsampling 0.8 | At most 600 rounds; AP patience 40 |
| `lstm/` | Concatenate mask to each bucket; one 64-unit LSTM; masked mean/max pooling plus final hidden state; 64-unit classification head | At most 30 epochs; AP patience 5 |
| `cnn/` | Concatenate mask; temporal Conv1d 66 -> 64 -> 128, kernels 3, padding 1; masked mean/max pooling; 64-unit head | At most 30 epochs; AP patience 5 |
| `hybrid/` | Temporal Conv1d 66 -> 64, kernel 3; one 64-unit LSTM; masked pooling plus final hidden state; 64-unit head | At most 30 epochs; AP patience 5 |

Neural defaults: batch 512, AdamW learning rate 0.001, weight decay 0.0001, dropout
0.2, gradient norm cap 1.0. Empty bins remain on the original grid, preserving
elapsed time; padding is zeroed, and pooling excludes inactive bins. Recurrent
states may evolve through empty time bins. Every family selects its saved best
checkpoint using the same sklearn average precision on the entire real validation
split. XGBoost uses a custom AP callback and saves only the best tree ensemble.
Model sizes and training budgets differ by family; this is baseline pipeline
screening, not a controlled equal-capacity architecture experiment.

Seed defaults to 4266; CPU thread limit is 4. The default `--device auto` uses CPU
XGBoost and CUDA neural models only when CUDA is available, otherwise CPU. On a Mac
the default is CPU; neither CUDA nor an NVIDIA GPU is required. Use `--device cpu`
to explicitly select the portable CPU path on any machine. Optional `--device mps`
selects an available Apple GPU for the three neural models; XGBoost remains CPU
and rejects MPS. MPS availability is checked through PyTorch's documented backend
API. Actual MPS training is unverified on this Windows development machine; if the
target Mac/backend cannot run an operation, use CPU. Software versions, platform,
actual device, source hashes, parameters and model hashes are recorded. Seed controls
randomness; cross-device/backend bitwise equality is not promised.

## Run from the repository root

For the four sequence-duration/bucket-duration comparisons using completed raw
preparations, use [TEMPORAL_LSTM.md](TEMPORAL_LSTM.md). `run_temporal_lstm.py`
compares the selected original LSTM baseline with four new-window LSTMs, retaining
its saved training/input-encoder settings. It reports Infilteration and Brute
Force Web recall at both FPR budgets, records current package versions and processes
train/validation only. The original 100s/10s LSTM is referenced without refitting.
The earlier [CNN driver](TEMPORAL_CNN.md) remains a separate optional experiment.

Each member uses their own environment, the same code revision, requirements,
dataset, seed and published baseline settings. Do not change optimization settings
independently while screening model families. On macOS/Linux, for example:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r train/requirements.txt
```

On Windows, an existing environment can be used with
`.\.venv-cnn\Scripts\python.exe` in place of `python`. A CUDA-enabled PyTorch
installation is optional for NVIDIA users. CPU training works without it.

From the repository root, run **only the assigned model's command**:

```bash
python train/xgboost/train.py
python train/lstm/train.py
python train/cnn/train.py
python train/hybrid/train.py

# Optional: verify/cache the real train and validation shards without fitting.
python train/check_data.py
```

These are four independent entry points for four members, not a command sequence
that every member must execute. Each member needs the repository's `train/` package
(including `common/`) and the delivered `datasets/dataset/`, excluding its old
`model_cache/`. A model folder by itself does not contain the shared runtime.

The initial data read verifies every accessed NPZ checksum, shape, label, mask,
padding, population and sample-ID ownership. A disposable mmap cache is written
under `train/cache/`, never inside the delivered dataset. It requires approximately
3 GiB for train + validation on this dataset, plus runtime memory and model storage.
XGBoost additionally materializes a 660-column matrix and tree-training structures;
allow several GiB of extra RAM. Available memory requirements depend on the backend.
An interrupted cache has no `ready.json`; choose a fresh `--cache <directory>` or
remove only that incomplete cache before retrying. Never delete source shards.

Explicit device selection or data relocation:

```bash
# Portable Mac/Windows/Linux CPU run.
python train/cnn/train.py --device cpu
# Optional Apple GPU for CNN, LSTM or hybrid, when MPS is available.
python train/lstm/train.py --device mps
# Example data relocation on a Mac.
python train/hybrid/train.py --dataset "/Users/member/shared/dataset" --device cpu
```

Outputs go to `train/MODEL/runs/<unique_run>/`. Existing output directories, even empty ones,
are refused. All outputs and caches are ignored by Git; source code is tracked.
Defaults live in each model's `config.json`; `--config` accepts an alternate copy.
Use `--epochs`, `--batch-size`, `--rounds` or `--patience` only when intentionally
changing a budget. Rerun all relevant candidates under a consistent screening plan.

## Common evaluation and model selection

For the separate four-model circular-shift retraining experiment, see
[the circular-shift guide](CIRCULAR_SHIFT.md). `python train/run_circular_shift.py`
runs fresh chronological/shifted pairs for all four families (eight fits) and
writes within-family metric deltas and per-attack comparisons. Add `--shift-only`
for four shifted fits without paired controls. Every individual model entry point
also accepts `--circular-shift-start 5`; start 0 retains chronological order.
This diagnostic reuses the training cohort and uses validation only, not test.

The prediction target is binary attack presence per 100-second sequence. Primary
checkpoint and ranking metric: **average precision (AP)** computed by sklearn
`average_precision_score`, not trapezoidal PR-AUC. AP is useful here because the
real validation prevalence is only about 0.365%; accuracy alone is misleading.

Every model reports the same metrics:

- AP and ROC-AUC from continuous scores, with positive prevalence as an AP baseline.
- Recall at validation FPR budgets 1% and 0.1%, with actual FPR and TP/FP/TN/FN.
- Precision, F1, accuracy, balanced accuracy and MCC at those thresholds and 0.5.
- Recall and sample/source-host counts per primary attack type; macro recall over
  primary types with at least 20 positive examples, using the global 1% threshold.
- Fit wall time (including checkpoint validation), model file size, and full
  validation prediction throughput including input preparation and host output.
  Timing is a single post-fit pass, not a hardware-normalized latency benchmark.

Operating thresholds are selected from real validation scores only. For each FPR
budget, maximize recall without exceeding it; tied scores are handled as a block.
An unattainable budget can yield an all-negative decision. Serialize thresholds
before any test scoring, and reuse them unchanged in later evaluation. Test FPR
can differ from the validation budget. No attack slice fits its own threshold.
Validation metrics are exploratory because validation selects both checkpoints
and thresholds. Per-type counts may be small; primary labels are not multilabel
membership, and macro metrics do not remove host/campaign dependence.

`compare.py` writes `COMPARISON.md`, `comparison.csv`, and
`comparison.json`. It recommends the highest-AP family, using recall at 0.1% FPR
then ROC-AUC only for exact ties. Inspect low-FPR recall, per-type weaknesses,
training cost and deployment constraints before adopting it. Close results need
additional seeds, not an unsupported significance claim.

After training, each member shares their completed `train/MODEL/runs/<unique_run>/`
folder. The coordinator can rename the received folders to `xgboost`, `lstm`, `cnn`
and `hybrid` under a common collection directory. Paths inside old configuration
files do not need editing for comparison. No dataset, local cache, or training
rerun is required by `compare.py`; it reads the saved `summary.json` files only.
The full run folder is recommended for reproducibility; `summary.json` alone is
sufficient for making the comparison table.

```bash
python train/compare.py --runs <xgboost-run> <lstm-run> <cnn-run> <hybrid-run>
# Alternative layout: collected/xgboost/summary.json, collected/lstm/summary.json, etc.
python train/compare.py --root collected
```

Comparison refuses mismatched dataset fingerprints, cohort identities, seeds,
metric policies or weighting. It also refuses duplicate model families. A progress
table is possible with `--allow-partial`, which makes no final recommendation.
Runs from different operating systems or devices are accepted. Detection metrics
use the same real validation population and protocol, while speed and fit time
are machine-specific references and never enter the automatic ranking. To make
an architectural speed comparison, benchmark saved models on one common machine.

Every completed run contains `config.json`, `data_audit.json`, `history.json`,
`model.pt` or `model.ubj`, `thresholds.json`, `metrics.json`, `predictions.npz`,
`curves.npz`, `summary.json`, and `status.json`. Curves store ROC and PR coordinates
for plotting without another prediction pass. Failed runs retain status and diagnostics.

## Frozen evaluation after selection

Training and result comparison do not open or score test shards. To verify a saved
checkpoint or explicitly evaluate the selected model:

```powershell
python train/evaluate.py --run <selected-run> --split validation
python train/evaluate.py --run <selected-run> --split test
# A moved dataset must have an identical manifest.
python train/evaluate.py --run <selected-run> --dataset "D:/shared/dataset" --split test
```

Evaluation verifies frozen model/threshold hashes and writes a new evaluation
directory. It never refits weights, transforms or thresholds. It uses all real
examples in the requested split with their original prevalence.

Historical project work already inspected this same Scheme2 test split. Keeping
it out of this round's selection prevents additional direct test tuning, but it
does not make the split newly untouched. A fresh holdout is needed for a new final
generalization claim after test-informed development. The time-block split also
does not guarantee unseen hosts/campaigns. This supervised screen is distinct from
the earlier temporal AE + train-fitted linear readout experiment.

## Focused implementation checks

```powershell
python -m unittest train.tests.test_screening -v
```

Checks use small artificial Scheme2-shaped fixtures: all four CPU training paths,
available CUDA paths, frozen checkpoint reload and validation/test evaluation,
independent-run collection, CPU fallback, optional-MPS device selection, absent-test
training, tied-score thresholds, group weights, checksum corruption and incompatible-comparison rejection. Fixture metrics are
not project performance results. No full real-data training is part of these checks.

API references: [XGBoost training and callbacks](https://xgboost.readthedocs.io/en/stable/python/python_api.html),
[PyTorch LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html),
[PyTorch MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html),
[sklearn AP definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).
