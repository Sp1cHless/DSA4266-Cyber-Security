# Paired multi-seed circular-shift diagnostic

Run XGBoost, LSTM, CNN and CNN + LSTM with seeds **4266, 4267, 4268, 4269, 4270**.
Each family/seed has a chronological control (start 0) and a middle-start rotation
(start 5). Both training and real validation use the same input rule. Defaults
give **40 fits** with the existing per-family architectures, group weights,
optimization budgets, checkpoint selection and validation threshold policy.

## Run with the current completed shift batch

From the repository root, reuse the four compatible seed-4266 shifted fits:

```powershell
.\.venv-cnn\Scripts\python.exe train/run_circular_shift_multiseed.py --reuse-batch train/runs/20261024_171214_153612_four_models_circular_start5_seed4266_194bef
```

This checks the prior fits before starting and schedules **36 new fits**, including
new seed-4266 chronological controls in the current environment. Referenced fits
are read in place; their predictions, thresholds and weights are preserved.
The older no-shift screening runs use a different software environment and are
not used as controls for this study.

To run all 40 fits from scratch:

```powershell
.\.venv-cnn\Scripts\python.exe train/run_circular_shift_multiseed.py
```

Use `--plan-only` with either command to validate the setup and create a saved
plan without fitting. The terminal prints the exact new study directory under
`train/runs/`. All fits run sequentially. Default `auto` uses CPU for XGBoost and
available CUDA for neural models, otherwise CPU. `--device cpu` requests all CPU;
`--device cuda` requests CUDA for all four. Optional MPS uses CPU for XGBoost;
actual MPS execution remains unverified on this Windows machine.

## Resume and report again

```powershell
# Run a saved plan or continue an interrupted study.
python train/run_circular_shift_multiseed.py --resume "<study-directory>"
# Regenerate a fresh report from completed fits without training.
python train/run_circular_shift_multiseed.py --summarize "<study-directory>"
```

Resume/summarize use the saved dataset, models, seeds, cache, devices and budgets;
do not repeat configuration flags. Resume validates completed fits and skips them.
A failed/incomplete fit gets a fresh retry directory, retaining its prior files.
Each report goes into a fresh `reports/` subdirectory. Existing experiment data
and model weights are never overwritten. The plan retains external reuse paths,
so keep the reused batch accessible.

Reuse and resume check dataset fingerprint, seed, transform, architecture,
training settings, thread count, device, Python/package/platform metadata,
relevant training-source hashes, and model/threshold/artifact hashes. Changed
training source or environment requires a fresh study. The only accepted source
migration is an explicitly reviewed pair of exact hashes for `common/data.py`
and `common/experiment.py`: replacing hardcoded shapes with manifest timesteps
and adding rejection guards for non-ten-bucket shifts. It applies only to the
original ten-bucket dataset and is recorded per reused run in `plan.json`.
Artificial-fixture checks verify identical loaded arrays, flattened inputs and
group weights under the reconstructed historical and current loader versions.
Unknown source changes remain rejected. Study planning reads
metadata and hashes only; it performs no training or test scoring. Reporting
checks validation sample IDs, labels, attack names and source hosts against a
saved validation index. It reads original validation masks once to define
active-bucket counts; train/test shards are not used for these slices.

## Outputs

`plan.json` freezes the settings and run references. `status.json` tracks progress
and points to the latest report. `configs/` stores the per-model configurations.
New fits live under `seedNNNN/MODEL/start0/` and `start5/`. A compact
`validation_index.npz` records validation IDs, labels, attack names, hosts and
original active-bucket counts.

Each report contains:

- `COMPARISON.md`: overall paired differences and Web sparsity tables.
- `per_seed.csv`: no-shift, shifted and delta values for each family/seed/metric.
- `aggregate.csv`: means, sample SDs, delta range and positive/negative/zero seed counts.
- `per_attack_per_seed.csv` and `per_attack_aggregate.csv`: all attack recalls
  under each run's global <=1% validation FPR threshold.
- `web_per_seed.csv` and `web_aggregate.csv`: Web recall at global <=1% and
  <=0.1% validation FPR thresholds, grouped as all, one, two, and three-or-more
  active buckets. Per-seed rows also retain gained/lost detections and counts.
- `summary.json`: aggregate values, comparison contract and interpretation limits.

Rate values are percentages, and their deltas are **percentage points**.
ROC-AUC values and deltas use their original absolute scale. For each seed,
`delta = shifted - no_shift`; aggregate SD is computed from those paired deltas
with `ddof=1`. One-seed SD and empty-slice recalls are N/A, not zero.
No run is selected as a best seed and no seed predictions are ensembled.

## Optional changes and interpretation

`--models lstm cnn hybrid` limits the study to three neural families.
`--seeds 4266 4267 4268` changes the seed list. These are new-study options.
`--dataset`, `--cache`, `--device`, `--threads`, `--circular-shift-start` and a
fresh `--out` are supported. Budget overrides `--epochs`, `--batch-size`,
`--rounds` and `--patience` apply identically within every relevant pair.
Reused runs must match these settings; reduced-budget checks cannot be mixed
with completed full-budget project runs.

All fits reuse the same original training cohort and fixed real validation split.
Different seeds estimate training variability conditional on this split; they
do not add independent hosts or campaigns. Five seeds are a practical starting
budget, not an automatic significance guarantee. Validation selects checkpoints
and thresholds, and all Web groups use those global thresholds without fitting
slice-specific cutoffs. Historical work already inspected this test split.
These are validation perturbation diagnostics, not independent final test results.

Circular shifting preserves bucket contents, occupancy counts and most adjacent
pairs while changing boundary positions. Retraining allows adaptation. Repeated
small deltas do not establish that temporal order is unimportant, and a consistent
Web effect still needs positional/score-distribution analysis before claiming a
specific mechanism.
