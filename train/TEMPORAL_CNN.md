# Four temporal configurations: baseline-matched CNN train/validation

The selected main-model comparison now uses LSTM; see
[TEMPORAL_LSTM.md](TEMPORAL_LSTM.md). This CNN entry point remains available as a
separate optional experiment. Both entry points share the model-neutral driver
in `common/temporal_experiment.py` and use their own historical model reference.

`run_temporal_cnn.py` sequentially trains one fresh CNN for each of the four
completed temporal preparations. It reuses their original buckets, splits,
vocabularies and scalers. It never rebuilds raw buckets or resplits the database.
The existing 100s/10s CNN is read as a reference and is not trained again.

## Baseline references and fixed settings

The default classifier reference is:

`train/cnn/runs/20261023_024348_386209_cnn_seed4266_84d958/`

The input-encoder reference is the `bucket_embeddings/config.json` under:

`CIC_new/outputs/scheme2_runs/20261022_115847_160650_scheme2_bucket64_shift0_seed4266_0fhh3gjt/`

The runner reads these saved settings and verifies the reference model/threshold
hashes. It does not infer training defaults from a possibly edited config file.
There are no epoch, optimizer or model-width override flags.

| Component | Historical setting retained |
|---|---|
| CNN | Conv1d 66 -> 64 -> 128, kernels 3, padding 1; masked mean/max pooling; 64-unit head; dropout 0.2 |
| Optimization | AdamW; lr 0.001; weight decay 0.0001; gradient norm cap 1.0 |
| Budget | Up to 30 epochs; patience 5; batch size 512; every training example once per shuffled epoch |
| Reproducibility | Seed 4266; four CPU threads for CNN; CUDA backend as in the historical run |
| Objective | Each of four real/synthetic benign/malicious groups contributes 25%; weights are recomputed for the new population using the same formula |
| Selection | sklearn average precision on the full real validation population |
| Thresholds | Select each run's thresholds by the unchanged validation FPR 1% / 0.1% rule; retain the 0.5 operating point |
| Input order | Chronological; circular shift explicitly zero |
| Bucket encoder | 5,000 steps; batch 512; lr 0.001; category dimension 16; bucket dimension 64; validation every 200 steps; validation limit 4,096; two CPU threads |
| Augmentation | Same seed offset, attack/source/occupancy sampling, P examples per synthetic class, zero displacement, frozen-vector addition and L2, scaled flow-count channel |

CNN kernels, channel counts, pooling and head sizes remain unchanged for T=10,
20 and 40. The loader reads T from the dataset manifest; sequences are not resized
or cropped to ten positions. This preserves the architecture while the requested
bucket duration changes the physical time covered by the same kernel.

The existing `train.common.experiment` and `torch_runner` perform optimization,
checkpoint selection, threshold selection and evaluation. The batch driver does
not implement a second training loop. Fresh models are initialized with the
baseline seed; they are not fine-tuned from the historical CNN weights.

## Use the existing environment

Use the computer's current `.venv-cnn` environment without creating an environment
or installing/downgrading packages. Historical/current package and backend
differences are recorded in output metadata and do not block training. No
`--allow-runtime-drift` flag is required. The archived architecture, optimizer,
budgets, checkpoint selection and threshold rules remain unchanged.

## Run all four cases

First preview the stages without creating outputs or fitting weights:

```powershell
.\.venv-cnn\Scripts\python.exe -u train/run_temporal_cnn.py --dry-run
```

Then run:

```powershell
.\.venv-cnn\Scripts\python.exe -u train/run_temporal_cnn.py
```

The runner selects the latest batch with exactly these completed preparations:

| Sequence seconds | Bucket seconds | Buckets |
|---:|---:|---:|
| 200 | 10 | 20 |
| 200 | 20 | 10 |
| 400 | 10 | 40 |
| 400 | 40 | 10 |

It prints the selected preparation batch. To make that selection explicit:

```powershell
.\.venv-cnn\Scripts\python.exe -u train/run_temporal_cnn.py --prepared-batch CIC_new/outputs/temporal_ablation_runs/20261024_184722_957599_b919ae
```

Each case performs:

1. Fit a new bucket encoder on the existing real training buckets, select its
   checkpoint on validation, then freeze it. Refit is necessary because the
   aggregation distribution/transforms have changed.
2. Encode and export **train and validation only**, using the same zero-shift
   Scheme2 arithmetic and validation checks. Test feature shards are not read,
   encoded, exported, cached or scored. The original manifest/SQLite catalog
   still contains split/label/ID metadata for all originals, as required by the
   unchanged donor-ID construction; this is not test-feature use or test fitting.
3. Train the fresh CNN and evaluate the selected checkpoint on real validation,
   using the unchanged baseline selection/threshold policy.

The raw preparation remains complete for all original splits. This workflow's
new exported input dataset intentionally contains only train/validation. The
full Scheme2 builder retains its original three-split default; the new runner
explicitly passes `--splits train validation`.

## Outputs and resume

A unique output directory is reserved under `train/runs/` and printed. It contains:

- `cnn_config.json`: architecture/training settings copied from the saved baseline.
- `batch.json`: frozen preparation hashes, reference hashes, runtime, source
  hashes, settings and completed-stage records.
- Per-case `representation_attempt_0001/` with encoder, frozen inputs and
  construction checks; `cnn_attempt_0001/` with standard model/metrics/predictions;
  sibling logs and disposable CNN caches.
- `validation_comparison.csv` and `.json`: the existing baseline row plus the four
  new validation rows, including AP, ROC-AUC, actual FPR, global recalls and
  Infilteration/Web primary-type counts and recall at 1% FPR. Partial reports are
  written as cases complete. Missing types have null recall, not zero recall.

Resume the same output with the same references, preparation selection, Python
path and runtime:

```powershell
.\.venv-cnn\Scripts\python.exe -u train/run_temporal_cnn.py --out <training-batch-directory> --resume
```

Completed representations and CNNs are hash-checked and reused. A failed stage
is retained; retry uses a fresh numbered attempt. An incomplete encoder stage
restarts its fitting budget, and an incomplete CNN restarts from initialization.
Existing completed experiment data and weights are not replaced. If a process
was forcibly killed, verify no builder is active before removing its stale
`RUNNING.lock`. The shared driver removes train/validation mmap caches and encoded
intermediates after verified representation export, and removes each detector's
loading cache after verified training. Final data, checkpoints and results remain;
cache JSON provenance and embedded weights are copied to `cache_cleanup/` first.
Interrupted train/validation `.partial` caches are removed before retry. Each new
stage checks estimated allocations plus a 2 GiB reserve before it starts.
`--resume --out <existing-batch> --cleanup-only` reclaims these disposable caches
without training. See the [LSTM storage policy](TEMPORAL_LSTM.md) for the narrow,
audited compatibility migration from the reviewed original driver.

## Interpretation and checks

These are real validation results selected on validation, not final independent
test results. Training examples are reused for fitting under the original
objective and are not reported as held-out evidence. Different sequence durations
have different cohorts, split assignments, prevalence and class coverage; the
comparison deliberately produces no paired-recall delta or significance claim.
The historical test split has previously been inspected elsewhere in the project.

Small artificial checks can be run with:

```powershell
.venv-cnn\Scripts\python.exe -m unittest train.tests.test_temporal_cnn -v
```

They use explicitly separate one-step/one-epoch fixture reference files, never
modify production reference budgets, and hide test feature directories to require
the complete train/validation path to succeed without them. They also exercise
runtime-difference recording without blocking, all four geometries, baseline-forward identity, equal
group weights, checkpoint reload and exact completed-stage reuse. These checks
are implementation evidence, not real-data performance measurements.
