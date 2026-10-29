# Scheme2 sequence and bucket ablations

`run_temporal_ablation.py` runs the existing Scheme2 pipeline sequentially with
different temporal geometry. Its default stage ends after raw sequence creation,
bucket aggregation, train/validation/test splitting and train-only transforms.
It does not train a CNN, LSTM, hybrid classifier or XGBoost model.

## Run the expensive raw preparations first

Run from the repository root with the existing NumPy environment:

```powershell
.venv-cnn\Scripts\python.exe -u CIC_new/run_temporal_ablation.py --dry-run
.venv-cnn\Scripts\python.exe -u CIC_new/run_temporal_ablation.py
```

The first command prints the four stage commands without creating outputs or
scanning source flows. The second executes them one at a time. It prints and
reserves a unique batch directory under
`CIC_new/outputs/temporal_ablation_runs/`.

To choose a descriptive directory explicitly, use a **new** path:

```powershell
.venv-cnn\Scripts\python.exe -u CIC_new/run_temporal_ablation.py --out CIC_new/outputs/temporal_ablation_runs/window_bucket_seed4266
```

The default source is `datasets/cic_2018_traffic.db`. Use `--db <source.db>` to
select another full-schema source. Batch processing does not modify this source.
Each preparation builds its own event cache and raw shards; allow substantial
disk space for all configurations rather than only one Scheme2 run.

## Default comparisons

The ordered list is in `configs/temporal_ablation.json`:

| Sequence seconds | Bucket seconds | Bucket count | Comparison |
|---:|---:|---:|---|
| 200 | 10 | 20 | Longer sequences at fixed bucket duration |
| 200 | 20 | 10 | Longer sequences with ten buckets |
| 400 | 10 | 40 | Longer sequences at fixed bucket duration |
| 400 | 40 | 10 | Longer sequences with ten buckets |

The existing Scheme2 100-second / 10-second / ten-bucket artifacts are the
baseline. This batch does not rebuild them. The four new configurations compare
200- and 400-second sequences while either retaining 10-second buckets or
retaining ten buckets per sequence.

Supply `--config <cases.json>` for a different ordered list. Each case accepts
only positive integer `sequence_seconds` and `bucket_seconds`; sequence duration
must be divisible by bucket duration. Bucket count is derived, not independently
set. The existing on-disk provenance types limit this implementation to 127
buckets per sequence. Duplicate configurations and unknown case fields fail
before processing.

For fewer comparisons, copy the config to a new file and retain the desired
cases. A config has this structure:

```json
{
  "cases": [
    {"sequence_seconds": 200, "bucket_seconds": 10},
    {"sequence_seconds": 200, "bucket_seconds": 20}
  ]
}
```

## What remains identical to Scheme2

- Source-IP grouping and epoch-aligned **nonoverlapping** windows use
  `floor(flow_start_ms / sequence_ms) * sequence_ms`. This is not the diagnostic
  sliding/past-context expansion used in earlier exploratory reports.
- A complete flow belongs to its start bucket; feature definitions, service
  groups, categorical counts, labels and primary-attack tie-breaking are unchanged.
  Empty buckets remain zero with a separate mask.
- All original raw buckets are computed before splitting or transform fitting.
- The global 15-minute block policy, approximate 60/20/20 targets, seed 4266,
  coverage reservations, balancing passes and **100-second embargo** are unchanged.
  Window-support and actual-flow-end overlap calculations use the requested
  sequence duration. The embargo is not silently increased to 200/400 seconds.
- Only retained real training active buckets fit numeric log1p/standardization
  and category vocabularies. Held-out-only categories remain UNK.
- If full dataset export is requested, the existing input encoder architecture,
  64-dimensional bucket vectors, 5,000 fitting steps, validation selection budget,
  seed offsets and learning rate remain unchanged. Changed preparations get their
  own fitted encoder and transforms.
- Training-only synthetic donor sampling retains the existing attack/source
  balancing and occupancy bands: 1, 2-4, and at least 5 active buckets. Each
  synthetic class has P samples, where P is the real training positive count.
  Zero displacement, vector addition, L2 normalization and the scaled flow-count
  channel are unchanged. Validation/test contain real originals only.

Changing sequence duration changes original sequence identities, counts and
composition. The same split algorithm is rerun on that population; global block
assignments are **not** frozen from the old 100-second run. Thus identical seeds
do not guarantee identical folds across sequence durations. Bucket-duration-only
changes at a fixed sequence duration preserve original identities and split
membership. This workflow is a preparation ablation, not a paired-recall claim.

## Outputs and resume

The batch directory contains:

- `batch.json`: exact cases, source file signature, Python path, pipeline source
  hashes, common settings, stage attempts and completion state.
- `index.csv`: each configuration's completed preparation/dataset path and the
  real train/validation/test counts.
- One case directory with `preprocess_attempt_0001/` and its sibling log. The
  preparation contains the same Scheme2 `manifest.sqlite`, `work.sqlite`,
  `raw_originals/`, `splits/`, schema, scaler and split diagnostics.

Resume with the original batch path and the same common settings:

```powershell
.venv-cnn\Scripts\python.exe -u CIC_new/run_temporal_ablation.py --out <batch-directory> --resume
```

Completed stages are checked against their recorded metadata hashes and skipped.
A partially completed stage is retained and restarted from its beginning in a
new `attempt_0002/` directory. There is no claim of resuming halfway through a raw
database scan or bucket shard. By default a failed case stops the batch; use
`--keep-going` to try later cases and receive a nonzero final exit status if any
case failed. Resume must use the original case list, source signature, Python
path, pipeline code and common settings.

`RUNNING.lock` prevents concurrent processes using one batch directory. A normal
exit or Ctrl+C removes it. If the process was forcibly killed, first verify that
no process is using the batch, then remove only this stale lock before resuming.
Completed data, interrupted outputs and model weights are never deleted by the
runner. Do not run a second builder against an active batch.

## Later: build complete Scheme2 classifier inputs

After all raw preparations finish, optionally reuse them for input-encoder
fitting, frozen encoding, zero-shift augmentation and complete export validation:

```powershell
.venv-cnn\Scripts\python.exe -u CIC_new/run_temporal_ablation.py --out <batch-directory> --resume --stage dataset
```

This sequentially runs the existing `prepare_scheme2_dataset.py --prepared ...`
workflow for each case. No downstream detector is trained. Each completed case
gets `dataset_attempt_0001/dataset/` containing `[N,T,65]` inputs. Defaults use
automatic CUDA/CPU selection. If you want a fixed device or other common encoder
settings, supply them already on the first preprocessing command and preserve
them when resuming; the runner rejects settings drift.

The downstream loader now supports `[N,T,65]`. For the four-case main LSTM baseline
comparison, use [`train/run_temporal_lstm.py`](../train/TEMPORAL_LSTM.md), which
reuses these preparations and encodes/trains/validates without reading test
feature shards. Its chronological LSTM keeps the original architecture and
training settings. The separate circular-shift experiment retains its ten-bin
contract. The raw preparation artifacts remain model-independent.

Single configurations can also use the existing entry points directly:

```powershell
.venv-cnn\Scripts\python.exe -u -m CIC_new.preprocessing.run_preprocessing --db datasets/cic_2018_traffic.db --out <fresh-preparation> --sequence-seconds 200 --bucket-seconds 10
.venv-cnn\Scripts\python.exe -u CIC_new/prepare_scheme2_dataset.py --out <fresh-workflow> --prepared <completed-preparation> --sequence-seconds 200 --bucket-seconds 10
```

The full builder rejects a prepared directory whose temporal geometry differs
from its requested configuration. It continues to support completed historical
100s/10s preparations and their input checkpoints when fingerprints match.

## Verification and interpretation

Run the small synthetic checks with:

```powershell
.venv-cnn\Scripts\python.exe -m unittest discover -s CIC_new/tests -t . -v
```

The temporal tests exercise all four default configurations through preprocessing and
full Scheme2 export with a one-step fixture encoder. They check geometry,
bucket-boundary assignment, unchanged embargo/block policy, leakage exclusions,
bucket-only split identity, histogram/mask reconciliation, natural held-out
populations, exact reuse, failed-stage preservation and retry in a fresh folder.
The existing baseline tests continue to exercise arithmetic export validation,
fixed donors and corruption rejection. These fixtures are not real validation
or test performance results.

During implementation, the pre-change baseline was also captured on the same
artificial source: all 670 arrays across 67 raw/split shards and all 300 original
sequence/split records matched the parameterized default exactly; scaler values
matched as well. No real-data CNN recall improvement is established by these
checks. Preparing or encoding test data does not evaluate a detector, and the
historical test split is not a newly untouched holdout.
