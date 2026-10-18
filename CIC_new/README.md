# CIC dataset preparation

This directory contains the CIC-only EDA and the bucket builder with feature v1 and split policy v2. It does not train a model.
The implementation is documented in [preprocessing_v1.md](preprocessing_v1.md).
The agreed split uses **global 15min blocks**, class-coverage reservations, a **100s guard on both sides** of fold
transitions and actual flow-end checks. All sources in a block share its assignment.

## Use the completed local dataset

Branch: `codex/cic-timeblock-buckets`. Code and reports are committed on this branch.
Raw data, generated shards/caches and the existing `.venv-cnn` environment stay on this machine and remain outside Git.
Switching branches does not remove those local files. A fresh clone on a different machine does not include them.

```powershell
.\.venv-cnn\Scripts\python.exe CIC_new/load_dataset.py
```

This read-only command checks completed-run metadata, all shard filenames and the first/last shard arrays of each split.
It uses the existing NumPy installation and does not download, install or rebuild anything.
The active run is fixed by [local_dataset.json](local_dataset.json): `buckets_v2_20261018_200028`.

| Split | Sequences | NPZ shards |
|---|---:|---:|
| Train, including augmentation | 781,838 | 382 |
| Validation, real only | 251,146 | 123 |
| Test, real only | 263,125 | 129 |

Every sample has ten buckets and **228 features per bucket**, with mask separate.
The full build completed in approximately 23.5 minutes; all 22 final checks passed.
The normalized shards occupy approximately 270 MB. Raw SQLite, event cache, raw features and prior EDA caches are retained too.

From Python launched in the repository root:

```python
from CIC_new.load_dataset import iter_shards

for batch in iter_shards("train"):
    X, mask, y = batch["X"], batch["mask"], batch["y_sequence"]
    # Consume this shard before requesting the next one.
```

## Optional: rebuild from the existing local database

The prepared data is ready to consume. Run the following only to create a new full build:

```powershell
.\.venv-cnn\Scripts\python.exe CIC_new/build_buckets.py
```

Defaults are `datasets/cic_2018_traffic.db`, source-IP windows of 100 seconds,
ten 10-second buckets, a target **60/20/20** split of retained originals, and seed **4266**.
Protect all attack types in training after exclusions, then greedily reserve feasible validation/test coverage.
The remaining whole blocks are allocated to approach the target proportions, with three deterministic corrections for purge loss.
No model score or seed search is used. Actual proportions and missing types are reported.
Keep all retained original train windows; append P B+A and P B+B windows, where P is the retained original train positive count.
The script never edits the source SQLite. It does not use the NB15 schema or require earlier EDA outputs.

The single Python entry point uses `bucket_features.py`, `split_policy.py`, `render_bucket_summary.mjs` and `summary_metrics.md` alongside it.
Keep these files together when cloning or sharing the project.

The full-data split preflight retained 765,652 / 251,146 / 263,125 original train/validation/test sequences
(59.82% / 19.62% / 20.56% of retained originals), with 14 / 13 / 12 attack types.
186,742 boundary/interval windows were excluded. Hulk remains train-only; Slowloris is absent from test.
P=8,093 gives 781,838 final training sequences after augmentation. The completed full build matches this preflight.
Preflight counts are archived in [reports/split_policy_v2_preflight.json](reports/split_policy_v2_preflight.json).
Full results are in [reports/buckets_v2_20261018_200028/](reports/buckets_v2_20261018_200028/).
The evaluation measures held-out time blocks in the same environment. Source IPs and attack campaigns can recur across sets.

## Excel runtime

Python analysis requires only NumPy. Excel is authored using the Codex bundled Node/artifact-tool runtime already available on this machine.
The script checks that runtime before the long scan. Defaults are:

- executable: `~/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe` on Windows;
- packages: `~/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules`.

If the runtime is installed elsewhere, pass `--node <executable>` and `--node-modules <node_modules directory>`.
No modification or package installation into the bundled runtime is performed. A local runtime junction is created inside the ignored output directory.

## Outputs and Git

Every run prints two locations immediately:

| Directory | Contents | Git policy |
|---|---|---|
| `CIC_new/outputs/buckets_v2_<timestamp>/` | Compressed dataset shards, raw feature shards, SQLite manifest/event cache, transforms, logs, stage status, report previews | Ignored |
| `CIC_new/reports/buckets_v2_<timestamp>/` | `bucket_summary.xlsx`, `metrics_explained.md`, `summary.json`, `split_notes.json`, `feature_schema.json`, `scaler.json` | **Not ignored; commit these** |
| `CIC_new/reports/EDA/` | Archived final EDA workbooks | **Not ignored; commit these** |

The timestamp uses UTC+08. The completed full run is `buckets_v2_20261018_200028`; its reports are already archived.
The test suite uses an artificial database and keeps its reports outside the published report area.

To choose a larger disk or a stable run name:

```powershell
python CIC_new/build_buckets.py --output "E:/cic_local/buckets_run1" --reports-dir "CIC_new/reports/buckets_run1"
```

The local event cache and raw shards are deliberately retained for provenance, checking and stage-level recovery.
They are not additional training samples. The normalized files under `splits/` are the model-ready dataset.
Compression and bounded-size shards limit RAM use; runtime and peak disk usage for the full CIC build have not been benchmarked.

## Resume after interruption

The six stages are `index`, `split`, `plan`, `aggregate`, `normalize`, `report`.
Completed stages are reused; the first incomplete stage restarts. Recovery is stage-level, not row-level.
For example, interruption halfway through aggregation restarts aggregation but reuses indexing, splitting and the synthesis plan.

```powershell
python CIC_new/build_buckets.py --output "CIC_new/outputs/buckets_v2_<original timestamp>" --resume
```

Use the same `--db`, seed, shard size, cache size and code files as the original run. The report path is recovered from the saved configuration.
Old v1 random-window outputs cannot be resumed with split policy v2; start a new output directory for this protocol.
`run_status.json` records completed stages and any exception, while `run.log` records progress.
Never run two processes against one output directory. Normal exit or Ctrl+C removes the lock.
After a forcibly killed process, confirm it has stopped, then remove only that output's `run.lock` before resuming.

Rebuild reports after normalization without repeating dataset preparation:

```powershell
python CIC_new/build_buckets.py --report-only "CIC_new/outputs/buckets_v2_<timestamp>"
```

## Read the finished data

```python
from pathlib import Path
import json
import numpy as np

run = Path("CIC_new/outputs/buckets_v2_<timestamp>")
schema = json.loads((run / "feature_schema.json").read_text())
for file in sorted((run / "splits/train").glob("part_*.npz")):
    with np.load(file, allow_pickle=False) as batch:
        X = batch["X"]                 # float32: (samples, 10, F)
        mask = batch["mask"]           # uint8:   (samples, 10)
        y = batch["y_sequence"]        # uint8:   (samples,)
        y_bucket = batch["y_bucket"]   # uint8:   (samples, 10)
        sample_id = batch["sample_id"] # int64:   (samples,)
        # Consume this shard before opening the next one.
```

F is 82 fixed statistics + V training-L7 ratio slots, where V includes UNK.
Mask is separate; concatenating it for a model gives width F+1. No learned embedding is required.
`flow_counts` and `attack_flow_counts` accompany every shard for audits.
Matching raw shards contain `X_raw` as float64, with the same IDs and row order.

`manifest.sqlite` contains:

- `original_sequences`: real source IP, absolute window, actual maximum flow end, raw flow counts, assigned and retained splits;
- `split_groups`: each global block's assignment and coverage reservation flag;
- `sequence_attacks`: all attack memberships/counts per original window;
- `samples`: real/synthetic type, donor IDs, second-donor offset, requested attack, expected flows and final shard location;
- `destinations`: equality IDs for destination-diversity calculations, never direct input values.

Labels, IP literals, sample type and donor provenance remain outside X. Batch balancing and training loss are later decisions.
Excluded originals remain in the audit manifest/cache but never enter samples, shards, donors, vocabulary or scaler fitting.
`exclusion_reason` is a bit mask: 1=boundary guard, 2=interval overlap, 3=both, 0=retained.

## Validation

```powershell
python CIC_new/test_build_buckets.py
python CIC_new/test_split_policy.py
```

Five focused tests cover known feature values, unknown categories, conditional denominators, duplicate starts,
whole-donor shifts, long flow ends, normalization of empty buckets, complete global-block construction,
train-only vocabulary/scaler fitting, augmentation counts, original-source immutability, stage recovery,
report-only export and cached Excel formulas. Two additional policy checks cover boundary equality and rare-type reservations.
The integration fixture has only 840 flows, not the real CIC data.

See [summary_metrics.md](summary_metrics.md) for report-column definitions and [EDA/README.md](EDA/README.md) for the earlier analyses.
