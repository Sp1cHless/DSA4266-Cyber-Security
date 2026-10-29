# Scheme2 network traffic classification

This branch contains the Scheme2 data construction workflow and the downstream
models in `train/`. Scheme2 converts CIC-IDS2018 flows into frozen bucket vectors,
adds zero-shift training donor vectors, applies L2 normalization, and appends a
separate flow-count channel. Default final input shape: `[N, 10, 65]`.

| Path | Purpose |
|---|---|
| `CIC_new/prepare_scheme2_dataset.py` | One entry point from source database to checked train/validation/test inputs |
| `CIC_new/preprocessing/` | Sequence construction, bucket statistics, temporal split, scaling, donor plans, export and metadata |
| `CIC_new/representation_learning/` | Fit the input bucket encoder, then freeze and encode originals |
| `CIC_new/tests/` | Small data-construction checks using artificial fixtures |
| `datasets/` | Local source database and delivered final dataset; see its README |
| `train/` | Current model training, comparison, evaluation and diagnostics |
| `docs/RESEARCH_FINDINGS.md` | Compact research conclusions and selected numerical results |

## Use an existing dataset

Place the complete delivered final dataset under `datasets/dataset/`. Keep its
manifests, donor plan, frozen encoder, metadata sidecars and all three shard folders.
Install `train/requirements.txt`, then follow [the training guide](train/README.md).
Training reads the delivered dataset directly; no data rebuild is required.

## Build a new dataset

Run from the repository root. An existing Python environment can also be used.

```powershell
python -m venv .venv-cnn
.\.venv-cnn\Scripts\python.exe -m pip install -r CIC_new/representation_learning/requirements.txt
.\.venv-cnn\Scripts\python.exe -u CIC_new/prepare_scheme2_dataset.py
```

The default source is `datasets/cic_2018_traffic.db`, table `network_flows`.
Use `--db <path>` for another full-schema SQLite source. The builder reserves a
unique directory under `CIC_new/outputs/scheme2_runs/` and prints its final
`dataset/` path. Pass that path to training with `--dataset <path>` or copy the
complete final directory to `datasets/dataset/`.

See [the data workflow](CIC_new/README.md) and
[the construction protocol](CIC_new/SCHEME2_DATASET.md) for stage and split details.
The bucket encoder is an input transform; downstream detector training belongs
to `train/`.

## Repository scope

Scheme1, old temporal-AE trainers/evaluators, exploratory analyses, unrelated
dataset workflows, old reports and historical experiments are stored locally
under `_local_archive/`. That directory is ignored by Git. Its dated cleanup
directory includes a move ledger and original copies of edited files.
Source data, completed datasets, checkpoints and generated model results remain
local. Existing completed results are preserved; new builds require fresh outputs.

Dataset exports placed under `train/dataset*/` and generated arrays, weights and
serialized models are also ignored. Detailed `docs/research_map_*/` evidence
inventories remain local, preserving their original paths for research review.
Share [the compact research findings](docs/RESEARCH_FINDINGS.md), source code,
experiment guides and fixed configuration files. Full run directories, prediction
arrays, data shards, caches and machine-specific cleanup records stay local.

The same historical Scheme2 test split has already been inspected in project work.
It is not a newly untouched holdout. Synthetic training examples reuse real
training donors and are not independent observations. Fixture checks verify
implementation behavior, not detection performance. Project files use English.
