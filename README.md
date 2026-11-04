# Temporal Network Intrusion Detection

NF-CICIDS2018-v3 flows are grouped into source-host windows, encoded as frozen
bucket embeddings, and classified by CNN, LSTM, CNN-LSTM or XGBoost. The baseline
input has ten positions with 64 embedding coordinates, one intensity channel and
a separate activity mask. Vector superimposition augments training only.

| Directory | Contents |
|---|---|
| [CIC_new/](CIC_new/README.md) | Preprocessing, temporal splitting, input encoder, dataset construction and fixture checks |
| [train/](train/README.md) | Models, shared runtime, experiment drivers, evaluation and diagnostics |
| [analysis/](analysis/README.md) | Post-fit research diagnostics and historical report-tool source |
| [docs/research/](docs/research/README.md) | Study README files and compact CSV results |
| [docs/code/](docs/code/README.md) | Source/configuration inventory and module dependencies |
| [datasets/](datasets/README.md) | Existing local data; left unchanged during cleanup |
| `archive/` | Local recovery material, ignored by Git |

## Run

Run commands from the repository root. Create a fresh environment and install
the requirements for the workflow you need:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r train/requirements.txt
.\.venv\Scripts\python.exe train/lstm/train.py --device cpu
```

Training reads `datasets/dataset/` by default. See [the training guide](train/README.md)
for other models, devices, paired comparisons and frozen evaluation.

To build a new dataset, install
`CIC_new/representation_learning/requirements.txt` and run:

```powershell
python CIC_new/prepare_scheme2_dataset.py
```

The default source is `datasets/cic_2018_traffic.db`. See
[the construction guide](CIC_new/SCHEME2_DATASET.md) for source options and the split
protocol. New outputs use unique directories; completed experiment data and
weights must not be overwritten.

## Repository scope

Git contains source, fixed configurations, requirements, tests, README files and
small selected result tables. Report documents, exported figures, renderings,
complete runs, prediction arrays, caches and local environments are excluded.
The 4 November 2026 cleanup moved old material to `archive/cleanup_20261104_01/`;
its relocation manifest preserves original paths. `datasets/` was not moved or
edited. Archiving consolidates local files; it does not free their disk space.

New runs may recreate the ignored output/cache paths documented by each workflow.
Historical diagnostics requiring saved predictions need the corresponding archived
inputs restored or supplied explicitly. Historical report helpers are retained
under `analysis/report_tools/` as source snapshots with documented input needs.

## Result interpretation

Keep full-source EDA, reused-training probes, oracle stress diagnostics,
validation model selection and historical-test evaluations distinct. Synthetic
mixtures reuse training donors. Repeated seeds measure training variability on
fixed samples. The historical test split has already been inspected; it is not
an untouched final holdout. [Each study](docs/research/README.md) records its scope.
