# Scheme2 data workflow

This directory ends at a checked, model-independent train/validation/test dataset.
Downstream classification training, evaluation and diagnostics live in
[`train/`](../train/README.md).

```text
Full-schema CIC SQLite source
  -> 100-second source-IP sequences and ten 10-second buckets
  -> approximate 6:2:2 temporal split and boundary exclusions
  -> vocabularies and scalers fitted on real training buckets
  -> whole-bucket input encoder fitted on train, selected on validation
  -> frozen encoding of original train/validation/test sequences
  -> training-only zero-shift vector mixtures and flow-count channel
  -> validated final dataset with checksums and provenance
```

Run from the repository root after [environment/data setup](../README.md):

```powershell
python -u CIC_new/prepare_scheme2_dataset.py
python -u CIC_new/prepare_scheme2_dataset.py --db <source.db> --out <fresh-workflow-directory>
```

Both commands build new data; they are alternatives. The default reserves a unique
`outputs/scheme2_runs/<timestamp>_scheme2_bucket64_shift0_seed4266_<suffix>/`.
Completed output includes `workflow.json`, `preprocessed/`, `bucket_embeddings/`,
`encoded_sequences/`, `donor_plan.npz`, and the final `dataset/`.
Only the final `dataset/` directory is required for downstream model training.

| Source | Responsibility |
|---|---|
| `prepare_scheme2_dataset.py` | Orchestrate all five stages and validate exports |
| `preprocessing/` | Build sequences/buckets, split originals, fit transforms, select donors and export inputs |
| `representation_learning/` | Fit/freeze the input encoder and encode originals |
| `tests/` | Bucket statistics, temporal split and database-to-dataset fixture checks |

Read [SCHEME2_DATASET.md](SCHEME2_DATASET.md) for the construction contract,
[preprocessing/README.md](preprocessing/README.md) for raw preparation and
[representation_learning/README.md](representation_learning/README.md) for the
input encoder. No Scheme1 source or historical detector is required.

```powershell
python -m unittest discover -s CIC_new/tests -t . -v
```

Tests use small artificial flow records and temporary output directories. They
do not scan the full source database or establish real-data accuracy.
