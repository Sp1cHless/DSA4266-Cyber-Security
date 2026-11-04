# Phase II Dataset Construction

This version constructs fixed 100-second source-host windows with ten 10-second
buckets. It supports a full-schema SQLite or CSV source and primary-label
training exclusions. It does not implement the root workflow's variable temporal
geometries.

| Folder or file | Role |
|---|---|
| `prepare_scheme2_dataset.py` | Preprocess, fit the bucket encoder, encode originals, superimpose training donors and validate the export |
| [preprocessing](preprocessing/README.md) | Sequence construction, split allocation, exclusion audit and train-only transformations |
| [representation_learning](representation_learning/README.md) | Train and freeze the input encoder |
| [tests](tests/README.md) | Artificial construction and holdout checks |
| [HOLDOUT_PROTOCOL.md](HOLDOUT_PROTOCOL.md) | Final-report experiment definitions |

Use the root-relative commands in the [workflow guide](../README.md). Pass source
paths explicitly and write fresh outputs under this workflow's ignored
`CIC_new/outputs/`. Downstream training consumes the exported `dataset/` directory.
Source implementation and requirements retain the committed holdout version.
