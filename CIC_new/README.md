# CIC-IDS2018 workflow

Start with [preprocessing/README.md](preprocessing/README.md).
The current path follows sequence construction → bucket aggregation → split →
training-only embedding-space superposition. Shared statistics and split functions
live inside preprocessing; no old CIC builder is required.

| Folder | Purpose |
|---|---|
| [preprocessing](preprocessing/README.md) | Prepare raw data and export reusable augmented datasets |
| [representation_learning](representation_learning/README.md) | Learn and freeze input embeddings, then encode original sequences |
| [modelling](modelling/README.md) | Optional downstream AE architecture and training, consuming exported data |
| tests | Small fixture checks; no full dataset or GPU required |
| [EDA](EDA/README.md) | Research analyses used to investigate dataset choices |
| [reports](reports/README.md) | Historical EDA and V2 results, retained as references |

## Run the current workflow

From the repository root, install preprocessing dependencies:

```bash
python -m pip install -r CIC_new/preprocessing/requirements.txt
python -m CIC_new.preprocessing.run_preprocessing \
  --csv datasets/NF-CICIDS2018-v3.csv \
  --out CIC_new/outputs/preprocessed
```

This command finishes raw preprocessing and writes real train/validation/test
splits, schema, scaler and audit metadata. It does not train neural networks.
It targets approximately 60/20/20 using global time blocks, with coverage
reservations and temporal exclusions; achieved proportions and missing attack
types are reported rather than assumed.

For the professor's embedding-space augmentation, next follow the
[representation-learning commands](representation_learning/README.md) to learn
and freeze embeddings and encode all three splits. PyTorch is needed for that
stage. Then export a dataset that any final architecture can consume:

```bash
python -m CIC_new.preprocessing.export_augmented_dataset \
  --data CIC_new/outputs/preprocessed \
  --encoded CIC_new/outputs/encoded_sequences \
  --out CIC_new/outputs/model_inputs
```

The exporter retains all real training sequences and adds P synthetic benign and
P synthetic malicious sequences, where P is the real malicious training count.
Only the optional training sampler balances classes; exported real data are retained.
The [modelling guide](modelling/README.md) gives the separate AE training command.
Fresh output directories prevent accidental overwrites.

## Check the code without the full dataset

With NumPy, pandas and PyTorch installed:

```bash
python -m unittest discover -s CIC_new/tests -t . -v
```

The fixture checks preprocessing, splitting, training-only fitting, circular
mixing, dataset export and short CPU training runs. They do not establish full
dataset runtime or detection accuracy.

## Earlier work

The previous raw-flow superposition builder, its loader/report renderer and the
older CIC pilot/merged-dataset builder were removed from this branch.
Their source is recoverable from `origin/codex/cic-timeblock-buckets`.
Generated local datasets, trained results and checkpoints were not deleted.
Historical report files remain unchanged and are not evidence that the revised
embedding pipeline has already been run on the full corpus.
