# Network traffic time-series project

The current CIC-IDS2018 workflow is in [CIC_new](CIC_new/README.md), on
`feature/cic-embedding-superposition`. Start with its preprocessing guide.

The stages are deliberately separate:

- **Preprocessing:** raw flows → source-IP sequences → ten 10-second buckets →
  train/validation/test splits → training-only scaling and category vocabularies.
- **Representation learning:** learn categorical and bucket embeddings on real
  training data, select the checkpoint on validation, then freeze the encoder.
- **Dataset export:** circularly shift and add training donor embeddings, apply
  L2 normalisation, and export reusable inputs. Validation/test stay real-only.
- **Modelling:** train the chosen detector on the exported inputs. The supplied
  temporal autoencoder is one optional model; CNN/LSTM/classifier experiments can
  read the same dataset.

See [preprocessing](CIC_new/preprocessing/README.md),
[representation learning](CIC_new/representation_learning/README.md) and
[modelling](CIC_new/modelling/README.md) for commands and output definitions.

Raw data and generated artifacts are not included in Git. Use
`datasets/NF-CICIDS2018-v3.csv` or a full-schema SQLite source. Keep datasets,
caches and checkpoints on persistent storage and never run two builders against
one output directory.

Earlier CIC raw-flow augmentation and pilot scripts were removed from this branch;
their source remains on `origin/codex/cic-timeblock-buckets`.
[CIC EDA](CIC_new/EDA/README.md) and [archived reports](CIC_new/reports/README.md)
remain as research references. Their historical dataset counts and feature schemas
do not describe newly generated embedding inputs.

Existing UNSW-NB15 scripts, analyses and experiment notes remain for reference;
they are not prerequisites for the current CIC pipeline.
