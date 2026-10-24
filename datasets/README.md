# Local datasets

Place the team's full-schema CIC-IDS2018 SQLite source at
`datasets/cic_2018_traffic.db` (table `network_flows`) to build new Scheme2 data.
The individual preprocessing command also accepts `NF-CICIDS2018-v3.csv`.
`NetFlow_v3_Features.csv` is the tracked source feature reference.

For downstream work, place the complete delivered Scheme2 dataset at
`datasets/dataset/`. It must include `manifest.json`, `validation.json`, schema,
scaler, frozen encoder, donor plan, metadata sidecars and all train/validation/test
NPZ shards. Do not rename shards or edit frozen manifests. Existing absolute paths
inside manifests record provenance and are not downstream runtime dependencies.

Source data and generated datasets are ignored by Git and require a separate
team handoff. Cloning this branch does not download them. Build instructions are
in [`CIC_new/SCHEME2_DATASET.md`](../CIC_new/SCHEME2_DATASET.md); model instructions
are in [`train/README.md`](../train/README.md).
