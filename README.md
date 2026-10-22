# Network traffic time-series project

**Scheme2 is the selected CIC-IDS2018 superimposition method.** It freezes learned
whole-bucket embeddings, adds donor vectors, applies L2 normalization and retains
a separate flow-count channel. The current configuration uses 16-dimensional
tables for protocol, L7 and destination-service categories, 64-dimensional bucket vectors, 65 final input
features, zero displacement and the approximate 6:2:2 time-block split.

**Scheme2 has completed one full temporal-AE training run, validation and a real
held-out test evaluation.** The run was conducted during method comparison and
serves as the current Scheme2 experimental baseline for reporting and subsequent
work. Test ROC-AUC was 0.84948 and average precision was 14.93%, using a frozen
AE and a linear readout fitted on real training data. See
[the completed run and results](CIC_new/SCHEME2_EXPERIMENT.md) for metrics,
artifact locations and evaluation scope.

For a fresh checkout, create an environment from the repository root in a Windows
VS Code terminal. Skip creation if you already have a suitable environment:

```powershell
python -m venv .venv-cnn
.\.venv-cnn\Scripts\python.exe -m pip install -r CIC_new\preprocessing\requirements.txt
.\.venv-cnn\Scripts\python.exe -m pip install torch
```

The environment, source data and completed outputs are not included in Git.
Obtain the team's full-schema `cic_2018_traffic.db` and place it in `datasets/`;
the default source table is `network_flows`. For another database location, pass
`--db <path>`. The [preprocessing guide](CIC_new/preprocessing/README.md) also
documents CSV input through the individual stage commands.

Run the active workflow from the repository root:

```powershell
# Build a NEW dataset only when regeneration is needed.
.\.venv-cnn\Scripts\python.exe -u CIC_new\prepare_scheme2_dataset.py
# Check the latest completed dataset without training.
.\.venv-cnn\Scripts\python.exe CIC_new\run_scheme2_experiment.py --stage check
# Train a NEW AE run, then evaluate its frozen checkpoint on validation.
.\.venv-cnn\Scripts\python.exe -u CIC_new\run_scheme2_experiment.py
```

On the experiment machine, completed Scheme2 data and models are under
`CIC_new/outputs/`. Other members must obtain the completed dataset separately or
build it from the source; cloning the repository does not download these files.
When using a shared dataset, pass `--dataset <dataset-directory>` explicitly to
the check/training command. Copy the entire final `dataset/` directory, including
its manifests, donor plan, metadata sidecars and train/validation/test shards.
New builds, training runs and evaluations use separate identifiable directories.
See [CIC_new/README.md](CIC_new/README.md) for the directory map and stage commands.
Scheme1 is retained as source for historical reproduction. Shared conclusions
and test reports are in
[the comparison archive](CIC_new/reports/superimposition_comparison_20261022/SUMMARY.md).
Full evidence, configurations, histories and cleanup audits remain in the ignored
local portion of that archive. Ignore rules are centralized in the root .gitignore.
Historical EDA and unrelated UNSW/temporal experiments remain in their existing
folders. Raw datasets, virtual environments and generated outputs are local and
ignored by Git. Project files are written in English.
