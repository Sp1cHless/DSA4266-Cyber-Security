# Phase II: Independent Attack-Holdout Workflow

This is the `feature/cic-holdout` implementation from commit
`e0529b6d60aca85b23128e0788c26d0640ef1698`, retained as a separate workflow for
Section VI of the final report. Its source, model settings and fixed ten-bucket
preprocessing are independent of the root temporal-scale/diagnostic workflow.
All imported Python, PowerShell, configuration and requirements files retain
their original bytes. [SOURCE_MANIFEST.csv](SOURCE_MANIFEST.csv) records origins
and identifies documentation updates.

## Layout

| Folder | Responsibility |
|---|---|
| [CIC_new](CIC_new/README.md) | Fixed 100-second windows, ten 10-second buckets, control/holdout datasets and input encoders |
| [train](train/README.md) | Matching detector runtime, frozen evaluation and single-/multi-seed holdout aggregation |
| [Protocol](CIC_new/HOLDOUT_PROTOCOL.md) | A, B-1 and B-2 labels, checks and selection boundaries |
| [Published results](../../docs/research/08_training_exclusions/README.md) | Final-report LSTM tables, historical summaries and separate local diagnostics |

## Environment and commands

Commands below run from the **repository root**. Install
`train/requirements.txt` and `CIC_new/preprocessing/requirements.txt` in a fresh
environment. Always pass the input path explicitly: the snapshot's original
default dataset path is relative to this independent workflow, not the root
`datasets/` directory.

Build a fresh control and a fresh Set A dataset:

```powershell
python experiments/holdout/CIC_new/prepare_scheme2_dataset.py --csv datasets/NF-CICIDS2018-v3.csv --out experiments/holdout/CIC_new/outputs/control_run
python experiments/holdout/CIC_new/prepare_scheme2_dataset.py --csv datasets/NF-CICIDS2018-v3.csv --out experiments/holdout/CIC_new/outputs/set_a_run --holdout-train-attacks "Brute_Force_-XSS,DoS_attacks-SlowHTTPTest,DDOS_attack-HOIC,SSH-Bruteforce"
```

These are new-build examples, not cleanup commands. Use unique output names;
existing completed outputs are preserved. For SQLite, replace `--csv` with `--db`.
Train LSTM on each arm using its own exported `dataset/` directory and seeds
4266, 4267 and 4268:

```powershell
python experiments/holdout/train/lstm/train.py --dataset experiments/holdout/CIC_new/outputs/set_a_run/dataset --seed 4266
python experiments/holdout/train/evaluate.py --run <run> --dataset <matching-dataset> --split test --out <run>/test_eval
python experiments/holdout/train/unseen_subtype_report.py --control <control-run>/test_eval --holdout <holdout-run>/test_eval --holdout-dataset <holdout-dataset>
python experiments/holdout/train/seed_report.py --arm control=<control4266>,<control4267>,<control4268> --arm set_a=<holdout4266>,<holdout4267>,<holdout4268> --arm-dataset set_a=<holdout-dataset>
```

The final report uses LSTM for Phase II. Other baseline families and their tests
remain in this source snapshot for completeness. Avoid mixing root and snapshot
runtime imports within one Python process. Run the snapshot's tests from its
own directory so its `CIC_new` and `train` packages resolve together:

```powershell
cd experiments/holdout
python -m unittest discover -s CIC_new/tests -t . -v
python -m unittest train.tests.test_seed_report train.tests.test_unseen_subtype_report -v
```

## Scope

Exclusion is by primary sequence label after the unchanged split is allocated.
Non-primary residual flows are audited rather than individually removed.
Validation retains excluded types for checkpoint and threshold selection.
Historical-test results, validation diagnostics and reused-training probes remain
distinct. Report CSVs are transcriptions of recorded results, not new experiments
or independent rescoring. Generated data, caches and runs are ignored by Git.
