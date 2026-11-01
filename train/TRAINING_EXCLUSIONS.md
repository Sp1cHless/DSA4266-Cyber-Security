# Training Exclusion Code and Version Boundaries

The report evaluates two fixed primary-label exclusion sets: A jointly removes
XSS, SlowHTTPTest and HOIC, while B removes all seven DoS/DDoS types. These are
two fixed comparisons rather than an exhaustive leave-one-out rotation.

## Available implementation

| Entry point | Responsibility |
|---|---|
| [seed_report.py](seed_report.py) | Aggregate supplied completed control/holdout predictions, pair seeds and retain the saved validation thresholds |
| [export_holdout_lstm_projector.py](export_holdout_lstm_projector.py) | Historical subtype-exclusion representations on matched validation IDs |
| [export_flooding_projector.py](export_flooding_projector.py) | Broad-family-holdout test and training-reference representation exports |
| [diagnose_flooding_projector.py](diagnose_flooding_projector.py) | Native-dimensional neighbors using the full training reference |

`seed_report.py` is byte-identical to the team code recorded at commit
`44ef865be777140b1ba34e7b0abe4ec156ea8708`. Its SHA256 and restoration source are
recorded in [TEAM_CODE_PROVENANCE.json](../docs/code/TEAM_CODE_PROVENANCE.json).
It writes a fresh `train/runs/` output by default and does not train models or
select new thresholds. Run from the repository root:

```powershell
python train/seed_report.py --help
```

Use repeated `--arm NAME=DIRS` arguments for completed run/evaluation directories,
including one arm named `control`. Supply each holdout arm's dataset manifest or
explicit held-out labels. The helper requires matching model families, seeds,
sample IDs and labels. Detailed arguments are documented by `--help`.

## Construction and result availability

The team protocol's `--holdout-train-attacks` construction interface belongs to
its code revision. It is not an argument of the current working-tree
`CIC_new/prepare_scheme2_dataset.py`. The named `train/unseen_subtype_report.py`
entry point from that protocol is also absent in this working tree. The original
protocol snapshot remains local at
`docs/research_map_20261029/team_snapshot/HOLDOUT_PROTOCOL.md`.

Existing delivered holdout datasets can be consumed by compatible model and
geometry entry points. Restoring the seed aggregator does not supply the team's
raw per-seed predictions or recreate its upstream holdout builder. Table H1
remains checked against the committed summary, rather than independently
recomputed from those unavailable arrays.

See [the study page](../docs/research/08_training_exclusions/README.md) for the
results and [the code guide](../docs/code/README.md) for the full dependency map.
