# Phase II Detector Training and Holdout Evaluation

Use this runtime with the matching [Phase II preprocessing](../CIC_new/README.md).
The final report selects LSTM and compares control, Set A, B-1 and B-2 with seeds
4266-4268. A detector remains binary; it does not classify the excluded subtype.

| Entry point | Role |
|---|---|
| [lstm/train.py](lstm/train.py) | Train the report's selected LSTM; pass `--dataset` explicitly |
| [evaluate.py](evaluate.py) | Reload a frozen checkpoint and saved validation thresholds |
| [unseen_subtype_report.py](unseen_subtype_report.py) | Compare one control/holdout pair with matching evaluation IDs and labels |
| [seed_report.py](seed_report.py) | Pair multiple seeds and aggregate held-out recall and AP |
| [common](common/README.md) | This version's data contract, fitting loop and output helpers |
| [tests](tests/README.md) | Artificial runtime and aggregation checks |

Other model folders retain historical source and configuration. Reported Phase II
results concern LSTM; the root four-model validation comparison is a different
study. Full commands are in the [workflow guide](../README.md).

Evaluate each control/holdout run with its own frozen thresholds and write test
predictions to `<run>/test_eval` when using `seed_report.py`. That helper accepts
a training-run directory containing `test_eval/`, or the `test_eval/` directory
itself. It does not accept arbitrary timestamped evaluation directories. Use a
fresh `--out` for every evaluation and aggregation.

Validation retains excluded types for selection. Historical test inspection and
host/campaign overlap limit zero-day interpretation. Saved complete runs and
prediction arrays are local artifacts; only compact result tables are shared.
