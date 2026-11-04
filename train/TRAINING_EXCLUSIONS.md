# Training Exclusions: Workflow Routing

The final report's Phase II uses the independent
[holdout workflow](../experiments/holdout/README.md), retained from
`feature/cic-holdout` at `e0529b6d60aca85b23128e0788c26d0640ef1698`.

| Purpose | Canonical implementation |
|---|---|
| Build control/holdout data | `experiments/holdout/CIC_new/prepare_scheme2_dataset.py` |
| Train selected LSTM | `experiments/holdout/train/lstm/train.py` |
| Frozen evaluation | `experiments/holdout/train/evaluate.py` |
| One control/holdout pair | `experiments/holdout/train/unseen_subtype_report.py` |
| Paired three-seed aggregation | `experiments/holdout/train/seed_report.py` |

Read [the protocol](../experiments/holdout/CIC_new/HOLDOUT_PROTOCOL.md) for A,
B-1 and B-2, source choices and evaluation boundaries. Root `train/seed_report.py`
retains the same aggregation implementation for existing analysis commands.
Root `CIC_new/` remains the Phase I/variable-temporal workflow.

The final report's recorded results are indexed by the
[training-exclusion study](../docs/research/08_training_exclusions/README.md).
The earlier three-subtype/four-model summary and later local single-seed geometry
diagnostics are retained separately. Raw per-seed arrays are still needed to
independently rescore published three-seed results; integration does not create
missing predictions or rerun real-data experiments.
