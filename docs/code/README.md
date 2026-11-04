# Code Guide

The inventories below cover root Phase I/diagnostic source and the independent
Phase II holdout workflow. See `INVENTORY_SUMMARY.json` for current counts.
Root preprocessing, model and analysis source retains the cleaned baseline
version; imported Phase II runtime source retains its committed version.

| Layer | Location | Responsibility |
|---|---|---|
| Data construction | [CIC_new](../../CIC_new/README.md) | Sequence/bucket features, split policy, training-only transforms, donor plan and export |
| Input encoder | [representation_learning](../../CIC_new/representation_learning/README.md) | Train and freeze bucket embeddings |
| Shared training runtime | [train/common](../../train/common/README.md) | Data validation, weights, training, thresholds and evaluation |
| Baseline models | [CNN](../../train/cnn/README.md), [LSTM](../../train/lstm/README.md), [hybrid](../../train/hybrid/README.md), [XGBoost](../../train/xgboost/README.md) | Architectures, configurations and entry points |
| Extensions | [Attention](../../train/lstm_context/README.md), [bucket supervision](../../train/lstm_bucket/README.md), [host adversarial](../../train/lstm_host_adversarial/README.md) | Targeted LSTM variants |
| Independent Phase II | [experiments/holdout](../../experiments/holdout/README.md) | Matching fixed-window preprocessing, runtime, LSTM and holdout aggregation |
| Diagnostics | [analysis](../../analysis/README.md), `train/audit_*.py`, `train/diagnose_*.py`, `train/export_*.py` | Post-fit comparisons, raw-flow audits and representation export |
| Visualization | `train/*part6*.py`, `train/part6_viewer_template.html` | Saved-model representation views and viewer composition |
| Historical report helpers | [report_tools](../../analysis/report_tools/README.md) | Source snapshots of retired preview/plot workflows |
| Fixture checks | `CIC_new/tests/`, `train/tests/` | Small artificial implementation checks |

- [CODE_INDEX.csv](CODE_INDEX.csv): paths, roles, imports, definitions and hashes.
- [MODULE_DEPENDENCIES.csv](MODULE_DEPENDENCIES.csv): static source-to-source import edges.
- [INVENTORY_SUMMARY.json](INVENTORY_SUMMARY.json): current structural counts.
- [TEAM_CODE_PROVENANCE.json](TEAM_CODE_PROVENANCE.json): restored seed-aggregation source version.
- [Research index](../research/README.md): compact findings and result tables.

Runtime paths and imports remain stable. The five scripts formerly stored with
report drafts are grouped under `analysis/report_tools/` without changing their
bytes. Their original inputs and output bundles are archived; read that folder's
README before attempting historical rendering.

Run documented model commands from the complete repository root. Model folders
depend on the shared `train/` package. Static import counts do not include every
dynamic dispatch; baseline architecture loading uses `importlib`.

## Independent workflow versions

Root `CIC_new/` and `train/` cover Phase I, temporal studies and diagnostics.
`experiments/holdout/` retains the separate Phase II implementation from commit
`e0529b6d60aca85b23128e0788c26d0640ef1698`, including its own `CIC_new` and
`train` packages. Its source manifest verifies imported bytes. The duplicate
package names preserve that workflow's original imports and require a separate
Python invocation; do not combine imports across source roots in one process.

Both code inventories record workflow scope. Static imports are resolved within
their own source root. Final-report tables and earlier diagnostic summaries are
kept separately. Read [WORKFLOWS.md](../WORKFLOWS.md) and
[TRAINING_EXCLUSIONS.md](../../train/TRAINING_EXCLUSIONS.md) before reproducing a
study. Integrating source does not reconstruct missing raw prediction arrays.

## Generated material

Complete historical runs, environments, caches, report documents and exported
figures are consolidated in the ignored local archive. New runs may recreate
ignored runtime directories. Fixture/CLI checks verify implementation paths;
they do not reproduce research metrics or create a new held-out assessment.
