# Code Guide

The active inventory contains **124 source/configuration resources**,
including **112 Python modules**. Existing training, preprocessing and
analysis source bytes were preserved during the 4 November 2026 cleanup.

| Layer | Location | Responsibility |
|---|---|---|
| Data construction | [CIC_new](../../CIC_new/README.md) | Sequence/bucket features, split policy, training-only transforms, donor plan and export |
| Input encoder | [representation_learning](../../CIC_new/representation_learning/README.md) | Train and freeze bucket embeddings |
| Shared training runtime | [train/common](../../train/common/README.md) | Data validation, weights, training, thresholds and evaluation |
| Baseline models | [CNN](../../train/cnn/README.md), [LSTM](../../train/lstm/README.md), [hybrid](../../train/hybrid/README.md), [XGBoost](../../train/xgboost/README.md) | Architectures, configurations and entry points |
| Extensions | [Attention](../../train/lstm_context/README.md), [bucket supervision](../../train/lstm_bucket/README.md), [host adversarial](../../train/lstm_host_adversarial/README.md) | Targeted LSTM variants |
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

## Holdout version boundary

The restored `train/seed_report.py` aggregates supplied predictions with saved
thresholds. Its team holdout builder belongs to a different source revision:
the active builder does not expose `--holdout-train-attacks`, and
`train/unseen_subtype_report.py` is absent. Supplied holdout datasets and inherited
three-seed summaries do not constitute a full upstream reconstruction. See
[TRAINING_EXCLUSIONS.md](../../train/TRAINING_EXCLUSIONS.md).

## Generated material

Complete historical runs, environments, caches, report documents and exported
figures are consolidated in the ignored local archive. New runs may recreate
ignored runtime directories. Fixture/CLI checks verify implementation paths;
they do not reproduce research metrics or create a new held-out assessment.
