# Independent Experiment Workflows

| Workflow | Report role | Entry point |
|---|---|---|
| [Phase I and diagnostics](../README.md) | Baseline comparison, temporal-scale studies and discussion diagnostics | Root `CIC_new/`, `train/` and `analysis/` |
| [Phase II holdout](holdout/README.md) | Fixed 100s/10s control and attack-exclusion experiments | `holdout/CIC_new/` and `holdout/train/` |

The two workflows retain their own preprocessing and detector runtime versions.
Choose a workflow before constructing data or running models. The holdout source
is versioned independently because the temporal extensions in the Phase I tree
were not the implementation used for the reported Phase II experiments.
