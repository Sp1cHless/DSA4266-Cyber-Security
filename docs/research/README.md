# Research Index

This directory keeps compact findings and selected completed results. It contains
no report drafts, exported figures, model weights, prediction arrays or caches.

| Study | Contents |
|---|---|
| [01 Data exploration](01_data_exploration/README.md) | Type/host coverage, sparsity, temporal scale and split feasibility |
| [02 Representation](02_representation/README.md) | Representation and reconstruction comparisons |
| [03 Model selection](03_model_selection/README.md) | Four detector families, five seeds and paired rotation diagnostics |
| [04 Attack profiles](04_attack_profiles/README.md) | Native-space neighborhood comparisons |
| [05 Host information](05_host_information/README.md) | Host probes and suppression/detection tradeoffs |
| [06 Web and temporal context](06_web_and_temporal_context/README.md) | Sparse evidence, model complementarity and longer-window comparisons |
| [07 Targeted studies](07_targeted_studies/README.md) | Input ambiguity, attention, feature probes, distribution extension and bucket supervision |
| [08 Training exclusions](08_training_exclusions/README.md) | Historical three-seed summaries and distinct single-seed diagnostics |

## Folder contract

- `README.md`: finding, evidence scope and source entry points.
- `results/`: compact CSV tables and limited machine-readable summaries.
- [RESULTS.csv](RESULTS.csv): file hashes, source paths and evidence scope. CSV
  copies preserve their original bytes; JSON table exports preserve saved values.

Full local evidence is consolidated in the ignored `archive/` directory. Original
paths in source records are historical identifiers; the local relocation manifest
maps them into the archive. They are absent from a clean clone. No model was
trained, threshold selected or prediction rescored during cleanup.

Keep full-source EDA, reused-training probes, oracle stress diagnostics,
validation comparisons and historical-test results distinct. The three-seed
holdout table is an inherited summary; separate local single-seed exports must
not be treated as its raw replications. See [the code guide](../code/README.md)
for implementation and version boundaries.
