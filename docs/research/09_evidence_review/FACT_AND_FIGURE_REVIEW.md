# Report fact and figure review

Review date: 30 October 2026. Navigation updated on 1 November 2026; the findings below refer to the audited baseline report. Scope: the revised Sections 5 and 6, with baseline population and inherited EDA provenance checks. No models were trained, experiment data changed, or new detection predictions generated. Restored plots use completed summaries only.

## Fact checking

The machine-readable audit is `output/report_factcheck_20261030/fact_checks.json`; the checker is `output/report_factcheck_20261030/audit_facts.py`. It checks 101 entries and assertions against 22 hashed source files. A further 33 source paths/hashes in existing figure provenance records match the available files. This verifies saved evidence and report transcription; it is not a full rerun of the raw-flow pipeline.

| Report evidence | Original record and verification | Interpretation retained |
|---|---|---|
| Table I | `datasets/dataset/manifest.json`, exact real/synthetic split counts | Synthetic data are training-only; validation/test are real |
| R1 representation comparison | Archived `three_way_comparison.json` and `test_metrics.json`, linked with exact paths/hashes in the audit; every printed AP, recall and FPR matches | Complete representation pipelines; separate from the supervised detector comparison |
| M1 and circular shift | Five-seed `aggregate.csv` and `per_attack_aggregate.csv` in run `20261024_223102_671768_circular_multiseed_start5_ef34ba`; means, SDs and five negative Web pairs checked | Shifted models are refitted; circular rotation preserves most local adjacency |
| D1 attack geometry | [Original source index](../04_attack_profiles/results/figure_provenance.json) | Projector pictures are attack-enriched PCA views; native 64D neighbor statistics are computed separately |
| S1 host intervention | [Original source index](../05_host_information/results/figure_provenance.json), saved probe/detection CSVs and integrity audit | Independent probes reuse detector-training records; probe metric is balanced accuracy. Host suppression is partial and detection AP changes |
| W1 Web evidence | [Original source index](../06_web_and_temporal_context/web/results/figure_provenance.json), fixed 181-example cohort and five-seed summaries | Conditional model profiles on observational occupancy groups |
| T1 temporal scale | Original `paired_summary.csv`, `validation_comparison.csv` and `attack_comparison.csv`, all plotted recalls cross-checked | Common-anchor activity is distinct from model comparisons on rebuilt cohorts. At 400 seconds Web n=117, Infilteration n=45; historical n=181/366 |
| Pure Infilteration ambiguity | Saved raw-flow `audit.json` and synthesis in run `20261024_122800_754357_pure_infilteration_raw_flow_audit_6146f6` | Ten of 34 misses and ten of 104 detections have identical benign counterparts; ambiguity exists before encoding |
| V1 interventions | Completed attention/distribution/bucket `comparison.json` files, all AP and subtype recall entries checked | Same-runtime reference, one seed, own validation thresholds; distribution extension includes encoder refitting |
| I2 traffic mixing | Saved `flow_family_profiles.csv` in distribution run `20261025_201726_319757_lstm_distribution_features_seed4266_01c453` | Post hoc groups among 23 losses at the strict budget; label-based flow removal uses unchanged thresholds and uncalibrated intervention FPR |
| Bucket localization | Saved `bucket_predictions.npz` aligned by sample ID to `predictions.npz`; independently recomputed 135 eligible Infilteration sequences, 116 top-bucket hits, uniform expectation 54.3798%; saved bucket AP=82.8775% | Localization is diagnostic, not the sequence checkpoint criterion or a held-out test improvement |
| H1 three-seed transfer | `docs/research_map_20261029/team_snapshot/HOLDOUT_PROTOCOL.md`, all printed means/SDs and paired deltas match | Two fixed exclusions, not exhaustive leave-one-out. Raw per-seed prediction arrays are not present locally, so the summary was transcribed and checked, not independently rescored |
| H2 subtype-exclusion geometry | [Original source index](../08_training_exclusions/results/figure_provenance.json), matched validation IDs in a separate historical one-seed comparison | Independent PCA axes and refitted preprocessing; spatial displacement is not a causal measure |
| H3 flooding diagnostic | Saved one-seed test/neighbor records in runs `20261029_192411_050197_all_flooding_test_projector_699e44` and `20261029_192945_155514_flooding_full_training_neighbors_56c0cf` | Separate from H1. Full 780,899-row training reference includes synthetic rows; the displayed Projector cohort is smaller |

Recorded protocol checks cover the original representations, paired donor/sample controls, circular shift/refitting, host-adversarial supervision, attention pooling, the fifteen-feature extension, auxiliary bucket targets and the primary-label holdout construction. They use saved run protocols, comparison records and integrity checks. This review does not independently rerun every training configuration or the approximately 20-million-flow reconstruction.

## Figure decisions

No fixed figure quota was used. Each retained visual must answer a different question from its neighbors.

| Item | Decision | Contribution |
|---|---|---|
| R1, M1, V1, H1 tables | Retain | Exact comparisons need aligned values; no duplicate performance bar charts |
| D1 | Retain | Establishes contrasting subtype distributions before the host/context analyses |
| S1 | Retain and enlarge labels | Connects host information, suppression and detection tradeoffs to the later research focus |
| W1 | Restore in 6.1 | Shows the occupancy-by-model interaction and five-seed variability hidden by overall metrics |
| T1 | Restore in 6.1 | Shows all temporal geometries, both budgets and changing sample support; prevents a misleading reading of the best recall |
| I2 | Restore in 6.3 | Distinguishes sparse attack mixtures from attack-heavy low-score inputs and contrasts reference/extension responses |
| H2 | Retain | Teacher-introduced Projector views support the related-subtype transfer interpretation on matched samples |
| H3 | Retain | Explains the composition of broad-family-holdout misses; its purpose differs from H2 |
| C1 attention performance chart | Keep out of main text | Sequence metrics duplicate V1; attention weights and the single-/multi-bucket contrast remain in prose |
| B1 bucket chart | Keep out of main text | Sequence panel duplicates V1; localization is conveyed clearly by 116/135 versus 54.4% in prose |
| Pure-case count table | Keep out of main text | Small comparisons remain compact in prose, including collisions in detected cases |

W1, T1, I2 and S1 were replotted with larger text for the report width. Old figure files remain intact. New plot inputs and hashes are in [figure_provenance.json](results/figure_provenance.json) and [host_figure_provenance.json](results/host_figure_provenance.json).

## Edits and preservation

- Corrected the host-probe metric wording to **balanced accuracy**.
- Added explicit temporal cohort support and benign counterparts among pure detections.
- Identified the provenance of the team three-seed summary.
- Restored three figures for distinct analytical purposes without adding duplicate performance charts.
- Preserved all native equations, original embedded media, and all body XML outside Sections 5 and 6. An original-document backup is stored in `_local_archive/report_factcheck_20261030/`.

At the audit date, Sections 1, 2, 7, 8 and the appendix contained existing team placeholders; this review does not imply that the entire report is submission-ready.

Original audit arrays, helpers, private team snapshots and backups named above are intentionally local. The [research index](../README.md), [artifact inventory](../ARTIFACTS.csv) and [saved-evidence routing](../EXPERIMENTS.csv) provide the current shared navigation.
