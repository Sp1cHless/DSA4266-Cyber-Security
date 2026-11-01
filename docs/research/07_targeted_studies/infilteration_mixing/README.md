# Infilteration Mixing Diagnostics

Report sections: 6.3. Organized on 1 November 2026 from completed project records.

## Finding

The largest post hoc strict-threshold loss groups have different responses: TCP/3389 cases have low median attack-flow share and largely cross the threshold after benign-flow removal, while attack-heavy UDP/53 cases largely remain below threshold under the extension. Similar overall AP conceals different responses to attack evidence.

## Evidence scope

Post hoc groups among 23 losses after the distribution extension. Oracle deletions use flow labels and unchanged per-pipeline thresholds; intervention FPR is not recalibrated. These are diagnostic responses, not detection gains.

## Report figures

- I2: [PNG](<figures/fig_infilteration_mixing.png>) and [SVG](<figures/fig_infilteration_mixing.svg>).

## Compact results

- [figure_provenance.json](<results/figure_provenance.json>)

The shared intervention comparison is [Table V1](<../results/table_v1_targeted_comparison.csv>).

## Code and protocol entry points

- [analysis/inspect_distribution_infilteration_cases.py](<../../../../analysis/inspect_distribution_infilteration_cases.py>)
- [analysis/review_distribution_feature_results.py](<../../../../analysis/review_distribution_feature_results.py>)

## Local supporting records

`local/` preserves the original dated insertion drafts, plotting workspaces, source screenshots and superseded displays for this topic. It is ignored by Git. These files are historical support; the current DOCX determines final section placement. Full experiment runs, arrays and checkpoints retain their original paths under the ignored runtime directories. Source paths and hashes in captured provenance describe the original records, not bundled datasets.

## Saved evidence locations

These original records retain their existing paths; full runs and private evidence maps are intentionally local.

- `train/runs/20261025_201726_319757_lstm_distribution_features_seed4266_01c453/reports/20261025_213233_854544_infilteration_changed_cases_0cf499/flow_family_profiles.csv`
