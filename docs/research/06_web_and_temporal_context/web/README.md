# Sparse Web Evidence

Report sections: 6.1. Organized on 1 November 2026 from completed project records.

## Finding

Of 181 Web validation sequences, 171 contain attack flows in one bucket and 96 have only one active traffic bucket. XGBoost has higher mean recall on the single-active-bucket group; LSTM is stronger on the two-bucket and three-or-more groups. Local and recurrent models show complementary detection profiles.

## Evidence scope

Observational groups with the original masks and five original-order seeds. Each fit uses its own global validation threshold. Occupancy is not an experimentally randomized factor.

## Report figures

- W1: [PNG](<figures/fig_web_local_evidence.png>) and [SVG](<figures/fig_web_local_evidence.svg>).

## Compact results

- [figure_provenance.json](<results/figure_provenance.json>)

## Code and protocol entry points

- [analysis/diagnose_web_temporal_structure.py](<../../../../analysis/diagnose_web_temporal_structure.py>)
- [train/LSTM_WEB_FINDINGS.md](<../../../../train/LSTM_WEB_FINDINGS.md>)

## Local supporting records

`local/` preserves the original dated insertion drafts, plotting workspaces, source screenshots and superseded displays for this topic. It is ignored by Git. These files are historical support; the current DOCX determines final section placement. Full experiment runs, arrays and checkpoints retain their original paths under the ignored runtime directories. Source paths and hashes in captured provenance describe the original records, not bundled datasets.

## Saved evidence locations

These original records retain their existing paths; full runs and private evidence maps are intentionally local.

- `train/runs/20261025_013446_294043_web_temporal_support_diagnostic_7c620b/cohort_profiles.csv`
- `train/runs/20261025_013446_294043_web_temporal_support_diagnostic_7c620b/web_recall_by_occupancy.csv`
