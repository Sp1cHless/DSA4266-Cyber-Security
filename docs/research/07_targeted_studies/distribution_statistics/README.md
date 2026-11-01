# Distribution Statistics and Feature Probes

Report sections: 6.3. Organized on 1 November 2026 from completed project records.

## Finding

Training-only feature probes motivate a fifteen-statistic extension. The resulting refitted pipeline gives similar overall validation AP (61.50% versus 61.53%), improves Web recall and reduces Infilteration recall. At the strict budget, 23 former Infilteration detections are lost and five are recovered.

## Evidence scope

Training-only probe development is distinct from the one-seed validation detector comparison. The feature extension includes a new bucket encoder, so the comparison evaluates the combined pipeline change.

## Presentation

The current report uses concise prose or a consolidated table for this study. Earlier standalone figures and insertion drafts remain local rather than duplicating the main evidence.

The shared intervention comparison is [Table V1](<../results/table_v1_targeted_comparison.csv>).

## Code and protocol entry points

- [analysis/INFILTERATION_FEATURE_AUDIT.md](<../../../../analysis/INFILTERATION_FEATURE_AUDIT.md>)
- [analysis/audit_infilteration_features.py](<../../../../analysis/audit_infilteration_features.py>)
- [analysis/recheck_infilteration_feature_probes.py](<../../../../analysis/recheck_infilteration_feature_probes.py>)
- [train/DISTRIBUTION_FEATURE_ABLATION.md](<../../../../train/DISTRIBUTION_FEATURE_ABLATION.md>)
- [train/run_distribution_feature_ablation.py](<../../../../train/run_distribution_feature_ablation.py>)
- [CIC_new/prepare_distribution_features.py](<../../../../CIC_new/prepare_distribution_features.py>)

## Local supporting records

`local/` preserves the original dated insertion drafts, plotting workspaces, source screenshots and superseded displays for this topic. It is ignored by Git. These files are historical support; the current DOCX determines final section placement. Full experiment runs, arrays and checkpoints retain their original paths under the ignored runtime directories. Source paths and hashes in captured provenance describe the original records, not bundled datasets.

## Saved evidence locations

These original records retain their existing paths; full runs and private evidence maps are intentionally local.

- `train/runs/20261025_194555_054312_infilteration_feature_audit_c9160d`
- `train/runs/20261025_201726_319757_lstm_distribution_features_seed4266_01c453/comparison.json`
