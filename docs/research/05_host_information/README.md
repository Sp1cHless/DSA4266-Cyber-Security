# Host Information and the Link to Targeted Studies

Report sections: 5.3. Organized on 1 November 2026 from completed project records.

## Finding

Independent probes recover substantial host information after direct identifiers are excluded. Moderate adversarial training reduces aggregate attack-host probe balanced accuracy from 46.31% to 36.56%, while validation AP changes from 61.53% to 59.89%. The study establishes a suppression/detection tradeoff and motivates the subsequent Web and Infilteration analyses.

## Evidence scope

One detector seed, matched fixed inputs, class-conditional host supervision, and independent probes on purged partitions of reused detector-training records. Host coverage and suppression vary by subtype. This is not a host-free validation result.

## Report figures

- S1: [PNG](<figures/fig_host_information_tradeoff.png>) and [SVG](<figures/fig_host_information_tradeoff.svg>).

## Compact results

- [figure_provenance.json](<results/figure_provenance.json>)

## Code and protocol entry points

- [train/lstm_host_adversarial/README.md](<../../../train/lstm_host_adversarial/README.md>)
- [train/run_lstm_host_adversarial.py](<../../../train/run_lstm_host_adversarial.py>)
- [train/lstm_host_adversarial/review.py](<../../../train/lstm_host_adversarial/review.py>)

## Local supporting records

`local/` preserves the original dated insertion drafts, plotting workspaces, source screenshots and superseded displays for this topic. It is ignored by Git. These files are historical support; the current DOCX determines final section placement. Full experiment runs, arrays and checkpoints retain their original paths under the ignored runtime directories. Source paths and hashes in captured provenance describe the original records, not bundled datasets.

## Saved evidence locations

These original records retain their existing paths; full runs and private evidence maps are intentionally local.

- `train/runs/20261030_000611_945296_lstm_host_adversarial_seed4266_b1a7f9/FINDINGS.md`
- `train/runs/20261030_000611_945296_lstm_host_adversarial_seed4266_b1a7f9/attack_supervision_coverage.csv`
- `train/runs/20261030_000611_945296_lstm_host_adversarial_seed4266_b1a7f9/integrity_audit.json`
- `train/runs/20261030_000611_945296_lstm_host_adversarial_seed4266_b1a7f9/overall_comparison.csv`
- `train/runs/20261030_000611_945296_lstm_host_adversarial_seed4266_b1a7f9/probe_comparison.csv`
