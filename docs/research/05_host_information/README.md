# Host Information and the Link to Targeted Studies

## Finding

Independent probes recover substantial host information after direct identifiers are excluded. Moderate adversarial training reduces aggregate attack-host probe balanced accuracy from 46.31% to 36.56%, while validation AP changes from 61.53% to 59.89%. The study establishes a suppression/detection tradeoff and motivates the subsequent Web and Infilteration analyses.

## Evidence scope

One detector seed, matched fixed inputs, class-conditional host supervision, and independent probes on purged partitions of reused detector-training records. Host coverage and suppression vary by subtype. This is not a host-free validation result.

## Compact results

- [attack_comparison.csv](results/attack_comparison.csv)
- [overall_comparison.csv](results/overall_comparison.csv)
- [probe_comparison.csv](results/probe_comparison.csv)

## Code and protocol entry points

- [train/lstm_host_adversarial/README.md](<../../../train/lstm_host_adversarial/README.md>)
- [train/run_lstm_host_adversarial.py](<../../../train/run_lstm_host_adversarial.py>)
- [train/lstm_host_adversarial/review.py](<../../../train/lstm_host_adversarial/review.py>)

## Local evidence

Completed runs and historical working material are consolidated in the ignored
`archive/` directory. Restore the matching inputs before rerunning a diagnostic
that requires saved arrays or checkpoints. See the [research index](../README.md) for the result catalog and evidence boundaries.
