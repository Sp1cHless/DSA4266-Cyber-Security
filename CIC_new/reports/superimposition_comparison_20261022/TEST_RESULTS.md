# Frozen Scheme1 L2 versus Scheme2: real held-out test

Population: 263125 original real test sequences; 851 positive and 262274 benign.
No synthetic mixtures, fitting, checkpoint selection or test-based threshold choice. Saved train-fitted linear readouts remain frozen.
The readout is not the planned CNN. This compares two complete representation pipelines from one seed.

| Model | Validation ROC-AUC | Test ROC-AUC | Validation AP | Test AP |
|---|---:|---:|---:|---:|
| scheme1_l2 | 0.84441 | 0.82768 | 10.40% | 7.27% |
| scheme2 | 0.87922 | 0.84948 | 19.65% | 14.93% |

## Fixed thresholds

Thresholds were serialized before test shards were opened. Actual test FPR is shown; no test ROC point is selected to force equal FPR.

| Threshold source | Model | Test FPR | Precision | Recall | TP | FP | Mixed recall |
|---|---|---:|---:|---:|---:|---:|---:|
| balanced_prior_0.5 | scheme1_l2 | 11.61% | 1.83% | 66.63% | 567 | 30439 | 65.17% |
| balanced_prior_0.5 | scheme2 | 12.47% | 1.77% | 69.10% | 588 | 32707 | 68.28% |
| train_benign_99pct | scheme1_l2 | 1.08% | 8.64% | 31.37% | 267 | 2824 | 31.03% |
| train_benign_99pct | scheme2 | 0.93% | 12.03% | 39.37% | 335 | 2449 | 29.66% |
| train_benign_99.9pct | scheme1_l2 | 0.11% | 19.78% | 8.34% | 71 | 288 | 3.79% |
| train_benign_99.9pct | scheme2 | 0.08% | 32.27% | 11.87% | 101 | 212 | 2.07% |
| validation_fpr_1pct | scheme1_l2 | 1.03% | 8.93% | 31.02% | 264 | 2693 | 30.34% |
| validation_fpr_1pct | scheme2 | 0.99% | 11.73% | 40.54% | 345 | 2595 | 32.07% |
| validation_fpr_0.1pct | scheme1_l2 | 0.10% | 20.53% | 8.23% | 70 | 271 | 3.45% |
| validation_fpr_0.1pct | scheme2 | 0.10% | 33.50% | 15.63% | 133 | 264 | 4.83% |

## Primary attack types: train_benign_99pct

| Attack | n | Hosts | L2 detected | L2 recall | S2 detected | S2 recall |
|---|---:|---:|---:|---:|---:|---:|
| Bot | 49 | 7 | 49 | 100.00% | 49 | 100.00% |
| Brute_Force_-Web | 166 | 114 | 7 | 4.22% | 7 | 4.22% |
| Brute_Force_-XSS | 62 | 4 | 0 | 0.00% | 7 | 11.29% |
| DDOS_attack-HOIC | 65 | 10 | 55 | 84.62% | 65 | 100.00% |
| DDOS_attack-LOIC-UDP | 14 | 10 | 2 | 14.29% | 0 | 0.00% |
| DDoS_attacks-LOIC-HTTP | 110 | 10 | 0 | 0.00% | 60 | 54.55% |
| DoS_attacks-GoldenEye | 6 | 1 | 3 | 50.00% | 6 | 100.00% |
| DoS_attacks-SlowHTTPTest | 7 | 1 | 7 | 100.00% | 7 | 100.00% |
| FTP-BruteForce | 7 | 1 | 7 | 100.00% | 7 | 100.00% |
| Infilteration | 334 | 63 | 131 | 39.22% | 121 | 36.23% |
| SQL_Injection | 25 | 1 | 0 | 0.00% | 0 | 0.00% |
| SSH-Bruteforce | 6 | 1 | 6 | 100.00% | 6 | 100.00% |

## Primary attack types: train_benign_99.9pct

| Attack | n | Hosts | L2 detected | L2 recall | S2 detected | S2 recall |
|---|---:|---:|---:|---:|---:|---:|
| Bot | 49 | 7 | 42 | 85.71% | 42 | 85.71% |
| Brute_Force_-Web | 166 | 114 | 0 | 0.00% | 1 | 0.60% |
| Brute_Force_-XSS | 62 | 4 | 0 | 0.00% | 0 | 0.00% |
| DDOS_attack-HOIC | 65 | 10 | 0 | 0.00% | 32 | 49.23% |
| DDOS_attack-LOIC-UDP | 14 | 10 | 0 | 0.00% | 0 | 0.00% |
| DDoS_attacks-LOIC-HTTP | 110 | 10 | 0 | 0.00% | 0 | 0.00% |
| DoS_attacks-GoldenEye | 6 | 1 | 0 | 0.00% | 3 | 50.00% |
| DoS_attacks-SlowHTTPTest | 7 | 1 | 7 | 100.00% | 0 | 0.00% |
| FTP-BruteForce | 7 | 1 | 7 | 100.00% | 7 | 100.00% |
| Infilteration | 334 | 63 | 15 | 4.49% | 10 | 2.99% |
| SQL_Injection | 25 | 1 | 0 | 0.00% | 0 | 0.00% |
| SSH-Bruteforce | 6 | 1 | 0 | 0.00% | 6 | 100.00% |

## Scope

- Test means the original held-out dataset/test shards at natural prevalence, not the previous train-derived stress test.
- Existing validation-selected AE checkpoints and train-fitted probes are used unchanged. No parameter fitting or test threshold calibration occurred.
- Validation-FPR threshold rows apply a threshold selected from the old saved validation predictions; test FPR may differ.
- Train/validation/test use the original time-block policy; this is not an unseen-host or unseen-campaign guarantee.
- Primary attack grouping is not multilabel membership; small group counts and flow-fraction denominators must be inspected.
- Paired source-host bootstrap intervals condition on these fixed models/thresholds and do not estimate training-seed uncertainty.
- Model/probe/representation hashes are checked unchanged. Scores, IDs, fractions and all fixed-threshold group counts are saved separately.
- Future tuning informed by these scores is test-informed development and needs a fresh untouched holdout for another final claim.
