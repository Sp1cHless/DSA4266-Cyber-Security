# Experiment Status

Snapshot date: 2026-10-10

## Established UNSW-NB15 control

- The official preprocessing path uses 10-second source-host buckets, 10 buckets per sequence, and 88 behavioural features.
- Exact IP addresses, exact port numbers, identifiers, and raw categorical codes are not model inputs.
- `run_cnn_ae_temporal_control.py` trains one CUDA CNN autoencoder on benign Round 1 training sequences.
- The completed seed-0 run is local under `experiments/cnn_ae_temporal_control_seed0/`.

Measured results from that run:

| Evaluation | ROC-AUC | PR-AUC | Detection rate at frozen 1% validation-FPR threshold |
|---|---:|---:|---:|
| UNSW Round 1 | 0.9806 | 0.7814 | 2.46% |
| UNSW Round 2 | 0.9803 | 0.7801 | 1.53% |

The high ranking AUC therefore does not imply strong performance at the frozen operating threshold.

The UNSW coverage audit found that 98.85% of raw flows were represented by valid contiguous sequences. Twenty-one hosts produced no valid sequence, but together accounted for only 0.26% of raw flows.

## CICIDS2018 exploration

CICIDS2018 has not replaced UNSW-NB15 as the project dataset. The current CIC work is an experimental cross-dataset extension:

- `datasets/cic_2018_traffic.db` contains 20,115,529 imported flows and the generated CIC bucket table.
- `processed_binary_temporal_unsw_cic/` contains a benign-only host-disjoint extension with 120,000 CIC training sequences, 25,000 CIC validation sequences, and 25,000 CIC holdout sequences.
- The CIC holdout is benign-only, so it is not yet a binary attack-detection test set and cannot support ROC-AUC or PR-AUC evaluation.
- A full CIC evaluation still requires a separately designed attack test split without fitting thresholds or selecting models on test labels.

CIC has much weaker temporal coverage under the current consecutive-window definition: 23.81% of benign raw flows occur in valid 10-bucket segments, and the selected windows cover 20.70%. This is a dataset characteristic that must be reported rather than hidden by zero filling.

## Stride status

The currently generated UNSW and UNSW+CIC manifests both record `stride = 1`. Changing the constants in the sequence builders does not change existing NPZ files; regenerate the corresponding sequence artifacts before claiming a stride-5 experiment.

## Local artifact checksums

These large files are intentionally excluded from Git:

| Artifact | Size | SHA256 |
|---|---:|---|
| `datasets/network_traffic.db` | 0.465 GiB | `199679BAB1EC23CBCB4A8E40B04526CA9300C7B3C21F1C48830CF5C8D0AA3014` |
| `datasets/cic_2018_traffic.db` | 6.123 GiB | `E3CD086482DA051639DF9FFC79DA78297A0D0BE7D4AE2D82CCB3EBE69EC024B0` |

For teammates who only need to train or evaluate models, share the processed NPZ/metadata package instead of the multi-gigabyte SQLite database. Keep the databases in external shared storage as optional reproducibility artifacts.
