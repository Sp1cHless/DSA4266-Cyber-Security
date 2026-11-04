# Representation and Scheme Selection

## Finding

Bucket L2 improves Scheme 1 validation AP from 3.44% to 10.40%; Scheme 2 reaches 19.65% in the temporal-autoencoder/linear-readout comparison. Scheme 2 uses 64 learned bucket coordinates plus a separate intensity channel and becomes the shared supervised input.

## Evidence scope

Complete representation pipelines at seed 4266, with paired original IDs, splits and donor plans. The autoencoder/readout comparison has a different objective from later supervised detector training. Historical test results use validation-selected thresholds.

## Compact results

- [reconstruction_error_concentration.csv](results/reconstruction_error_concentration.csv)
- [representation_comparison.csv](results/representation_comparison.csv)

## Code and protocol entry points

- [CIC_new/SCHEME2_DATASET.md](<../../../CIC_new/SCHEME2_DATASET.md>)
- [CIC_new/representation_learning/README.md](<../../../CIC_new/representation_learning/README.md>)
- [CIC_new/prepare_scheme2_dataset.py](<../../../CIC_new/prepare_scheme2_dataset.py>)

## Local evidence

Completed runs and historical working material are consolidated in the ignored
`archive/` directory. Restore the matching inputs before rerunning a diagnostic
that requires saved arrays or checkpoints. See the [research index](../README.md) for the result catalog and evidence boundaries.
