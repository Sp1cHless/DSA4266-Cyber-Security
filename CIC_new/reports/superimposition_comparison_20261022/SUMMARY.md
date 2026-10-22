# Superimposition comparison and method selection

Decision on 2026-10-22: **continue with Scheme2 only**. Its representation, port
rules, category dimensions and composition formula remain unchanged. Scheme1 and
Scheme1 with L2 are retired; their small implementation files remain reproducible.
This archive preserves the completed comparison evidence for deletion of their
large local generated datasets, model checkpoints, caches and draw logs.

The shared report contains this summary, [the test tables](TEST_RESULTS.md) and
[machine-readable test metrics](test_metrics.json). The complete evidence tree
and machine-specific cleanup audits remain archived locally and are ignored by
Git. Scheme1 output cleanup completed in the local terminal on 2026-10-22;
the local cleanup_result.json records the operation and preservation checks.

## Methods and controls

- Scheme1: exact numerical-statistic composition and field-level categorical
  representations, 137 features per bucket.
- Scheme1 with L2: the same final inputs, normalized across all 137 coordinates
  per active bucket, including the flow-count feature.
- Scheme2: frozen whole-bucket embeddings; donor vectors are added and L2
  normalized. A separate scaled log-flow-count channel gives 65 input features.
  Protocol, L7 and service embeddings each use 16 coordinates; bucket width is 64.
- The original identities, time-block split, donor plan and zero displacement
  were paired across methods. Split target was 6:2:2. Train has 781838 examples:
  757559 real benign, 8093 real malicious, 8093 synthetic benign and 8093 synthetic
  malicious. Validation/test contain only original real sequences.
- Temporal AE controls: seed 4266, batch 128, 500 updates/epoch, maximum 500
  epochs, width/latent 32, learning rate 0.001, patience 20, relative minimum
  improvement 0.001. Four training groups were sampled equally with replacement.
- Detection results below use fixed linear readouts fitted on real training
  latent vectors. They are not CNN or AE/CNN hybrid results.

## Final real held-out test

Population: 263125 real test sequences, including 851 malicious and 262274 benign.
Thresholds and model/readout weights were frozen before reading test features.

| Metric | Scheme1 with L2 | Scheme2 |
|---|---:|---:|
| Validation ROC-AUC | 0.84441 | 0.87922 |
| Test ROC-AUC | 0.82768 | 0.84948 |
| Validation average precision | 10.40% | 19.65% |
| Test average precision | 7.27% | 14.93% |
| Test recall, train-benign 99th-percentile threshold | 31.37% | 39.37% |
| Actual test FPR at that threshold | 1.08% | 0.93% |
| Test recall, threshold fixed at validation FPR 1% | 31.02% | 40.54% |
| Actual test FPR at that threshold | 1.03% | 0.99% |

Scheme2 performed better on these overall validation and real-test metrics.
Advantages are not uniform across attack types or background fractions. For
example, mixed-attack recall at the train-benign 99th-percentile threshold was
31.03% for Scheme1 with L2 and 29.66% for Scheme2; at the validation-FPR-1%
threshold it was 30.34% and 32.07%, respectively. Inspect denominators and fixed
threshold definitions before comparing subgroup results.

Full test tables and limitations:
[TEST_RESULTS.md](TEST_RESULTS.md). [test_metrics.json](test_metrics.json)
preserves the complete machine-readable test report, including metrics,
per-group counts, provenance and fixed thresholds. Paths inside it are original
experiment provenance. The full local evidence tree additionally contains the
split tables and evaluation plan; original paired scores remain under
outputs/test_evaluations/. Those large/local artifacts are not included in Git.

## Earlier reconstruction and stress diagnostics

Per-method AE MSE measures reconstruction in different target spaces; smaller
raw MSE does not establish a better attack detector. L2 improved Scheme1's
behavior but did not establish superiority over Scheme2 on the real test.

Controlled dilution diagnostics reused training attack flows and benign donors.
Their low-flow examples were dominated by particular attacks/hosts and were
not the held-out test population. Some favored Scheme1 with L2. They describe a
specific synthetic response, not evidence that Scheme2 lost the real-test
comparison. Earlier notes recommending Scheme1 based on those diagnostics are
superseded by this decision and the completed real-test report; they remain in
the evidence tree as historical records.

Neither experiment proves E(A+B)=E(A)+E(B), causal augmentation benefit,
cross-seed robustness, unseen-host generalization or improved detection of every
background attack. The choice is an empirical full-pipeline decision for further
work. Future tuning informed by the observed test results needs a new untouched
holdout for another final performance claim.

## Archive and reproduction

- The ignored local evidence/ tree preserves JSON/Markdown/CSV results, configurations, histories,
  validation records and donor plans. Historical absolute paths inside copied
  reports are original provenance; some will point to intentionally deleted outputs
  once local cleanup has completed.
- The ignored cleanup_plan.json lists archived hashes and the exact deletion scope.
- The ignored preserved_files.json records Scheme2/source files before deletion.
- The ignored cleanup_result.json records removal and post-cleanup verification.
- After cleanup, rebuild Scheme1 with prepare_dataset.py; train_embeddings.py and
  validate_dataset.py are separate stage entry points. Keep scheme1/ and shared
  preprocessing/runtime modules. Scheme1 learned parameters are removed by cleanup.
- Rebuild detector runs with train_autoencoder.py and an explicit Scheme1
  --dataset. Reproduce the L2 variant with --input-transform bucket_l2, using
  the recorded controls. Historical comparison tools require rebuilt paths.
- Scheme2 builds, training and validation now run independently of Scheme1
  outputs. The sampling helper remains shared source code; no Scheme1 model or
  generated data is needed. Newly generated donor plans use the original seed
  offset (+1), attack/host balancing and background activity bands.

No model was retrained and no new performance result was created during cleanup.
