# Historical report archive

The selected method and completed Scheme1/Scheme2 comparisons are recorded in
[superimposition_comparison_20261022/SUMMARY.md](superimposition_comparison_20261022/SUMMARY.md).
That subfolder shares SUMMARY.md, TEST_RESULTS.md and test_metrics.json. The full
evidence tree and machine-specific cleanup audits remain local and are ignored
by the root .gitignore. Scheme2's completed training and real-test results remain
available in the shared reports without uploading datasets or model weights.

These files describe earlier EDA and V2 raw-flow superposition. Their schemas,
counts and checks do not validate the current embedding pipeline. The earlier
builder is retained on `origin/codex/cic-timeblock-buckets`.

Curated reports are eligible for version control. Generated evidence and runtime
dependencies are excluded; do not add a blanket exception for every report file.

- `EDA/` contains the first three full-data exploratory workbooks archived from local ignored output folders,
  plus `04_splits_<timestamp>/` with the split comparison workbook, machine-readable results and English guide.
  English definitions are in [../EDA/](../EDA/README.md).
- `split_policy_v2_preflight.json` records the full-data split-only check for the adopted 15min coverage policy; no buckets were built in that check.
- `buckets_v2_<timestamp>/` records an earlier completed raw-flow bucket build. It contains its workbook,
  English metric explanation, machine-readable summary, split notes, feature schema and training-only scaler.

Large datasets, event caches, manifests, raw shards and rendering previews belong under ignored `CIC_new/outputs/` instead.
The builder's synthetic test fixture is not a full CIC report and is not archived here.
