# Historical report archive

These files describe earlier EDA and V2 raw-flow superposition. Their schemas,
counts and checks do not validate the current embedding pipeline. The earlier
builder is retained on `origin/codex/cic-timeblock-buckets`.

This directory is deliberately not ignored.

- `EDA/` contains the first three full-data exploratory workbooks archived from local ignored output folders,
  plus `04_splits_<timestamp>/` with the split comparison workbook, machine-readable results and English guide.
  English definitions are in [../EDA/](../EDA/README.md).
- `split_policy_v2_preflight.json` records the full-data split-only check for the adopted 15min coverage policy; no buckets were built in that check.
- `buckets_v2_<timestamp>/` records an earlier completed raw-flow bucket build. It contains its workbook,
  English metric explanation, machine-readable summary, split notes, feature schema and training-only scaler.

Large datasets, event caches, manifests, raw shards and rendering previews belong under ignored `CIC_new/outputs/` instead.
The builder's synthetic test fixture is not a full CIC report and is not archived here.
