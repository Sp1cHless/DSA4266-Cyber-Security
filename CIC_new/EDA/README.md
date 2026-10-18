# CIC exploratory analyses

These scripts describe local NF-CICIDS2018-v3 without editing raw data or training models.
The final pipeline is [../build_buckets.py](../build_buckets.py); its contract is [../preprocessing_v1.md](../preprocessing_v1.md).

## Runtime and Git

Run from the repository root. EDA 01/02 use the standard library; EDA 03 also needs NumPy.
Excel uses the automatically resolved bundled Node/artifact-tool runtime; see [../README.md](../README.md#excel-runtime).
Override with `--node` and `--node-modules`.

Large caches/catalogs/CSV/JSON/previews remain under ignored `EDA/outputs/`.
Final archived workbooks are in **[../reports/EDA/](../reports/EDA/)**, which is not ignored.
The English guides here are also Git-visible.
Fresh runs need new/empty output directories; completion requires analysis and Excel success.
`--render-only <existing result directory>` rebuilds Excel without rescanning.

## EDA 01: observation perspectives

```powershell
python CIC_new/EDA/01_compare_perspectives.py
python CIC_new/EDA/test_01_compare_perspectives.py
```

Defaults: `datasets/cic_2018_traffic.db`, 10s buckets, top 50 examples, timestamped output.
Options include `--db`, `--output`, `--bucket-seconds`, `--top-n`, runtime paths and render-only.
Compare source, destination, endpoint and network. Endpoint counts each distinct endpoint once per flow;
self-flows count once and other flows may have two memberships. Network is a reference.
`perspective_counts.sqlite` caches active counts. Full entity CSVs remain available; Excel top examples are bounded.

Workbook: Overview, Entity classes, Bucket density, Daily, Attack types, Top entities, Method.
Definitions/results: [01_metrics_and_analysis.md](01_metrics_and_analysis.md).
Source IP was selected for its greater attack-source diversity.

## EDA 02: sequence lengths

```powershell
python CIC_new/EDA/02_compare_sequence_lengths.py
python CIC_new/EDA/test_02_compare_sequence_lengths.py
```

Reuse the latest completed EDA 01 cache after raw/cache fingerprint checks. Explicit example:

```powershell
python CIC_new/EDA/02_compare_sequence_lengths.py --perspectives-dir CIC_new/EDA/outputs/01_perspectives_20261018 --lengths-seconds 50 100 200 500 1000 --output CIC_new/EDA/outputs/my_sequence_eda
```

Lengths are positive multiples of 10s, at most one day. Stride equals length; each source/date aligns to UTC midnight.
Keep internal empty buckets and omit empty windows. Do not cross midnight.
Exclude/reconcile incomplete tails; 500/1000s each leave a 400s daily tail.
Network no-start intervals do not prove capture unavailability or no ongoing connections.

Workbook: Overview, Classes, Distributions, Occupancy, Attack types, Daily, Coverage, Method.
`windows_<length>s.csv.gz` contains every retained window, not a sample; bit 0 in its hex mask is the first position.
CSV/JSON distributions are exact. Final length is 100s; this study did not establish model-performance optimality.
All columns/examples/results: [02_metrics_and_analysis.md](02_metrics_and_analysis.md).

## EDA 03: feature audit

```powershell
python CIC_new/EDA/03_audit_features.py
python CIC_new/EDA/test_03_audit_features.py
```

Audit raw fields against historical `processed_binary_temporal/categorical_schema.json`.
Pass `--legacy-schema` if moved. The final builder does not require that ignored historical file.

Quality counts/extrema/moments/categories/TCP support use the full table.
Quantiles/correlations use Bernoulli sampling, target 100,000, seed 4266; actual count is recorded.
This descriptive audit fits no final vocabulary, scaler or selector.
Excel ports show top 50; CSV/JSON contain all values.

Workbook: Overview, Profiles, Categories, Legacy 88, Aggregation, Redundancy, Diagnostics, Method.
`aggregation_registry.json` contains candidate rules, not the final schema.
Options: `--db`, `--output`, `--sample-target`, `--seed`, `--batch-size`, `--legacy-schema`, runtime and render-only.
Definitions/results: [03_metrics_and_analysis.md](03_metrics_and_analysis.md).

## EDA 04: split feasibility and temporal isolation

```powershell
python CIC_new/EDA/04_audit_splits.py
python CIC_new/EDA/test_04_audit_splits.py
```

Reuse the latest completed EDA 02 full 100s catalog, then perform one read-only raw scan for actual flow ends
and exact malicious activity. NumPy is required. Compare the pre-v2 random-window baseline, chronological
windows, chronological/shuffled whole dates, global 15/30/60min blocks, and 30min blocks unioned across
300s-gap candidate attack episodes. Use seed 4266 without a label-based seed search.

Global schemes exclude boundary windows using actual flow ends and a default 100s guard on each side.
Episode separation is measured at 60/300/900s gaps; these inferred episodes are not verified independent campaigns.
This EDA does not change the builder or freeze a final split. Exact full-row duplicates are not audited.
The production builder now separately implements the approved coverage-constrained 15min policy in `../split_policy.py`;
its counts are in `../reports/split_policy_v2_preflight.json`. It is not the unconstrained `blocks_15m` EDA candidate.

Excel: Comparison, Composition, Attack coverage, Activity, Episodes, Daily, Isolation, Method.
Final Excel/JSON/English guide are Git-visible under `../reports/EDA/04_splits_<timestamp>/`.
Candidate NPZ manifests, detailed CSVs and previews stay under ignored `outputs/04_splits_<timestamp>/`.
`--render-only <existing output>` reuses saved results. Full definitions: [04_metrics_and_analysis.md](04_metrics_and_analysis.md).

## Interpretation

Flows can share a millisecond. One-flow buckets are active; no new start need not mean no active connection.
Whole-flow statistics are assigned by start, not redistributed as packet-level interval traffic.
Special addresses remain in counts and are not asserted to be physical hosts.
Removing literal IP features does not remove all behavioral fingerprints.

Natural hybrid means both labels occur in a window; mixed bucket means both have starts in one bucket.
Neither proves connection-interval overlap. More windows do not create more independent attack sources.
