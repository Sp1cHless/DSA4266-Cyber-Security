# Scheme2 preprocessing and export

Use [`prepare_scheme2_dataset.py`](../SCHEME2_DATASET.md) for the complete checked
workflow. This package implements raw preparation and frozen-vector export;
the input encoder lives in `representation_learning/`, and downstream models in
`train/`.

## Raw preparation

Run exactly one source option from the repository root with a fresh output path:

```powershell
python -m CIC_new.preprocessing.run_preprocessing --db datasets/cic_2018_traffic.db --out CIC_new/outputs/preprocessed
python -m CIC_new.preprocessing.run_preprocessing --csv datasets/NF-CICIDS2018-v3.csv --out CIC_new/outputs/preprocessed_from_csv
```

These are alternative SQLite/CSV commands. The SQLite source requires the full
CIC schema. This stage uses NumPy and pandas without importing PyTorch.

For sequential sequence-duration/bucket-duration comparisons, use
[`run_temporal_ablation.py`](../TEMPORAL_ABLATION.md). It defaults to these raw
preparation stages and supports a later, separate full Scheme2 dataset export.
Both single-run entry points accept `--sequence-seconds` and `--bucket-seconds`;
their defaults remain 100 and 10.

`sequence_index.py` constructs source-IP sequences; `bucket_statistics.py`
aggregates 82 numeric features and raw category histograms into ten buckets.
`split_policy.py` applies global time blocks, boundary exclusions and attack
coverage reservations. Vocabularies and numeric scaling are fitted afterward on
retained real training active buckets. Unknown categories use slot 0; actual
category code 0 has its own slot. Whole-flow statistics belong to the start bucket
and represent offline summaries rather than exact within-bucket transmission.

Output contains `splits/{train,validation,test}/part_*.npz`, `schema.json`,
`scaler.json`, `split_notes.json`, `manifest.json`, `manifest.sqlite`,
`raw_originals/` and `work.sqlite`. Original shards contain numeric values,
protocol/L7/service histograms, masks, labels, IDs and flow/attack-flow counts.
`dataset.py` reads these originals with disposable mmap caching and provides
training-only donor pools. `donor_plan.py` persists the selected zero-shift plan.

## Frozen-vector export

After [fitting/freeze/encoding](../representation_learning/README.md), the direct
Scheme2 builder calls `export_augmented_dataset.py` and validates all outputs.
Prefer that builder to obtain the complete checksummed handoff contract.
The lower-level exporter is available for individual-stage work:

```powershell
python -m CIC_new.preprocessing.export_augmented_dataset --data <prepared> --encoded <encoded> --out <fresh-export> --shift-mode none --donor-plan <donor_plan.npz>
```

The selected policy uses zero displacement and a fixed donor plan. The low-level
exporter also retains optional circular shifts and standalone sampling for
explicit experiments; those are different configurations. Its historical default
shift is circular, so pass `--shift-mode none` explicitly. Standalone export alone
does not run the direct builder's final validator or register a completed workflow.

The first donor is benign; the second is benign or malicious. Both are real train
originals from different source IPs. Background sampling balances occupancy band
(1, 2-4, 5-10 active buckets), source, then sequence; malicious sampling balances
attack family, source, then sequence. Donor reuse is allowed. Only training receives
P examples per synthetic class. Validation/test retain real natural populations.

`augmentation.py` adds frozen bucket vectors, L2-normalizes each sum, propagates
masks/labels/counts and appends log1p total-flow count scaled with the real training
scaler. Numerical statistics are not added. Empty positions stay zero; cancellation
of active vectors does not mark them empty.

| Final shard key | Meaning |
|---|---|
| `X` | float32 `[N,10,65]` with default encoder dimensions |
| `mask` | Active bucket positions |
| `y_sequence`, `y_bucket` | Targets separate from inputs |
| `sample_id`, `sample_type` | Exported identity and real/synthetic group |
| `left_id`, `right_id`, `right_shift_buckets` | Donor identities and displacement; absent second donor is -1 |
| `flow_counts` | Combined bucket flow counts |
| `left_ip`, `right_ip`, `primary_attack` | Audit metadata, never model inputs |

`flow_metadata.py` exports checksummed real validation/test flow-count sidecars.
The complete final dataset also includes manifests, schema, scaler, donor plan
and frozen encoder. Downstream loading, caching and weighting use `train/common/`.

Allow tens of GB of persistent free disk for full preparation and intermediate
caches. Use fresh outputs and avoid concurrent builders against the same cache.
Run the small checks with `python -m unittest discover -s CIC_new/tests -t . -v`.
