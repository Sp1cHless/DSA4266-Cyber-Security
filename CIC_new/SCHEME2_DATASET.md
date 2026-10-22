# Scheme2: database to whole-bucket embedding inputs

Scheme2 is the selected method. Existing port/category rules and model dimensions
are unchanged. Run from the repository root only when a new build is needed:

```powershell
.\.venv-cnn\Scripts\python.exe -u CIC_new\prepare_scheme2_dataset.py
```

1. Construct 100-second source-IP sequences and ten 10-second raw buckets.
2. Apply the approximate 6:2:2 global time-block split, coverage reservations
   and temporal exclusions. Fit vocabularies/scalers on real training buckets.
3. Fit the bucket AE on real training active buckets. Validation selects the
   checkpoint. Defaults: protocol/L7/service dimensions 16 each, bucket dimension
   64, Adam lr 0.001, 5000 updates, batch 512; validation every 200 updates on at
   most 4096 fixed validation buckets. Test does not fit/select parameters.
4. Freeze and encode real train/validation/test.
5. Retain all originals. Generate P benign+attack and P benign+benign training
   examples with different donor source IPs. Reuse the established attack/host
   balancing, background activity bands and donor seed offset (+1). Zero shift;
   add frozen vectors and L2-normalize; append scaled log1p flow count.
6. Check every exported vector/count/mask/label, donor ownership, plan, original
   coverage and shard checksum. Save compact checksummed real validation/test
   flow-count sidecars for evaluation without another representation's data.

No Scheme1 output is required. The donor sampler is shared source code retained
in scheme1/; it does not train or read a Scheme1 representation. The original
completed Scheme2 dataset still contains historical paired-reference paths in
its frozen manifest; these record provenance and are no longer runtime inputs.

Each build creates a unique
outputs/scheme2_runs/<timestamp>_scheme2_bucket64_shift0_seed4266_<suffix>/,
containing workflow.json, preprocessed/, bucket_embeddings/, encoded_sequences/,
donor_plan.npz and dataset/. Final input shape is [N,10,65]: 64 embedding
coordinates and one intensity channel. Downstream --dataset points to dataset/.

| Split | Real benign | Real malicious | Synthetic benign | Synthetic malicious |
|---|---:|---:|---:|---:|
| Train | 757559 | 8093 | 8093 | 8093 |
| Validation | 250229 | 917 | 0 | 0 |
| Test | 262274 | 851 | 0 | 0 |

These counts describe the completed source/seed, not every possible input.
Completion requires workflow status=complete and dataset validation passed=true.
Construction checks do not establish detection accuracy.

```powershell
python CIC_new\prepare_scheme2_dataset.py --out CIC_new\outputs\scheme2_new_run
python CIC_new\prepare_scheme2_dataset.py --out CIC_new\outputs\scheme2_new_run --resume
python CIC_new\prepare_scheme2_dataset.py --prepared <completed-v3-directory> --bucket-checkpoint <bucket-best.pt>
```

Resume reuses completed stages with identical recorded settings, not partial
optimizer state. External checkpoints must match the exact preparation and
supplied dimensions. Defaults resolve the SQLite source at
datasets/cic_2018_traffic.db. Optional --reference-dataset <rebuilt-scheme1-dataset>
restores historical paired checking; it is unnecessary for normal use.

Follow with [Scheme2 training/evaluation](SCHEME2_EXPERIMENT.md).
