# Scheme2 construction protocol

Run `python -u CIC_new/prepare_scheme2_dataset.py` from the repository root after
installing `CIC_new/representation_learning/requirements.txt`. The default source
is `datasets/cic_2018_traffic.db`, table `network_flows`.

## Construction

1. Construct 100-second source-IP sequences and ten 10-second raw buckets.
2. Apply the approximate 6:2:2 global time-block split, coverage reservations
   and temporal exclusions. Fit vocabularies/scalers on real training buckets.
3. Fit the bucket AE on real training active buckets. Validation selects the
   checkpoint. Defaults: protocol/L7/service dimensions 16 each, bucket dimension
   64, Adam learning rate 0.001, 5000 updates, batch 512; validation every 200
   updates on at most 4096 fixed validation buckets. Test does not fit/select
   parameters. This is an input encoder, not the downstream detector.
4. Freeze the encoder and encode real train/validation/test originals.
5. Retain all originals. Add P benign+attack and P benign+benign training
   examples, where P is the real malicious training count. Donors must have
   different source IPs. Preserve attack/host balancing, background activity
   bands and donor seed offset (+1). Add corresponding frozen vectors with
   zero displacement, apply L2 normalization, and append scaled log1p flow count.
6. Check every exported vector/count/mask/label, donor ownership, plan, original
   coverage and shard checksum. Save checksummed real validation/test flow-count
   sidecars for later evaluation.

The donor policy lives in `preprocessing/donor_plan.py`. No historical method is
required. Donors may be reused; P synthetic examples are not P new independent
attacks, and sampling does not guarantee every original attack is selected once.
Synthetic mixtures exist in learned vector space, not as reconstructed raw flows.

The final shape is `[N, 10, 65]`: 64 embedding coordinates and one intensity
channel. Empty buckets stay zero and have mask 0. Labels, attack names and source
identities remain separate audit metadata and never enter model inputs.

## Completed dataset population

| Split | Real benign | Real malicious | Synthetic benign | Synthetic malicious |
|---|---:|---:|---:|---:|
| Train | 757559 | 8093 | 8093 | 8093 |
| Validation | 250229 | 917 | 0 | 0 |
| Test | 262274 | 851 | 0 | 0 |

These counts describe the existing completed source/seed, not every input.
Completion requires `workflow.json` status `complete` and final
`dataset/validation.json` with `passed=true`. Construction checks do not establish
detection accuracy. Existing manifests may contain historical absolute reference
paths; preserve them as provenance. Downstream training does not dereference them.

## Outputs and stage reuse

Each default build reserves a unique directory under `CIC_new/outputs/scheme2_runs/`.
It contains `workflow.json`, `preprocessed/`, `bucket_embeddings/`,
`encoded_sequences/`, `donor_plan.npz` and final `dataset/`. Preserve completed
data and checkpoints. Supply a fresh `--out` for intentional new builds:

```powershell
python CIC_new/prepare_scheme2_dataset.py --out CIC_new/outputs/scheme2_new_run
python CIC_new/prepare_scheme2_dataset.py --out CIC_new/outputs/scheme2_new_run --resume
python CIC_new/prepare_scheme2_dataset.py --prepared <completed-v3-directory> --bucket-checkpoint <bucket-best.pt>
```

The second command reuses completed stages from the first with identical settings;
it does not resume a partial optimizer. The third builds a fresh export from
matching existing preparation and encoder artifacts. External checkpoints must
match the exact preparation and requested dimensions. Historical workflows made
with retired comparison flags should be consumed as completed data; use a fresh
build for the current command. The old `--reference-dataset` flag is retired.

Copy the complete final `dataset/` directory to `datasets/dataset/`, or provide
its path through the downstream `--dataset` option. Follow
[`train/README.md`](../train/README.md) for model training and evaluation.

## Training holdout for unseen attack subtypes

Pass `--holdout-train-attacks <labels>` to remove declared primary attack labels
from the training fold only, then rebuild. The option is recorded in the workflow
settings and the preparation manifest, so `--resume` cannot change it, and it is
rejected together with `--prepared`. Fold assignment, coverage reservations, the
seeded RNG and the temporal purge are identical to a run without a holdout, so
validation and test populations do not move; only training membership and every
transform fitted on the training fold (vocabulary, scaler, bucket encoder, donor
plan) exclude the declared labels. `preprocessed/split_notes.json` records removed
sequences and residual non-primary flows. Use `--csv` to build from a full-schema
CSV source. See [HOLDOUT_PROTOCOL.md](HOLDOUT_PROTOCOL.md) for the frozen label set,
checks and limits.

## Evaluation scope

The split uses global 15-minute blocks, approximately 60/20/20 allocation,
100-second guards and actual-flow-end exclusions. Train coverage is protected;
held-out attack coverage is reserved where feasible. Hosts/campaigns can recur
across splits. Historical work has already inspected this test split; it is not
a newly untouched holdout. Training-derived diagnostics and synthetic fixtures
must be reported separately from real held-out validation/test results.
