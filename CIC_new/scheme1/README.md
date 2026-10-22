# Scheme1: CIC database to zero-shift training inputs

Historical reproduction only. Scheme2 is the selected method. Scheme1 generated
data, caches, checkpoints and large logs are scheduled for deletion after preserving results in
[the comparison archive](../reports/superimposition_comparison_20261022/SUMMARY.md).
The source package and shared dependencies remain; rebuild data before training.
Use `python CIC_new/train_autoencoder.py --dataset CIC_new/outputs/scheme1/dataset`
explicitly after preparation, since the generic trainer now defaults to Scheme2.

## Run

```powershell
.\.venv-cnn\Scripts\python.exe CIC_new\prepare_dataset.py
python CIC_new\prepare_dataset.py --db datasets\cic_2018_traffic.db --out CIC_new\outputs\scheme1_run2
```

Default absolute paths resolve internally: the full network_flows table in
datasets/cic_2018_traffic.db and output CIC_new/outputs/scheme1/.
Stages: sequences, raw buckets, split, train-fitted transforms, independent field
embeddings, zero-shift composition, full export, and data validation. Final AE/CNN
training is separate. Auto device selects CUDA if available, otherwise CPU;
explicit cpu/cuda are supported. Use an interpreter with NumPy and PyTorch.

## Category learning

```powershell
python CIC_new\prepare_dataset.py --prepare-only
python CIC_new\train_embeddings.py
python CIC_new\prepare_dataset.py --resume
```

These stages share the direct workflow. The learner reads --data .../prepared/
and writes a fresh embedding directory; --embeddings supplies custom tables.
Tables must match the same preparation. L7/source/destination-port dimensions
default to 8 each and are configurable independently. The learner masks one field
and predicts its original category distribution using numeric/fixed/other-field
features. Input/output tables are tied; the auxiliary MLP is not a bucket encoder.
Targets are fixed distributions, not moving embeddings. Labels/attack/IP are excluded.

All observed real-training ports are retained without top-K truncation. The pool
starts with up to 50000 uniformly sampled real active buckets and adds category
coverage examples; it can exceed --train-pool. Training alternates fields over
shuffled pools and completes at least one pass. Steps is a minimum budget and may
increase for coverage. Only coverage-complete checkpoints are eligible. Defaults:
3000 minimum updates, batch 128. Both real benign/malicious buckets are eligible.
At most 2048 validation active buckets select the checkpoint. Test does not fit or
select parameters. Final scalers use all real train active buckets, not only the
pool. Frozen weights are saved in embeddings.npz; detection quality needs separate evaluation.

## Design

- Source-IP windows of 100 seconds, ten 10-second buckets; assign entire flows by start.
- Global 15-minute blocks, coverage reservations, 100-second guard and actual-end
  purging. Target 6:2:2 before augmentation; exact retained proportions vary.
  Hosts/campaigns can recur across splits; this is not an unseen-host/campaign test.
- Retain 82 statistics. Sum totals, weight means by applicable counts, combine
  extrema, recompute diversity from merged frequencies and gaps from merged offsets.
- Retain raw frequencies, packet bins and offsets for exact composition. Do not
  add standardized numeric vectors or fit a whole-bucket encoder.
- Protocol/service/ICMP/DNS use fixed training-vocabulary proportions with UNK.
  ICMP/DNS applicability and TCP/UDP port/service eligibility remain explicit.
  Historical numeric port diversity still uses all-flow histograms, stored separately.
- L7/source/destination ports use count-weighted mean learned category embeddings.
  Source/destination tables differ; actual port 0 differs from UNK. Missing fields
  have raw zero vectors; masks identify empty buckets.
- Apply log1p after composition, excluding ratios/entropy. Scale all final channels
  with real-training active-bucket mean/std; restore padding to zero. The original
  export has no whole-bucket L2; see [the L2 ablation](../SCHEME1_L2_EXPERIMENT.md).
- IPs support grouping/pairing/destination diversity/audits only; labels and donor
  metadata do not enter X.

Retain all N real train examples, add P B+A and P B+B, with P the real positive
count. A can already contain background flows; B is benign-only. Sample attack
type/host/sequence uniformly in that order; sample backgrounds by active-bucket
band (1, 2-4, 5-10), host, sequence. Hosts differ; only real train donors are used.
Reuse is allowed; every attack need not be selected. Do not filter compositions
by density/overlap, remove real benign examples, or recursively mix synthetic ones.
The default zero-shift donor_plan.npz stores IDs, target attack, background band
and shift, with seed 4267 independent of embeddings and batch size. Future relative
shift must move a donor before recomposition, not shift the final mixed sequence.

## Outputs and use

```text
scheme1/
  workflow.json
  prepared/
    manifest.sqlite, work.sqlite
    raw_originals/part_*.npz
    schema.json, numeric_scaler.json, split_notes.json, source_controls.json
    embedding_pools/train.npz, validation.npz
    manifest.json
  embeddings/
    embeddings.npz, manifest.json, history.json, best.pt
  dataset/
    train/part_*.npz
    validation/part_*.npz
    test/part_*.npz
    donor_plan.npz, donor_cache.sqlite
    embeddings.npz, embedding_manifest.json
    schema.json, scaler.json, manifest.json, validation.json
```

Raw originals include excluded sequences for audits, not model fitting. X is
float32 [N,10,D]; D is 82 plus fixed vocabulary widths including UNK plus learned
dimensions, as recorded in schema.json. Labels/masks/counts/provenance are separate.

```python
from CIC_new.preprocessing.export_augmented_dataset import iter_model_shards
for batch in iter_model_shards("CIC_new/outputs/scheme1/dataset", "train"):
    X, mask, y = batch["X"], batch["mask"], batch["y_sequence"]
```

ModelInputs also reads this format; close it or use a context manager. Exported
prevalence differs from optional balanced four-group sampling. AE trains on both classes.

## Validation and recovery

```powershell
python CIC_new\validate_dataset.py
python CIC_new\validate_dataset.py --data CIC_new\outputs\scheme1_run2\dataset --deep-samples 16
```

Exit 0 plus passed=true indicates success. Builds validate unless --skip-validation.
Checks cover shapes, finiteness, padding, labels/counts/masks, IDs, ownership,
donor split/host/type, zero shift, N+2P/P budgets, flow conservation, category
coverage and hashes. Vocabularies/scalers are independently refitted on all real
train active buckets; temporal isolation and pool ownership are checked. Up to
8 examples per split/type are rebuilt from cached source flows. Structural
checks are global; independent feature arithmetic is sampled. --deep-samples 0
skips that sample. Data validation is not detector evaluation.

Sources are read-only; output is fresh. Resume reuses completed stages, not partial
stages/optimizers. Keep interrupted output and use new directories. Export interrupted
only during validation can be checked directly. Avoid concurrent writes to one output.
Caches/intermediates remain on disk. Fixtures do not measure full-run runtime,
memory, disk or quality. Scheme2 remains available with the same optional donor plan;
see [SCHEME2_DATASET.md](../SCHEME2_DATASET.md).

```powershell
.\.venv-cnn\Scripts\python.exe -m unittest discover -s CIC_new/tests -t . -v
```

Tests use artificial databases for feature arithmetic, unknown categories/port 0,
direct/staged entry points, shared plans, stage reuse and corruption rejection.
They do not rebuild the full local CIC source.
