# V3: buckets first, then split, then vector-space augmentation

This is the revised implementation following the professor's reply. It is a
separate version: existing V1/V2 datasets, raw-flow augmentation code, reports and
the previous 24-feature Runpod experiment remain intact. **Do not run the old
`CIC_new/build_buckets.py` command to build V3.** It still implements the historical
raw-flow pipeline. V3 has its own preparation and training entry points below.

## What changed

```text
Group by source IP, sort, construct nonoverlapping 100s original sequences
  -> aggregate into ten 10s buckets, retaining categorical histograms
  -> split ORIGINALS (global 15min blocks + boundary/actual-end purging)
  -> fit vocabularies and numerical scaling on retained real TRAIN only
  -> learn categorical/bucket embeddings using real TRAIN only
  -> freeze the bucket encoder and encode original sequences
  -> TRAINING ONLY: pick two donors, circularly rotate donor B's bucket axis
  -> add bucket embedding vectors, then L2-normalise per bucket
  -> train temporal AE; evaluate on untouched real validation sequences
```

The professor specified the ordering and vector-space addition. The two-stage
embedding architecture, per-bucket normalisation, volume channel and initial
hyperparameters below are implementation choices, not further professor instructions.
We interpret circular shifting as rotating one donor before mixing. This changes
the relative alignment of the two donors; it is not a rotation of entire capture
days or a timestamp delay with cropping. Confirm this interpretation with the
professor if their brief “that is correct” referred to another option in the email.

## Why the embeddings are learned and then frozen

Categorical protocol, L7 application and destination-service categories have
trainable embedding lookups (default 16 dimensions per field). Each bucket's
category proportions weight those vectors. The field vectors are concatenated
with a projection of its 82 original numerical/statistical features, then encoded
to a 64-dimensional **unit** bucket embedding.

A bucket AE first learns this representation from original training buckets.
It reconstructs numerical features (MSE) AND the categorical distributions
(soft-target cross-entropy). Category identity is therefore a training target,
not an arbitrary numeric identifier. Normalisation is included while training the
bucket AE, so we do not discard norm-encoded information only at export.

We then freeze the bucket encoder. The temporal AE's inputs and reconstruction
targets cannot change through gradients in the embedding network. This prevents
a joint model from reducing reconstruction loss simply by collapsing its
trainable input representation. Different random seeds and downstream classifier
performance must still be evaluated; this design is not proof of useful embeddings.

For donor vectors at bucket t:

```text
b_shifted = circular_roll(b, k, axis=time)
mixed[t]  = L2_normalise(a[t] + b_shifted[t])
```

Each component is zeroed where its mask is zero; normalisation uses an epsilon.
If two nonempty vectors cancel, their mask remains active even if the resulting
vector is zero. Empty positions remain zero. The output mask is the union of the
two masks. Bucket labels, masks and flow counts follow the same rotation, with
bucket labels combined by maximum. Sequence labels are their maximum. No labels,
IP literals, source times or donor IDs enter the encoder.

L2 normalisation removes magnitude. We therefore append ONE separate feature:
the log1p combined flow count, standardised using the original-training count
scaler. Counts can be added meaningfully. Means, entropies and the full original
feature vector are **not** added. Default temporal input width is 64+1=65, with
the activity mask supplied separately. Synthetic vectors represent a mixture in
learned space, not reconstructed physical NetFlow records or exact mixed byte totals.

## Preparation and leakage controls

`prepare.py` first calculates all label-independent bucket statistics and raw
category histograms without a fitted vocabulary. Only then does it assign and
purge splits. Fitted category dictionaries and the numerical scaler are computed
after splitting from retained real training buckets only. Excluded originals do
not enter either fitting or training. Raw histograms observed in held-out data
are storage, not an expanded model vocabulary. Unseen categories map to UNK slot
0, whose vector is fixed at zero; actual category code 0 is an ordinary separate
training category when present.

The branch's split policy is retained: 15-minute network-wide blocks, approximately
60/20/20 retained-original proportions, 100-second boundary guards and actual
flow-end envelope exclusions. All attack types are protected in training; validation
and test coverage is reserved only where feasible. No calibration fold is added.
Hosts/campaigns can recur across sets. This is same-environment held-out-time-block
evaluation, not unseen-host/campaign or zero-day evidence.

Whole-flow features belong to the flow-start bucket; this remains an offline
representation, not an exact measurement of packets transmitted within ten seconds.
The 82 original branch statistics are kept for a controlled representation change.
Protocol/L7/service histograms additionally support categorical embedding learning.

## Training-only donor selection and amount

The first component is wholly benign. The second is attack-bearing for synthetic
positives and benign for synthetic negatives. Donors are retained originals from
training, from different source IPs. Natural mixed benign/attack windows are kept.
Backgrounds use uniform available occupancy band (1, 2–4, 5–10), then source, then
sequence, preserving the selection rule of this remote branch. This is deliberately
different from the training-proportion-weighted sampler in the earlier local code.
Positive donors use attack family -> source -> sequence, with reuse allowed.

Each batch defaults to 25% real benign, 25% synthetic benign, 25% real malicious,
25% synthetic malicious. Labels select donors; they are not AE reconstruction
targets. A generation cycle counts P new variants of each synthetic class, where
P is the eligible original malicious training count. This is not P independent
new attacks, nor a guarantee that each original attack is selected exactly once.
Fresh combinations are generated online. Epochs are fixed optimiser budgets;
the generation counter continues across epochs. Every pair/shift is logged.

The selected circular shift k is uniform over 0..9, including zero. Use
`--shift-mode none` for a zero-shift mixing control; with the same seed its donor
choices match the circular run. The circular operation preserves
all donor positions, flow counts, labels and internal cyclic order, but creates
an artificial join at the old end/start boundary. It does not reduce a donor's
number of empty bins by itself. A real-only temporal control shares the exact
frozen bucket encoder and validation selection.

## Run from the repository root

Use the new checkout with NumPy, pandas and a working PyTorch installation. Do not
replace CUDA PyTorch on a GPU Pod with a CPU-only wheel. No Node/Excel runtime is
needed for V3. Run the fixture tests first:

```bash
python -m unittest CIC_new.vector_v3.test_pipeline -v
```

1. Prepare from the **full-schema raw CIC SQLite**, not the old reduced-column
   `cic_superposition/.../flows.sqlite` store:

```bash
python -m CIC_new.vector_v3.prepare \
  --db datasets/cic_2018_traffic.db \
  --out CIC_new/outputs/vector_v3_data
```

Alternatively, import the original supplied CSV inside the new output directory:

```bash
python -m CIC_new.vector_v3.prepare \
  --csv datasets/NF-CICIDS2018-v3.csv \
  --out CIC_new/outputs/vector_v3_data
```

Use exactly one source option. Existing 228-feature V2 shards contain the old
augmentation and do not substitute for this new preparation. Existing reports
and `local_dataset.json` continue to describe V2, not a completed V3 dataset.
The output contains raw originals, split/audit manifest, scaler/vocabularies and
normalised **real-only** shards. Whole-corpus preparation has not been run as part
of the implementation change. New V3 class counts must come from its actual build.

2. Learn categorical/bucket embeddings from real training buckets. For a smoke
   test use `--steps 20 --validate-every 10 --val-limit 128`; these are not useful
   final embeddings. Defaults below are initial development budgets:

```bash
python -u -m CIC_new.vector_v3.train bucket \
  --data CIC_new/outputs/vector_v3_data \
  --out CIC_new/outputs/vector_v3_bucket_ae \
  --device cuda --steps 5000 --batch-size 512
```

3. Freeze the selected checkpoint and encode real train/validation originals:

```bash
python -u -m CIC_new.vector_v3.train encode \
  --data CIC_new/outputs/vector_v3_data \
  --checkpoint CIC_new/outputs/vector_v3_bucket_ae/best.pt \
  --out CIC_new/outputs/vector_v3_encoded \
  --device cuda
```

The encoder is in eval mode with gradients disabled. The output records its
checkpoint checksum and data fingerprint. Test encoding is excluded by default;
`--splits test` to a new output is a later explicit frozen inference step.

4. Train the temporal AE with circularly shifted vector-space mixtures:

```bash
python -u -m CIC_new.vector_v3.train temporal \
  --data CIC_new/outputs/vector_v3_data \
  --encoded CIC_new/outputs/vector_v3_encoded \
  --out CIC_new/outputs/vector_v3_temporal_augmented \
  --device cuda --epochs 10 --steps-per-epoch 200 --batch-size 32
```

For a smoke test use `--epochs 1 --steps-per-epoch 10 --val-limit 128`.
For the matched control repeat with the SAME encoded store and optimiser budget,
a new `--out` and `--real-only`. Neither stage computes classifier detection
metrics or touches test examples for checkpoint selection. Both classes are
reconstructed; error is not a benign-only anomaly score.

The temporal run writes `history.csv`, `best.pt`, `last.pt`, `provenance.jsonl`,
frozen encoder/scaler copies and validation `projector_validation/vectors.tsv`
plus `metadata.tsv`. The vectors are sequence embeddings; metadata distinguishes
class, primary attack type, original source and occupancy. No extra test-derived
training choices should be made from visual clustering.

## Storage, recovery and limits

Original NPZ shards are cached once into disk-backed mmap arrays to avoid random
SQLite flow queries and repeated shard decompression for every GPU batch. Encoded
sequences are also mmap-backed; training mixes vectors only. This addresses the
CPU bottleneck observed in the earlier raw-flow online generator. Actual speed
must still be benchmarked on the target machine.

Allow tens of GB of free persistent disk for a full build: raw DB/event cache,
raw/normalised originals, dense histogram cache and frozen embeddings coexist.
Inspect `df -h`/disk capacity before building. Full CPU runtime, disk usage and
GPU throughput are unmeasured. Keep outputs under persistent storage on Runpod.

Every build/run requires a fresh directory. V3 does not implement stage resume or
an exact optimiser/generator resume; an incomplete directory is retained for
inspection rather than overwritten. Do not run concurrent cache builders against
one prepared directory. `.partial` cache folders cause an explicit error.

For validation, a fixed seed selects up to `--val-limit` real examples/buckets
without balancing or synthesis. Report class counts and per-class sequence MSE;
the overall curve may be dominated by benign traffic. Bucket reconstruction loss
includes categorical CE; temporal reconstruction loss is MSE on normalised learned
vectors plus the count channel. Their values, and those of the old 24/228-feature
models, are not directly comparable.

Additional experiments remain necessary: zero-shift versus circular-shift control,
real-only versus augmented training, shuffled temporal order, downstream classifier
evaluation and stricter event/host holdouts. Architecture/embedding dimensions
and the norm/volume design are starting decisions, not validated optima.
