# Preprocessing: raw CIC data to reusable datasets

Preprocessing does not train a prediction model. The main command runs the whole
raw-data preparation path:

```text
Raw CSV / full-schema SQLite
 -> group by source IP and construct nonoverlapping 100s sequences
 -> aggregate ten 10s buckets, retaining raw categorical histograms
 -> assign/purge original train / validation / test sequences
 -> fit numerical scaling and category vocabularies on real TRAIN only
 -> save the three prepared splits
```

## 1. Run preprocessing end to end

From the repository root:

```bash
python -m CIC_new.preprocessing.run_preprocessing \
  --csv datasets/NF-CICIDS2018-v3.csv \
  --out CIC_new/outputs/preprocessed
```

Alternatively use `--db datasets/cic_2018_traffic.db`, supplying exactly one source.
The SQLite source must have the full CIC schema; the old reduced-column
`cic_superposition/.../flows.sqlite` is insufficient. Only NumPy and pandas are
needed for this command. There is no PyTorch import, CUDA requirement, neural
network, optimiser or AE training in this stage. No Node/Excel runtime is needed.

Output:

```text
preprocessed/
  splits/train/part_*.npz
  splits/validation/part_*.npz
  splits/test/part_*.npz
  schema.json             # category dictionaries and feature definitions
  scaler.json             # fitted only on retained real training buckets
  split_notes.json        # assignments, exclusions and family coverage
  manifest.json           # completion and source metadata
  manifest.sqlite         # original IDs, sources and provenance
  raw_originals/          # unfitted bucket statistics/histograms for auditing
  work.sqlite            # original event cache
```

Each prepared shard contains `numeric` (82 statistical features per bucket),
`hist_protocol`, `hist_l7`, `hist_service`, `mask`, `y_bucket`, `y_sequence`,
`sample_id`, `flow_counts` and `attack_flow_counts`. These are real originals;
train/validation/test splitting is complete at this point. A model may use these
statistics/histograms directly, or use a separately learned representation.

## 2. Optional representation learning, outside preprocessing

The professor requested augmentation in an embedding space. If embeddings are
learned, their weights need a separate fitting step. That is representation
learning, not part of the raw preprocessing command or a choice of final detector.
See [representation_learning/README.md](../representation_learning/README.md).

The supplied embedding learner uses a bucket autoencoder as one self-supervised
method. It is not a requirement that the final model must also be an AE. It learns
16-dimensional protocol/L7/destination-service lookup vectors and a 64-dimensional
bucket representation, then freezes that encoder. The frozen representations can
feed CNN, LSTM, AE or classifier experiments. Other embedding-learning methods
can be substituted if they export the same frozen-sequence contract.

## 3. Export a fixed augmented dataset, independent of any final model

After separately learning and freezing embeddings and encoding all three splits:

```bash
python -m CIC_new.preprocessing.export_augmented_dataset \
  --data CIC_new/outputs/preprocessed \
  --encoded CIC_new/outputs/encoded_sequences \
  --out CIC_new/outputs/model_inputs
```

This command runs on NumPy, with no model training. It retains all real originals,
adds P synthetic benign and P synthetic malicious examples to TRAIN, and exports
real-only validation and test. P is the original malicious training count. Use
`--originals-only` for an unaugmented export and `--shift-mode none` for a zero-shift
mixing control. Fresh output directories prevent overwriting previous data.

```text
model_inputs/
  train/part_*.npz         # N real + P synthetic benign + P synthetic malicious
  validation/part_*.npz    # real originals only
  test/part_*.npz          # real originals only
  manifest.json
  schema.json
  scaler.json
  frozen_bucket_encoder.pt
```

Final shard keys:

| Key | Meaning |
|---|---|
| `X` | float32 `[samples,10,features]`; default 64 vector components + one flow-count channel |
| `mask` | Active positions; empty positions remain zero in X |
| `y_sequence`, `y_bucket` | Sequence and bucket targets, separate from X |
| `sample_id`, `sample_type` | Unique exported ID and real/synthetic class group |
| `left_id`, `right_id`, `right_shift_buckets` | Donor IDs and circular shift; absent second donor is -1 |
| `flow_counts` | Flow counts after any rotation/combination |
| `left_ip`, `right_ip`, `primary_attack` | Audit/visualisation metadata, never model inputs |

Any model can read exactly these same inputs, including TensorFlow or other
frameworks without importing PyTorch:

```python
from CIC_new.preprocessing.export_augmented_dataset import iter_model_shards

for batch in iter_model_shards("CIC_new/outputs/model_inputs", "train"):
    X, mask, y = batch["X"], batch["mask"], batch["y_sequence"]
    # Pass to your chosen CNN, LSTM, AE, classifier, etc.
```

`ModelInputs` in `dataset.py` adds optional mmap caching and balanced batch
sampling. Batch priors are modelling choices; the exported dataset itself is
not made 50:50 by deleting real benign examples. The optional AE experiment uses
four-way balanced training batches. Validation and test must stay unaugmented
and naturally distributed.

## Augmentation definition

`augmentation.py` rotates donor B along its ten time positions, adds the two
bucket embedding vectors and L2-normalises each resulting vector. Masks, bucket
labels and counts follow the same rotation. All flow starts/positions survive;
there is no raw-record merging, cropping or shift of entire capture days.
Empty positions are zeroed. Cancellation of active vectors does not mark them empty.

The first donor is benign-only. The second is benign or attack-bearing according
to the intended synthetic label. Both are retained training originals from
different source IPs. Backgrounds are sampled by uniform available occupancy
band (1, 2–4, 5–10), then source, then sequence; positives by family/source/sequence.
Reuse is allowed. P variants do not mean P new independent attacks or guarantee
every original attack was selected once. Rotation is uniform over 0..9, including
zero. It preserves cyclic order but introduces an artificial end/start join;
rotation alone does not reduce the number of empty buckets.

Numerical feature vectors are not added: means and entropies are not additive.
A separate log1p total-flow-count channel, using the original-training scaler,
preserves intensity lost through L2 normalisation. Synthetic vectors are mixtures
in learned space, not exact physical reconstructed NetFlow records.

## Splits, leakage and practical limits

The existing branch's 15-minute global blocks, approximate 60/20/20 allocation,
100-second boundary guards and actual-flow-end envelope exclusions are retained.
All attack types are protected in training; held-out coverage is reserved where
feasible. Hosts and campaigns can recur across splits, so this is not an
unseen-host/campaign or zero-day evaluation. No extra calibration fold is added.

Category histograms and bucket statistics are calculated before splitting, but
vocabularies and numerical scaling are fitted afterward on retained real TRAIN
only. Unknown categories use slot 0; actual category code 0 is a separate value.
Whole-flow statistics belong to the start bucket, which is an offline summary,
not an exact measurement of traffic transmitted within that ten-second interval.

Valid existing V3 prepared data and embedding checkpoints keep their formats and
can still be used after this code reorganisation. Historical V2 raw-flow mixtures
and the old 24-feature experiment are different inputs. They are not substitutes.

Allow tens of GB of persistent free disk for full preparation, raw caches,
histograms, encoded originals and exported/mmap model inputs. Use `df -h` before
building. No full-corpus runtime or GPU speed claim is established by fixture
tests. New commands require fresh output directories; there is no exact stage or
optimiser resume. Do not run concurrent builders against one cache/output.

Tests: `python -m unittest CIC_new.tests.test_embedding_pipeline -v`.
