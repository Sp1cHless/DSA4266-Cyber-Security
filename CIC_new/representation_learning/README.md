# Learn reusable input embeddings

This is the selected Scheme2 whole-bucket learner. Retired Scheme1 independently learns
only L7/source-port/destination-port tables using `CIC_new/train_embeddings.py`;
see [its guide](../scheme1/README.md). No teammate checkpoint is needed for scheme1.

This stage fits an input representation after raw preprocessing has already
created train/validation/test. It does not select or train the final detector.

Learned categorical embeddings require weight fitting. The current method uses
a **bucket autoencoder** to reconstruct scaled numerical statistics (MSE) and
category distributions (soft-target cross-entropy). This prevents the learned
representation from becoming an arbitrary category-ID sum. The bucket representation
is trained as a unit vector, so exporting unit vectors does not discard information
encoded only in magnitude. The method is an implementation choice; the professor
did not require a bucket AE or a final temporal AE.

## Learn embeddings using real training buckets

```bash
python -u -m CIC_new.representation_learning.learn_bucket_embeddings \
  --data CIC_new/outputs/preprocessed \
  --out CIC_new/outputs/bucket_embeddings \
  --device cuda --steps 5000 --batch-size 512
```

For a plumbing check use `--steps 20 --validate-every 10 --val-limit 128`.
Defaults are 16 dimensions per categorical field and 64 per combined bucket.
The fields are protocol, L7 and service. Service is a destination-port grouping
(DNS, web, SSH, FTP, mail, RDP, SMB and other), not an embedding of every raw port
number. Source/destination-port diversity statistics remain in the numeric input.
Vocabularies/scaling already come from preprocessing; only real training buckets
fit the weights, and real validation buckets select the checkpoint. UNK's vector
is fixed at zero. Test does not select weights or hyperparameters.

The network lives in `bucket_autoencoder.py`. Its optimisation loop lives in
`learn_bucket_embeddings.py`, clearly outside preprocessing. Once trained, the
encoder is frozen to keep later augmentation inputs/targets stable. Optimising
a downstream reconstruction model must not collapse or move its own input targets.

## Encode original sequences without further training

```bash
python -u -m CIC_new.representation_learning.encode_sequences \
  --data CIC_new/outputs/preprocessed \
  --checkpoint CIC_new/outputs/bucket_embeddings/best.pt \
  --out CIC_new/outputs/encoded_sequences \
  --device cuda --splits train validation test
```

This is frozen inference: no optimiser or gradient updates run. All three folds
use the same encoder. The default without `--splits` encodes train/validation only;
include test explicitly when exporting the final three-fold dataset. Test encoding
is application of an already selected transform, not fitting on test.

Output per split is disk-backed bucket embeddings, masks, labels, counts and
original metadata. The manifest records checkpoint and preparation fingerprints.
The encoder and schema/scaler are copied for reproducibility. Numerical and histogram
source arrays are cached as mmap files, with NPZ decompression performed once.

Next run [export_augmented_dataset](../preprocessing/README.md) to produce fixed
model-independent training/validation/test arrays. Any final architecture may use
them. Existing Torch checkpoints remain state-dict compatible after the file moves.
Keep code, caches and checkpoints under persistent storage; full-data cache disk
requirements and GPU throughput must be measured on the target machine.
