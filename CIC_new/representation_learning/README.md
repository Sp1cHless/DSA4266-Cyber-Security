# Scheme2 input representation

This stage fits an input encoder after preprocessing has assigned the original
train/validation/test splits. The complete checked entry point is
[`prepare_scheme2_dataset.py`](../SCHEME2_DATASET.md). Downstream prediction-model
training belongs to `train/`.

The bucket autoencoder reconstructs scaled numeric statistics with MSE and
category distributions with soft-target cross-entropy. Default protocol, L7 and
destination-service tables have 16 dimensions each; the whole-bucket vector has
64 dimensions and is trained as a unit vector. Service groups destination ports
into DNS, web, SSH, FTP, mail, RDP, SMB and other. Numeric features retain port
diversity statistics. UNK vectors stay zero.

Only real training active buckets fit weights. A fixed real validation subset
selects the checkpoint. Test does not fit or select any parameters. The bucket
encoder is frozen before superimposition so downstream training cannot move its
own input representation.

## Individual stages

From the repository root, with new output directories:

```powershell
python -u -m CIC_new.representation_learning.learn_bucket_embeddings --data <prepared> --out <fresh-bucket-embeddings> --device cpu --steps 5000 --batch-size 512
python -u -m CIC_new.representation_learning.encode_sequences --data <prepared> --checkpoint <bucket-embeddings>/best.pt --out <fresh-encoded-sequences> --device cpu --splits train validation test
```

CUDA can be selected on a supported machine. The direct builder defaults to
automatic CUDA/CPU selection. For a small artificial plumbing check, use a tiny
fixture rather than fitting a full source database.

`bucket_autoencoder.py` defines the input encoder; `learn_bucket_embeddings.py`
fits it; `encode_sequences.py` applies the frozen checkpoint; `runtime.py`
provides their argument, device, output and seed helpers.

Encoding runs without optimizer updates. Include all three splits explicitly;
the low-level encoding command otherwise defaults to train/validation only.
Outputs are disk-backed embeddings, masks, labels, counts and original metadata.
Manifests record checkpoint/preparation fingerprints; schema, scaler and encoder
are copied for provenance. Raw numeric/histogram arrays use mmap caches to avoid
repeated NPZ decompression. Keep these artifacts on persistent storage.

Continue with the direct builder's frozen-vector export and full validation.
See [the preprocessing guide](../preprocessing/README.md) for the final schema.
