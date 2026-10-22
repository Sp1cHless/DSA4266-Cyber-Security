# Temporal CNN baseline

Input: frozen Scheme2 `[N,10,65]` sequences plus the explicit bucket mask.
Two temporal convolutions (64/128 channels, kernel 3), masked mean/max pooling,
and a 64-unit binary classification head. Dropout 0.2. The temporal axis is the
ten buckets; embedding coordinates are channels, not an arbitrary image grid.

From the repository root:

```bash
python train/cnn/train.py
# Explicit CPU mode works on macOS, Windows and Linux.
python train/cnn/train.py --device cpu
```

Run only this assigned model on your own machine. Default `auto` uses CUDA only
when available and otherwise CPU, including on a Mac. Optional `--device mps`
requests an available Apple GPU; actual MPS training needs verification on the Mac.
Share the completed run folder with the coordinator; local caches are unnecessary.

Defaults: `datasets/dataset`, batch 512, AdamW lr 0.001, at most 30 epochs,
early stopping patience 5 on complete real-validation AP. `config.json` contains
the architecture and optimization settings. Runs go to a fresh `runs/` directory
under this folder. See [the shared guide](../README.md) for installation, weighting,
metrics, output files, all-model comparison and frozen evaluation.
