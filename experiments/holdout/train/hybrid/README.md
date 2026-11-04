# CNN + LSTM hybrid baseline

Input: frozen Scheme2 `[N,10,65]` sequences plus the explicit bucket mask.
A 64-channel temporal convolution (kernel 3) feeds a 64-unit LSTM, followed by
masked mean/max pooling, its final hidden state and a 64-unit classification
head. Dropout 0.2. This is CNN followed by LSTM, not an ensemble of two predictions.

From the repository root:

```bash
python train/hybrid/train.py
# Explicit CPU mode works on macOS, Windows and Linux.
python train/hybrid/train.py --device cpu
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
