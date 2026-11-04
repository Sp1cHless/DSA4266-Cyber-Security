# XGBoost baseline

Input: flatten the ten Scheme2 buckets into 650 features, then append the ten
mask values. Preserve temporal positions; do not average away the sequence.
Use histogram gradient-boosted binary trees, depth 5, learning rate 0.05,
row/column subsampling 0.8, minimum child weight 5 and L2 regularization 1.

From the repository root:

```bash
python train/xgboost/train.py
python train/xgboost/train.py --device cpu
```

Run only this assigned model on your own machine. CPU hist is the default on
macOS, Windows and Linux; CUDA is unnecessary. XGBoost does not use MPS.
Share the completed run folder with the coordinator; local caches are unnecessary.

Defaults: `datasets/dataset`, CPU hist, four threads, at most 600 rounds,
early stopping patience 40 using the same sklearn AP as the neural models.
The best ensemble is saved as `model.ubj`; it excludes post-best trees.
`config.json` contains parameters. Runs go to a fresh `runs/` directory under
this folder. See [the shared guide](../README.md) for installation, weighting,
metrics, memory, all-model comparison and frozen evaluation.
