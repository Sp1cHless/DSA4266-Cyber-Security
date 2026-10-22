# Scheme1 per-bucket L2 ablation

Historical reproduction only. Scheme2 is now selected. After local cleanup,
rebuild retired Scheme1 outputs before running this tool. Completed results are in
[the comparison archive](reports/superimposition_comparison_20261022/SUMMARY.md).

Run from the repository root in VS Code:

```powershell
.\.venv-cnn\Scripts\python.exe -u CIC_new\run_scheme1_l2_experiment.py
```

This selects completed original Scheme1/Scheme2 references, verifies pairing,
checksums and L2 arithmetic, trains a new Scheme1 L2 temporal AE, and performs
frozen evaluation with three-way comparisons. Existing data, caches and models
remain unchanged. Each run creates a unique outputs/ae_runs/<timestamp>_scheme1_l2_ae_aug_<suffix>/.

The only added operation is X / max(||X||_2, 1e-8) per active bucket after final
statistical composition, category embeddings and scaling. All 137 coordinates,
including flow count, are normalized without a separate count channel. Padding
stays zero. The transform is stateless and identical during training/evaluation.
L2 controls vector length, not relative coordinate scales. The shared X.npy cache
is unchanged: use ModelInputs(..., input_transform="bucket_l2").take()/batches().
Both config and checkpoint record the transform; evaluation restores it.

Inherited controls: seed 4266, batch 128, 500 updates/epoch, maximum 500 epochs,
patience 20, cumulative relative improvement 0.1%, hidden width/latent size 32.
Four sample groups each contribute 25%, with the same donor plan, zero shift,
and all real validation examples.

```powershell
python CIC_new\run_scheme1_l2_experiment.py --stage check
python CIC_new\run_scheme1_l2_experiment.py --stage train
python CIC_new\run_scheme1_l2_experiment.py --stage evaluate --run <completed-l2-run>
```

Check fits nothing. Evaluation-only creates a new evaluation directory with AE
weights frozen. Pin references with --baseline-run/--scheme2-run; --out selects
fresh training output and refuses nonempty directories. No optimizer resume.

Results under evaluations/<timestamp>_scheme1_l2_comparison_<suffix>/ include
THREE_WAY.md/three_way_comparison.json, comparison.md/json against Scheme1,
vs_scheme2/comparison.md/json, report.json and evaluation_status.json.
Metrics use a real-train-only linear readout of frozen AE latent vectors, not
the planned CNN. Compare AP, ROC-AUC and low-FPR/mixed-flow recall rather than raw
MSE. Fixed-FPR points are exploratory validation statistics; test is reserved.
