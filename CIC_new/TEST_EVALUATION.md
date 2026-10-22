# Frozen evaluation on the real held-out test set

Historical paired-comparison tool. Scheme2 is now selected; after local cleanup,
retired Scheme1 data/checkpoints must be rebuilt to repeat this command. Results remain
in outputs/test_evaluations/ and [the comparison archive](reports/superimposition_comparison_20261022/SUMMARY.md).

The user explicitly requested evaluation of the existing test set. Run from the
repository root after matched Scheme1 L2 / Scheme2 validation evaluation:

```powershell
.\.venv-cnn\Scripts\python.exe -u CIC_new\evaluate_test_set.py
```

This evaluates the original real test shards at natural prevalence, with no
synthetic construction, AE retraining, probe fitting or test threshold selection.
The two frozen AE checkpoints and real-train-only linear probes are taken from
the completed L2 three-way comparison. It is not the planned CNN evaluation.

The command writes `frozen_evaluation_plan.json` before opening test features:

- Freeze model/probe/representation hashes and input transformations.
- Preserve balanced-prior 0.5 and train-benign 99%/99.9% score thresholds.
- Additionally fix 1%/0.1% FPR thresholds using only the old saved validation
  predictions. Applying these to test does not guarantee exactly the same FPR.
- Verify paired test shard checksums, metadata, IDs, labels, masks, counts and
  original-only membership. Legacy Scheme2 attack-flow counts come from the
  exactly paired Scheme1 test metadata.

Inference streams shards directly, creating no test mmap caches. Model weights,
existing training/evaluation artifacts and datasets are unchanged. Test scores
cannot adjust thresholds, model parameters or checkpoint choice during the run.

Each run creates a unique directory:
`outputs/test_evaluations/<timestamp>_scheme1_l2_vs_scheme2_<suffix>/`.

- `TEST_COMPARISON.md`: validation/test ROC-AUC/AP, actual test FPR, recall,
  precision and counts at frozen thresholds, plus per-attack results.
- `test_slices.csv`: attack, attack x mixture, flow fraction, active-bucket and
  volume slices at every frozen operating point.
- `scheme1_l2_test_scores.npz` / `scheme2_test_scores.npz`: paired IDs, labels,
  flow fractions, metadata, logits and reconstruction errors.
- `report.json`: complete metrics, paired source-host bootstrap and provenance.
- `status.json`: completion; explicitly records test_used=true and no fitting.

Pin references with `--l2-run`; use a fresh `--out`, `--device cpu/cuda`, or
`--batch-size` if needed. Existing output directories are refused.

The first completed real-test comparison used 263125 originals: 851 attack and
262274 benign sequences. Scheme2 test ROC-AUC/AP were 0.84948/14.93%, versus
Scheme1 L2 0.82768/7.27%. This supersedes any prior recommendation based only on
train-derived weak-fragment stress tests. Those stress results remain valid for
their limited population but are not held-out test performance.

These are single-seed temporal AE plus linear-readout results under the existing
time-block split, not unseen-host/campaign guarantees. Subsequent tuning informed
by these test results requires an additional untouched holdout for another final
generalization claim.
