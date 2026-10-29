# Scheme2 distribution-feature comparison

Run from the repository root using the existing environment:

```powershell
.\.venv-cnn\Scripts\python.exe -u train/run_distribution_feature_ablation.py --dry-run
.\.venv-cnn\Scripts\python.exe -u train/run_distribution_feature_ablation.py
```

The script trains only `distribution97`: original 82 plus 15 fixed
distribution/joint features, with a fresh bucket encoder and original LSTM.
It reuses the completed seed-4266 original-feature baseline from
`train/runs/20261024_223102_671768_circular_multiseed_start5_ef34ba/seed4266/lstm/start0`.
No baseline training is scheduled. Existing baseline weights, predictions,
configuration and donor plan are read and hash-checked without modification.
Use `--reference-run` to select another completed compatible baseline.
No attention or circular shift is used.
The default is seed 4266, 30 maximum epochs and patience 5, with the original
optimizer, group weighting, batch size and checkpoint/threshold policies.
Encoder defaults copy the original 5,000-step, 64D, batch-512 protocol. The
current Python executable is used by every child process; no environment or
package is installed or changed.

## Added features

The bundle `distribution_joint_v1` is fixed before formal training:

- Flow-level standard deviations of incoming/outgoing bytes and packets, and duration: 5.
- 90th percentiles of incoming/outgoing bytes and duration: 3.
- Fraction of flows with zero outgoing packets: 1.
- Mean and standard deviation of `in_bytes / max(in_bytes + out_bytes, 1)`: 2.
- Mean and standard deviation of total bytes per packet: 2.
- Distinct `(destination, destination port)` pairs divided by flow count: 1.
- Median flow duration: 1.

These are a compact subset of the earlier 25-feature distribution screening
family, not a bundle whose final LSTM performance has already been established.
TTL, throughput, packet IAT, raw identifiers, absolute timestamps and DNS query
IDs are excluded. Destination identities contribute only equality/diversity.

## Preserved data and method

Original source-IP windows, 100s/10s alignment, train/validation IDs, labels,
category vocabularies, masks, flow counts and the old 82 normalized values are
preserved. There is no split or original bucket recomputation. The extended
preparation makes one extra pass over existing staged training/validation flows
to compute the new statistics. It does not rescan/reimport the source CSV.

Only the new scalers are fitted, using all real retained training active buckets.
Original numeric scales remain exact prefixes. Ratios remain unlogged; new
nonnegative size/duration statistics use log1p and standardization. The expanded
numeric width is 97. Encoder bottlenecks and output remain unchanged; the encoder
input/output layers grow, while downstream LSTM inputs and parameter counts stay
unchanged at 65 feature channels plus the mask input.

After encoding, Scheme2 augmentation remains the original frozen-vector addition
and L2 normalization. Standard deviations, quantiles and ratios are never added
as if they were raw superimposable counts. The new run and existing baseline
must have identical donor plans and prediction cohorts before comparison.

Only train and validation feature files are produced/read. Original all-split
ID/label metadata is retained for donor ownership and synthetic-ID allocation;
test features are not read, encoded or scored. Completed-flow summaries preserve
the original offline information-availability assumption.

## Outputs and recovery

Default outputs use a new unique directory:

`train/runs/<timestamp>_lstm_distribution_features_seed4266_<id>/`

- `batch.json`: settings, source/code/runtime identities and stage records.
- `distribution97/`: preparation, representation, original-LSTM
  run directories and live logs.
- `COMPARISON.md`, `attack_comparison.csv`: Infilteration/Web TP/FN and recall
  at both global FPR budgets, including realized FPR.
- `comparison.json`: full overall metrics and paired-cohort checks.

If interrupted, resume with the same environment/code/settings:

```powershell
.\.venv-cnn\Scripts\python.exe -u train/run_distribution_feature_ablation.py --resume "D:\path\to\the\batch"
```

Completed stages are hash-checked and reused. An incomplete stage is preserved
and restarted in a new numbered attempt; partial optimizer state is not resumed.
Existing experiment data and weights are not overwritten. The runner checks
free disk before representation fitting. Once a representation is verified,
it removes only this batch's disposable prepared mmap and encoded-array caches;
once the detector is verified, it removes that attempt's disposable input cache.
Weights and JSON provenance inside caches are copied to `cache_cleanup/` first.
Prepared split shards, raw extension values, final datasets, model weights,
predictions, logs and reports remain. Original source caches are never removed.
A failed stage may retain
partial caches; preserve them or inspect them before resuming when disk is tight.
Run a single process for a given batch directory.

This is an exploratory same-cohort, single-seed comparison. The new fit reuses
original training data and previously inspected validation data. The new feature
width requires a fresh encoder while the baseline keeps its existing encoder;
the comparison does not isolate added features from encoder refitting. It is not a new
independent held-out test. Overall AP still selects checkpoints; target attack
recall does not select checkpoints or attack-specific thresholds.

## Verification status

The artificial end-to-end check creates a small completed baseline, then verifies
that the experiment schedules only the 97-feature case and preserves every
baseline file hash. It also checks unchanged old features, frozen encoding,
Scheme2 augmentation, identical donor/cohort IDs, unavailable test files and resume.
The default real configuration was dry-run checked. The real experiment has not
been launched by the coding agent.
