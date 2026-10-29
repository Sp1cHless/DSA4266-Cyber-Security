# One-run LSTM bucket-supervision experiment

Run from the repository root with the existing environment:

```powershell
.\.venv-cnn\Scripts\python.exe -u train/run_lstm_bucket_ablation.py --dry-run
.\.venv-cnn\Scripts\python.exe -u train/run_lstm_bucket_ablation.py
```

The runner schedules one new seed-4266 model and compares it with the completed
chronological original LSTM at
`train/runs/20261024_223102_671768_circular_multiseed_start5_ef34ba/seed4266/lstm/start0`.
It does not retrain the baseline, bucket encoder or distribution-feature model.
Source features, sequence/bucket lengths, split IDs, synthetic donor pairs and
augmentation are reused exactly through the reference dataset manifest.
Existing weights and data are never overwritten. Every new batch has a unique
directory. No environment changes or dependency installation are required.

## Fixed intervention

The original LSTM, masked mean/max pooling, final hidden state and sequence
classification head are retained. A zero-initialized linear bucket head maps
each recurrent state to one bucket logit. It adds 65 parameters at hidden size
64, giving 46,274 total versus the original 46,209. Shared parameters retain
the original seeded initialization; creating the auxiliary head preserves the
baseline initialization RNG stream. Both models train from scratch.

The existing `y_bucket` labels provide training targets. A bucket is positive
when any constituent flow has an attack label; mixed attack/benign buckets are
positive. Synthetic labels are reused from the original exported donor union.
Targets are loaded in original shard/sample order and reconciled against
cached IDs, masks and sequence labels. They remain separate from model inputs.
The original shared feature cache is reused without adding or replacing files.

For each sequence i, the objective is

`w_i * (BCE(sequence_logit_i, y_i) + 0.5 * mean_active_buckets(BCE(bucket_logits_i, y_bucket_i)))`.

The final minibatch loss is the mean over sequences. Empty buckets have no
auxiliary loss. The per-sequence active-bucket mean prevents dense sequences
from receiving extra weight just because they contain more buckets. The same
original four-group weights apply to both terms, with no added positive-bucket
weighting. Lambda is fixed at 0.5 before training; this runner performs no
weight search or lambda-zero baseline run.

AdamW, learning rate .001, weight decay .0001, batch size 512, gradient clipping
at norm 1, shuffled batches, maximum 30 epochs and patience 5 follow the
reference. The combined objective is clipped using that same global norm rule.
Checkpoint selection remains full real-validation sequence AP. Bucket-level
validation targets are loaded only after the checkpoint has been selected.
Sequence inference continues to use the original sequence head. The auxiliary
head does not provide an alternative sequence score or an oracle mask.
Global 1%/0.1% FPR thresholds follow the unchanged validation policy.

## Outputs

`train/runs/<timestamp>_lstm_bucket_supervision_seed4266_<id>/` contains:

- `plan.json`: immutable reference hashes, data/runtime/code identities and the fixed intervention.
- `bucket_supervised/`: checkpoint, training history, normal sequence predictions and evaluation audits.
- `bucket_supervised/bucket_target_audit.json`: label checks, population counts and label-array hashes.
- `bucket_supervised/bucket_predictions.npz` and `bucket_metrics.json`: descriptive active-bucket validation results, unused for checkpoint or threshold selection.
- `COMPARISON.md`, `attack_comparison.csv`, `comparison.json`: overall metrics and Infilteration/Web TP, FN, recall, recovered/lost counts at both global FPR budgets.

The runner reads and hashes completed baseline artifacts, and verifies their
hashes again after the new fit. Detector architecture/training settings,
prediction cohorts, dataset identity and environment must match. The original
four baseline adapters continue to use the original training engine; only
`lstm_bucket` opts into the auxiliary engine.

The direct module entry point is also supported:

```powershell
.\.venv-cnn\Scripts\python.exe -m train.lstm_bucket.train --help
```

Use the ablation runner above for the established reference settings and
automatic comparison. An interrupted run is preserved; a new invocation starts
a new batch and trains from scratch. There is no partial-optimizer resume.

## Interpretation and verification

This is one exploratory supervised validation comparison using the original
training data and the previously inspected validation cohort. It adds temporal
label supervision while preserving the inputs; gains cannot be attributed to
architecture capacity alone. Bucket labels inherit source-flow label quality
and the any-attack convention. No test feature or label shard is opened.

Focused artificial checks cover identical baseline parameter initialization and
RNG state, exact sequence outputs at initialization, masked per-sequence loss,
auxiliary gradients into the LSTM, a lambda-zero training equivalence check,
new-model fit/reload, unchanged source/reference hashes, label alignment and
unavailable test files. Real training is left to the user.
