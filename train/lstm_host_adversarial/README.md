# LSTM host-adversarial experiment

This experiment uses the original frozen 100s/10s Scheme2 baseline dataset and
the original LSTM detector. It does not rebuild the encoder, scaler, split or
synthetic donor plan. Source IP is an auxiliary supervision target, never a
detector input. All generated artifacts stay in one new `train/runs/` directory.

Run from the repository root in the same environment as the selected baseline:

```powershell
.venv-cnn/Scripts/python.exe -u train/run_lstm_host_adversarial.py
```

`--dry-run` checks the reference/runtime/dataset contract without creating a run.
`--reference-run`, `--metadata`, `--cache`, and `--out` select existing inputs or a
fresh output location. Existing output directories are refused.

The fixed seed is inherited from the baseline (4266). Three new fits use maximum
gradient reversal strengths 0, 0.01 and 0.1, linearly ramped over the first three
epochs. The zero-strength refit must reproduce the saved same-runtime baseline
validation scores exactly. Detector initialization, dropout RNG, detection batch
order, equal-four-group BCE, AdamW, clipping, AP selection, epoch budget and
patience match the reference. Auxiliary initialization and sampling use isolated
RNG streams. The classifier still produces one binary attack score.

Two auxiliary 192 -> 64 -> K host classifiers operate separately on real benign
and real malicious representations. Only hosts with at least 20 real training
sequences enter these classifiers. Each detection step adds a separate batch of
128 examples per binary class, sampled uniformly by host then by sequence, with
replacement. The extra batch contributes only auxiliary host loss. Each host CE
is divided by log(K), then the two losses are averaged. Gradient reversal scales
and negates the representation gradient; auxiliary heads minimize their own CE
at full strength. Detector and auxiliary parameters have separate norm-1 clips.
Synthetic samples continue to train the original detector but have no single-host
auxiliary target. All host eligibility and coverage are recorded.

Independent 192 -> 64 -> K MLP probes use frozen selected-checkpoint features.
Original training sequences are partitioned into seeded global 15-minute blocks
(80/20), with 100-second guards and cross-fold actual-flow overlap exclusions.
No detector test feature or label shards are read. Probe hosts require at least
10 fit and 3 evaluation sequences. Up to 64 fit / 32 evaluation records per host
are sampled without replacement. Probe sampling, tasks and IDs are shared across
variants. Fit-only standardization, uniform-host training batches, and a fixed
25 epochs avoid selecting probes on diagnostic evaluation scores. Tasks include
benign, all attacks, and feasible individual attack subtypes. Unsupported tasks
are recorded as not estimable rather than silently treated as zero accuracy.

`REPORT.md` and the overall, attack and probe comparison CSVs distinguish real
validation detection metrics from training-derived probe diagnostics. Every
attack uses the model's global validation-FPR threshold; subtype and host-macro
recall are reported at both 1% and 0.1% budgets. Each variant retains its
checkpoint, history, validation predictions, thresholds, and independent probes.
The run root also records source hashes, reference hashes, coverage, exact probe
IDs, partition rules, and status.

These are single-seed exploratory fits on reused training and already inspected
validation cohorts. Hosts/campaigns can recur across original splits. Probe
evaluation records were used to train the detector; probe metrics are therefore
training-derived diagnostics, not held-out detector evidence. Reduced probe
accuracy is not proof of complete host invariance. Subtypes confined to one
source cannot separate attack, host and campaign effects. Multiple supported
hosts are required for within-subtype host probes. Results do not establish
cross-host generalization or a new independent test claim.
