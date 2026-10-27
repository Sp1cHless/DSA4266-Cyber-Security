# Training-holdout protocol: unseen attack subtypes and families

This protocol trains the Scheme2 detector without selected attack labels, then
measures whether those labels are still flagged. The detector stays binary; a
holdout changes only which real training sequences and fitted transforms see the
labels. Two holdout sets have been built and run with the same protocol.

## Holdout sets

### A. One subtype per multi-subtype family

| Family | Held out from training | Retained for training | Removed train sequences | Validation | Test |
|---|---|---|---:|---:|---:|
| Web | `Brute_Force_-XSS` | `Brute_Force_-Web`, `SQL_Injection` | 122 | 37 | 62 |
| DoS | `DoS_attacks-SlowHTTPTest` | GoldenEye, Hulk, Slowloris | 13 | 7 | 7 |
| DDoS | `DDOS_attack-HOIC` | LOIC-HTTP, LOIC-UDP | 81 | 94 | 65 |

### B. All flooding (DoS and DDoS together)

| Held out from training | Removed train sequences | Removed attack flows | Validation | Test |
|---|---:|---:|---:|---:|
| `DDOS_attack-HOIC` | 81 | 193,295 | 94 | 65 |
| `DDoS_attacks-LOIC-HTTP` | 141 | 98,859 | 64 | 110 |
| `DDOS_attack-LOIC-UDP` | 60 | 1,094 | 54 | 14 |
| `DoS_attacks-SlowHTTPTest` | 13 | 39,170 | 7 | 7 |
| `DoS_attacks-GoldenEye` | 6 | 32,860 | 2 | 6 |
| `DoS_attacks-Hulk` | 6 | 75,640 | 0 | 0 |
| `DoS_attacks-Slowloris` | 6 | 8,914 | 16 | 0 |

Single-subtype labels (`FTP-BruteForce`, `SSH-Bruteforce`, `Bot`, `Infilteration`)
do not define a held-out family and stay in training as seen reference types in
both sets. Counts are real sequences in the control Scheme2 dataset and were
recorded before any scoring. Set B removes 313 of 8,093 training positives (3.9%).

## Rules

1. **Train-fold only.** Held-out labels are removed from `split == "train"`
   sequences, selected by `primary_attack`. They stay in validation and test.
2. **Fold assignment is unchanged.** Reservations, seeded RNG, boundary purge and
   ratio balancing run exactly as without a holdout; the flip to `excluded`
   happens afterwards. Validation and test populations are identical to the
   control dataset, which is checked during the build.
3. **Every fitted transform excludes them.** Vocabulary, numeric scaler, frozen
   bucket encoder and donor plan fit on the training fold only, so they never see
   held-out training buckets. The bucket encoder must be refitted for every
   holdout set, even though sequence construction is identical.
4. **No transform refits at evaluation.** Operating thresholds are selected on each
   model's own real validation split and reused unchanged for test.
5. **Disclosure.** Validation retains the held-out labels, so they influence
   input-encoder checkpoint selection. Test is scored once per model after
   training. The Scheme2 test split was inspected in earlier project work; these
   results are a frozen comparison, not a new untouched holdout.

## Build

```powershell
python -u CIC_new/prepare_scheme2_dataset.py --csv datasets/NF-CICIDS2018-v3.csv `
  --out CIC_new/outputs/scheme2_runs/<fresh>_holdout_all_flooding `
  --holdout-train-attacks "DoS_attacks-GoldenEye,DoS_attacks-Hulk,DoS_attacks-Slowloris,DoS_attacks-SlowHTTPTest,DDOS_attack-HOIC,DDoS_attacks-LOIC-HTTP,DDOS_attack-LOIC-UDP" `
  --device cuda
```

`prepare_scheme2_dataset.py --db <source.db>` is the alternative source option. A
holdout cannot be added to an existing `--prepared` directory. The final `dataset/`
records `train_holdout_attacks`; copy only that directory for training.
`workflow.json` and `preprocessed/split_notes.json` carry the per-type audit,
including how many held-out flows survive as non-primary members of retained
training sequences (sequence-level exclusion, not flow-level). `split_notes.json`
separates primary-label counts from membership counts.

## Checks before training

- `workflow.json` status `complete`; `dataset/validation.json` `passed: true`.
- No training row has a held-out `primary_attack`; validation/test per-subtype
  counts equal the control dataset; training counts equal control minus the
  removed sequences; `P` equals the reduced real malicious training count.
- Set A: `P = 7877`. Set B: `P = 7780`.
- Shard checksums differ from the control dataset; manifest fingerprints differ.

## Report

```powershell
python train/unseen_subtype_report.py --control <four control runs> `
  --holdout <four holdout runs> --holdout-dataset datasets/dataset_holdout_all_flooding
```

The report refuses mismatched prediction populations, so it also verifies that the
split did not move. It writes `UNSEEN_SUBTYPE.md`, `unseen_subtype.csv` and
`manifest.json` in a fresh run directory: per-subtype recall at the 1% and 0.1%
false-positive budgets for control and holdout, deltas, Wilson intervals, pooled
held-out recall and overall AP. `train/compare.py` is still used per dataset; it
refuses to mix the two dataset fingerprints.

## Frozen test outcome

Control versus holdout on the frozen test split, each run scored with its own
saved validation thresholds. Every arm was trained with seeds 4266, 4267 and 4268
and paired by seed; values are mean +/- sample standard deviation over the three
seeds, and every delta is the paired holdout-minus-control difference.

| Model | Set A pooled R@1% (n=134) | Delta | Set B pooled R@1% (n=202) | Delta | Seeds lower (B) |
|---|---|---|---|---:|---:|
| cnn | 99.75% -> 99.00% +/- 0.43 | -0.75 +/- 0.75 | 100.00% -> 68.32% +/- 0.00 | -31.68 +/- 0.00 | 3/3 |
| hybrid | 99.75% -> 98.76% +/- 1.14 | -1.00 +/- 0.86 | 100.00% -> 68.32% +/- 0.00 | -31.68 +/- 0.00 | 3/3 |
| lstm | 99.25% -> 98.76% +/- 1.14 | -0.50 +/- 1.14 | 99.50% -> 72.94% +/- 4.00 | -26.57 +/- 3.14 | 3/3 |
| xgboost | 96.27% -> 99.00% +/- 0.43 | +2.74 +/- 0.43 | 96.37% -> 68.65% +/- 0.29 | -27.72 +/- 0.50 | 3/3 |

Overall test average precision:

| Model | Control | Set A | Set B | Delta AP (A) | Delta AP (B) |
|---|---|---|---|---|---|
| cnn | 0.5747 +/- 0.0088 | 0.5806 +/- 0.0041 | 0.5047 +/- 0.0105 | +0.006 +/- 0.012 | -0.070 +/- 0.004 |
| hybrid | 0.5788 +/- 0.0139 | 0.5718 +/- 0.0107 | 0.4868 +/- 0.0058 | -0.007 +/- 0.017 | -0.092 +/- 0.008 |
| lstm | 0.5803 +/- 0.0088 | 0.5555 +/- 0.0189 | 0.5081 +/- 0.0031 | -0.025 +/- 0.021 | -0.072 +/- 0.009 |
| xgboost | 0.4153 +/- 0.0016 | 0.4125 +/- 0.0027 | 0.3204 +/- 0.0052 | -0.003 +/- 0.004 | -0.095 +/- 0.007 |

Set A moved held-out recall by at most 3 points, with mixed signs across models and
seeds. Set B produced a uniform loss of 27-32 points, lower in every seed. The
difference is the surviving sibling: set A left DoS floods in training. Seed
replication narrows the set A conclusion to "no effect larger than about 2 recall
points or 0.02 AP"; it does not change the set B conclusion.

Set B is not uniform across subtypes. Threshold-free average precision, cnn, per
seed under the holdout:

| Subtype | Flows per train sequence | AP per seed | Mean |
|---|---:|---|---:|
| `DoS_attacks-GoldenEye` | 5,477 | 1.000 / 1.000 / 1.000 | 1.000 |
| `DoS_attacks-SlowHTTPTest` | 3,013 | 1.000 / 1.000 / 1.000 | 1.000 |
| `DDOS_attack-HOIC` | 2,386 | 1.000 / 1.000 / 1.000 | 1.000 |
| `DDoS_attacks-LOIC-HTTP` | 701 | 0.542 / 0.547 / 0.546 | 0.545 |
| `DDOS_attack-LOIC-UDP` | 18 | 0.000 / 0.000 / 0.000 | 0.000 |

LOIC-HTTP recall is exactly 54.55% (60 of 110) in every seed, and LOIC-UDP has no
ranking signal at all in every seed, so the volume gradient is not a threshold or
initialization artifact.

### Corrections from the single-seed report

The first single-seed write-up overstated two retained-family effects. They do not
survive replication and are retracted:

- `Brute_Force_-Web` (cnn) "45.2% -> 20.5%": control per seed 45.2 / 35.5 / 24.1,
  holdout 20.5 / 41.6 / 22.9, delta -6.6 +/- 16.1 points with 2 of 3 seeds lower.
  Seed 4266 was the control's best and the holdout's worst. The drop is not
  distinguishable from retraining noise, and Web recall itself is high-variance
  across seeds.
- `Brute_Force_-XSS` at 0.1% (cnn) "14.5% -> 96.8%": control itself scores 14.5 /
  96.8 / 12.9 and holdout 96.8 / 14.5 / 12.9, mean 41.40% in both arms. A few XSS
  sequences sit on the 0.1% threshold, so it flips between seeds. It is a fragile
  metric (n=62 over 4 hosts), not a capability change.

What does survive is concentrated in xgboost: `Brute_Force_-Web` fell 36.55% ->
9.04% (delta -27.51 +/- 2.43, 3/3 seeds, AP 0.015 -> 0.004) and `Infilteration`
fell 52.99% -> 42.81% (delta -10.18 +/- 1.50, 3/3 seeds, AP 0.098 -> 0.047).
`lstm` also lost Web recall (37.55% -> 26.31%, -11.24 +/- 4.23, 3/3) with a nearly
unchanged AP, which points at a threshold shift rather than lost ranking.

Interpretation: the detector generalises to unseen attacks only when their
signature is a gross volume anomaly. High-volume floods remain detectable with no
flood exemplars; the low-rate UDP flood becomes undetectable. The set A result was
supported by surviving DoS siblings, not by learning the removed subtypes. Part of
the set B loss is a global decision-boundary shift, but it is model-dependent and
reliable only for xgboost.

## Seed replication

Single-seed differences cannot separate a holdout effect from retraining noise.
The control, set A and set B arms were therefore retrained with seeds 4266, 4267
and 4268, all other settings unchanged, and scored on the frozen test split with
each run's own validation thresholds. A seed changes weight initialization, batch
order and dropout only: the split, the frozen bucket encoder, the features and the
class weights are fixed, so the sweep bounds initialization noise, not sampling
uncertainty.

Aggregate the arms with `train/seed_report.py`:

```powershell
python train/seed_report.py --arm control=<dirs> --arm subtype=<dirs> `
  --arm flooding=<dirs> --arm-dataset subtype=datasets/dataset_holdout_xss_slowhttp_hoic `
  --arm-dataset flooding=datasets/dataset_holdout_all_flooding
```

The three-seed output is kept in `train/runs/seed_sweep_3seed/` (ignored by git).

The report pairs each holdout run with the control run of the same seed, refuses
mismatched populations or seed sets, and writes `SEED_SWEEP.md`,
`seed_sweep_arms.csv`, `seed_sweep_deltas.csv`, `seed_sweep_subtypes.csv` and
`manifest.json`. It reports mean, sample standard deviation and the raw per-seed
values. Test is scored once per seed, so repeated test use is disclosed in the
manifest.

## Limits

Small per-subtype counts, one-host DoS campaigns, single campaigns per flood
subtype and shared hosts/campaigns across folds all limit inference. Wilson
intervals describe sampling uncertainty only, and three seeds bound retraining
noise only; neither removes host or campaign dependence. Held-out recall is
detection of the label as generic malicious traffic, not subtype or family
classification. Set B removes only 3.9% of positives, yet xgboost shows a
consistent retained-family shift (Web -27.5 and Infilteration -10.2 points, 3 of 3
seeds), so part of the set B loss is a global decision-boundary change rather than
an isolated unseen-family effect. Neural retained-family changes stay inside the
seed band and should not be read as evidence either way.
