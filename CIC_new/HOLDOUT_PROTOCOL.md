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

Control versus holdout, frozen test split, each model scored with its own saved
validation thresholds. Values are recall at the validation 1% FPR budget.

| Model | Set A pooled (n=134) | Set B pooled (n=202) | Set B DDoS (n=189) |
|---|---|---|---|
| cnn | 100.0% -> 99.3% | 100.0% -> 68.3% | 100.0% -> 66.1% |
| hybrid | 99.3% -> 97.8% | 100.0% -> 68.3% | 100.0% -> 66.1% |
| lstm | 99.3% -> 98.5% | 100.0% -> 75.3% | 100.0% -> 73.5% |
| xgboost | 96.3% -> 99.3% | 96.5% -> 68.8% | 96.3% -> 66.7% |

Set A moved held-out recall by at most 6.5 points, inside the intervals, while
seen subtypes drifted by up to 32 points. Set B produced a real, uniform loss of
25-32 points. The difference is the surviving sibling: set A left DoS floods in
training.

Set B is not uniform across subtypes. Threshold-free average precision per
subtype, cnn, control -> holdout:

| Subtype | Flows per train sequence | AP control -> holdout |
|---|---:|---|
| `DoS_attacks-GoldenEye` | 5,477 | 1.000 -> 1.000 |
| `DoS_attacks-SlowHTTPTest` | 3,013 | 1.000 -> 1.000 |
| `DDOS_attack-HOIC` | 2,386 | 1.000 -> 1.000 |
| `DDoS_attacks-LOIC-HTTP` | 701 | 0.954 -> 0.542 |
| `DDOS_attack-LOIC-UDP` | 18 | 1.000 -> 0.000 |

The retained families also moved: `Brute_Force_-Web` fell 45.2% -> 20.5% (cnn) and
36.8% -> 9.6% (xgboost) at 1% FPR, and XSS rose from 14.5% to 96.8% at the 0.1%
budget with a near-unchanged threshold (0.9869 -> 0.9891). Detection is therefore
relative to the training mixture, not an absolute property of the attack.

Interpretation: the detector generalises to unseen attacks only when their
signature is a gross volume anomaly. High-volume floods remain detectable with no
flood exemplars; the low-rate UDP flood becomes undetectable. The set A result was
supported by surviving DoS siblings, not by learning the removed subtypes.

## Limits

Small per-subtype counts, one-host DoS campaigns, single campaigns per flood
subtype and shared hosts/campaigns across folds all limit inference. Wilson
intervals describe sampling uncertainty only. Held-out recall is detection of the
label as generic malicious traffic, not subtype or family classification. Set B
removes only 3.9% of positives, yet retained families still moved by up to 27
points, so part of the set B loss is a global decision-boundary shift rather than
an isolated unseen-family effect.
