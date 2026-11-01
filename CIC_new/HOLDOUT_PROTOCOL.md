# Training-holdout protocol: unseen attack subtypes and families

This protocol trains the Scheme2 detector without selected attack labels, then
measures whether those labels are still flagged. The detector stays binary; a
holdout changes only which real training sequences and fitted transforms see the
labels. Three holdout sets have been built and run with the same protocol: one
subtype-level set and two whole-family sets.

## Holdout sets

### A. One subtype per multi-subtype family

The published set. Three multi-subtype families each contribute one held-out label,
and `SSH-Bruteforce` is added as a fourth arm even though it is a single-member
family, so that arm is a whole-family removal nested inside a subtype-level set.
`Bot` and `Infilteration` are also single-member families and stay in training. The
earlier three-family build and its four-model results are archived in
`CIC_new/archive/HOLDOUT_PROTOCOL_legacy_3subtype.md`.

| Family | Held out from training | Retained for training | Removed train sequences | Held-out-type flows removed | Validation | Test |
|---|---|---|---:|---:|---:|---:|
| Web | `Brute_Force_-XSS` | `Brute_Force_-Web`, `SQL_Injection` | 122 | 136 | 37 | 62 |
| DoS | `DoS_attacks-SlowHTTPTest` | GoldenEye, Hulk, Slowloris | 13 | 39,170 | 7 | 7 |
| DDoS | `DDOS_attack-HOIC` | LOIC-HTTP, LOIC-UDP | 81 | 193,295 | 94 | 65 |
| SSH (single-member) | `SSH-Bruteforce` | none | 30 | 103,548 | 10 | 6 |
| **Total** | | | **246** | **336,149** | **148** | **140** |

### B. Whole-family holdouts

#### B-1. All flooding (DoS and DDoS together)

| Held out from training | Removed train sequences | Held-out-type flows removed | Validation | Test |
|---|---:|---:|---:|---:|
| `DDOS_attack-HOIC` | 81 | 193,295 | 94 | 65 |
| `DDoS_attacks-LOIC-HTTP` | 141 | 98,859 | 64 | 110 |
| `DDOS_attack-LOIC-UDP` | 60 | 1,094 | 54 | 14 |
| `DoS_attacks-SlowHTTPTest` | 13 | 39,170 | 7 | 7 |
| `DoS_attacks-GoldenEye` | 6 | 32,860 | 2 | 6 |
| `DoS_attacks-Hulk` | 6 | 75,640 | 0 | 0 |
| `DoS_attacks-Slowloris` | 6 | 8,914 | 16 | 0 |

#### B-2. Whole Web family

Every web-facing label is removed together, including `SQL_Injection`, so no web
sibling survives to support the others. This is the arm that disambiguates the Set A
XSS result.

| Held out from training | Removed train sequences | Held-out-type flows removed | Validation | Test |
|---|---:|---:|---:|---:|
| `Brute_Force_-Web` | 841 | 1,038 | 181 | 166 |
| `Brute_Force_-XSS` | 253 | 270 | 37 | 62 |
| `SQL_Injection` | 180 | 217 | 15 | 25 |
| **Total** | **1,032** | **1,525** | **233** | **253** |

Counts are real sequences in the control Scheme2 dataset and were recorded before
any scoring. The validation and test columns are primary-label counts taken from the
control dataset; the build audit's `*_sequences_primary` field is not comparable,
because it counts sequences that contain the type and whose primary label is any
held-out type, which inflates the Web totals to 320.

`Bot` and `Infilteration` stay in training in every arm. `FTP-BruteForce` also stays
in training in every arm, including Set A where `SSH-Bruteforce` is removed.

Sequence counts and flow volume point in opposite directions. Set A removes 246 of
8,093 training positives (3.0%) but 336,149 held-out-type flows, about 31.6% of the
roughly 1.065M attack flows in the control training fold, because HOIC and SSH
sequences are high-volume. B-2 removes 1,032 positives (12.8%) but only 1,525
held-out-type flows, about 0.1%, because web attacks are many short sequences. B-1
removes 313 positives (3.9%).

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
records `train_holdout_attacks`. Pass that directory itself to `--dataset`
(`--dataset <out>/dataset`): the workflow root has no `manifest.json`, and nothing
needs to be copied into `datasets/`.
`workflow.json` and `preprocessed/split_notes.json` carry the per-type audit,
including how many held-out flows survive as non-primary members of retained
training sequences (sequence-level exclusion, not flow-level). `split_notes.json`
separates primary-label counts from membership counts.

Holdout label strings for the published arms:

- Set A (4 families):
  `Brute_Force_-XSS,DoS_attacks-SlowHTTPTest,DDOS_attack-HOIC,SSH-Bruteforce`
- Set B-1 (all flooding):
  `DoS_attacks-GoldenEye,DoS_attacks-Hulk,DoS_attacks-Slowloris,DoS_attacks-SlowHTTPTest,DDOS_attack-HOIC,DDoS_attacks-LOIC-HTTP,DDOS_attack-LOIC-UDP`
- Set B-2 (Web family): `Brute_Force_-Web,Brute_Force_-XSS,SQL_Injection`

Builds: Set A `CIC_new/outputs/scheme2_runs/20261030_holdout_seta4_xss_slowhttp_hoic_ssh`
(02:03-03:18) and B-2 `CIC_new/outputs/scheme2_runs/20261030_holdout_web_family`
(03:18-04:22), both on 2026-10-30; B-1 uses
`CIC_new/outputs/scheme2_runs/20261026_holdout_all_flooding`. Set A and B-2 were
rebuilt from scratch: sequence construction, fold assignment and purge identical to
control, but the vocabulary, numeric scaler, frozen bucket encoder and donor plan
refit without the held-out labels. Validation and test counts equal control
(251,146 / 917 and 263,125 / 851).

## Checks before training

- `workflow.json` status `complete`; `dataset/validation.json` `passed: true`.
- No training row has a held-out `primary_attack`; validation/test per-subtype
  counts equal the control dataset; training counts equal control minus the
  removed sequences; `P` equals the reduced real malicious training count.
- Set A: `P = 7847`. Set B-1: `P = 7780`. Set B-2: `P = 7061`.
- Shard checksums differ from the control dataset; manifest fingerprints differ.
- Both rebuilt datasets were compared element-wise against control on the test split:
  `sample_id`, `y`, `primary_attack` and `source_host` are identical, and the schema
  hash matches control.

## Report

```powershell
python train/unseen_subtype_report.py `
  --control train/lstm/runs/seed_sweep_control_seed4266 `
  --holdout train/lstm/runs/seed_sweep_subtype4_seed4266 `
  --holdout-dataset CIC_new/outputs/scheme2_runs/20261030_holdout_seta4_xss_slowhttp_hoic_ssh/dataset
```

One control run and one holdout run are enough; the published numbers come from the
three-seed aggregation described under Seed replication, not from this command.

The report refuses mismatched prediction populations, so it also verifies that the
split did not move. It writes `UNSEEN_SUBTYPE.md`, `unseen_subtype.csv` and
`manifest.json` in a fresh run directory: per-subtype recall at the 1% and 0.1%
false-positive budgets for control and holdout, deltas, Wilson intervals, pooled
held-out recall and overall AP. `train/compare.py` is still used per dataset; it
refuses to mix the two dataset fingerprints.

## Frozen test outcome

LSTM only. Control versus each holdout on the frozen test split, every run scored
with its own saved validation thresholds. All arms were trained with seeds 4266, 4267
and 4268 and paired by seed against the control run of the same seed; values are mean
+/- sample standard deviation over the three seeds, and every delta is the paired
holdout-minus-control difference.

| Arm | Held out | Test sequences (hosts) | Pooled R@1% | Delta | Seeds lower |
|---|---|---:|---|---|---:|
| Set A (4 families) | XSS, SlowHTTPTest, HOIC, SSH | 140 (15) | 99.29% -> 98.57% +/- 1.24 | -0.71 +/- 1.24 | 2/3 |
| Set B-1 (all flooding) | 7 DoS and DDoS flood subtypes | 202 (12) | 99.50% -> 72.94% +/- 4.00 | -26.57 +/- 3.14 | 3/3 |
| Set B-2 (Web family) | Web BF, XSS, SQL | 253 (116) | 51.65% -> 12.12% +/- 1.60 | -39.53 +/- 5.74 | 3/3 |

Overall test average precision:

| Arm | Control | Holdout | Delta AP |
|---|---|---|---|
| Set A (4 families) | 0.5803 +/- 0.0088 | 0.5716 +/- 0.0140 | -0.0087 +/- 0.0192 |
| Set B-1 (all flooding) | 0.5803 +/- 0.0088 | 0.5081 +/- 0.0031 | -0.072 +/- 0.009 |
| Set B-2 (Web family) | 0.5803 +/- 0.0088 | 0.5469 +/- 0.0070 | -0.0334 +/- 0.0156 |

At the 0.1% budget, pooled held-out recall went 66.19% -> 73.57% +/- 21.03 under Set A,
96.20% -> 68.65% +/- 0.57 under B-1, and 7.25% -> 3.16% +/- 0.99 under B-2.

### Where the B-1 loss comes from

B-1 loses 26.6 points, but not evenly. Threshold-free average precision per held-out
subtype, per seed:

| Subtype | Flows per train sequence | AP per seed | Mean |
|---|---:|---|---:|
| `DoS_attacks-GoldenEye` | 5,477 | 1.000 / 1.000 / 1.000 | 1.000 |
| `DoS_attacks-SlowHTTPTest` | 3,013 | 1.000 / 1.000 / 1.000 | 1.000 |
| `DDOS_attack-HOIC` | 2,386 | 1.000 / 1.000 / 1.000 | 1.000 |
| `DDoS_attacks-LOIC-HTTP` | 701 | 0.550 / 0.547 / 0.547 | 0.548 |
| `DDOS_attack-LOIC-UDP` | 18 | 0.005 / 0.010 / 0.001 | 0.006 |

The pooled figure is fully accounted for by the two lowest-volume subtypes. GoldenEye,
SlowHTTPTest and HOIC hold 100% recall in every seed (78 of 202 sequences) and
LOIC-HTTP exactly 54.55% (60 of 110); LOIC-UDP catches 14, 14 and 0 of its 14, which
reproduces 147.3 of 202 sequences, or 72.94%.

LOIC-UDP has essentially no ranking signal (AP at most 0.010) even though its
thresholded recall is 100% in two seeds: it trains on 18 flows per sequence against
701 for LOIC-HTTP and thousands for the three at ceiling, so whether it clears the
threshold is decided by the seed. Average precision is threshold-free, so the gradient
itself is not an artifact of the operating point; the recall figure for the smallest
subtype is, which is why it is reported with its spread rather than quoted.

### Per-subtype outcome under Set A and B-2

Per held-out subtype at the 1% false-positive budget, paired by seed:

| Subtype | Set A | Delta | Set B-2 | Delta |
|---|---|---|---|---|
| `Brute_Force_-XSS` (n=62, 4 hosts) | 98.39% -> 96.77% | -1.61 +/- 2.79 | 98.39% -> 16.13% | -82.26 +/- 0.00 |
| `Brute_Force_-Web` (n=166, 114 hosts) | retained: 37.55% -> 35.14% | -2.41 +/- 13.62 | 37.55% -> 12.45% | -25.10 +/- 10.03 |
| `SQL_Injection` (n=25, 1 host) | retained: 29.33% -> 81.33% | +52.00 +/- 38.16 | 29.33% -> 0.00% | -29.33 +/- 16.17 |
| `DDOS_attack-HOIC` (n=65, 10 hosts) | 100.00% -> 100.00% | 0.00 +/- 0.00 | retained: 100.00% -> 100.00% | 0.00 +/- 0.00 |
| `DoS_attacks-SlowHTTPTest` (n=7, 1 host) | 100.00% -> 100.00% | 0.00 +/- 0.00 | retained: 100.00% -> 100.00% | 0.00 +/- 0.00 |
| `SSH-Bruteforce` (n=6, 1 host) | 100.00% -> 100.00% | 0.00 +/- 0.00 | retained: 100.00% -> 100.00% | 0.00 +/- 0.00 |

Together, Set A and B-2 answer a question a subtype-level holdout alone cannot.
`Brute_Force_-XSS`
survives Set A at 96.8% but collapses to 16.1% in B-2, where its web siblings are
removed as well. The Set A XSS result was therefore carried by the retained
`Brute_Force_-Web` and `SQL_Injection` exemplars, not by an XSS-invariant signature.
The same mechanism explains B-1: what generalises is a gross volume anomaly, and a
family is only unseen in the sense that matters once its siblings are removed too.
B-2 is the strongest counter-example to a "detects novel attacks" reading, because
its removal is small in flow volume yet the largest in recall.

Retained families were stable under B-2 (`Infilteration` 74.65% -> 83.43%, 0/3 seeds
lower; `DDoS_attacks-LOIC-HTTP` +0.91 points at 1% and +6.97 at 0.1%), so the loss is
localised to the held-out web labels rather than a global decision-boundary collapse.
`SQL_Injection` rising to 81.33% under Set A rests on 25 sequences from one host and
should not be read as an improvement.

### Superseded material

Two things are archived in `CIC_new/archive/HOLDOUT_PROTOCOL_legacy_3subtype.md`: the
earlier three-family subtype arm with its single-seed retractions and interpretation,
and the cnn, hybrid and xgboost rows of the B-1 tables. Neither is part of the
published results. B-1 itself still is, on LSTM only.

## Seed replication

Single-seed differences cannot separate a holdout effect from retraining noise.
Every arm reported here was trained with seeds 4266, 4267 and 4268, all other
settings unchanged, and scored on the frozen test split with each run's own
validation thresholds. A seed changes weight initialization, batch order and dropout
only: the split, the frozen bucket encoder, the features and the class weights are
fixed, so the sweep bounds initialization noise, not sampling uncertainty.

Aggregate the two new arms with `train/seed_report.py`, against three control runs
(seeds 4266, 4267 and 4268):

```powershell
python train/seed_report.py `
  --arm control=train/lstm/runs/seed_sweep_control_seed4266,train/lstm/runs/seed_sweep_control_seed4267,train/lstm/runs/seed_sweep_control_seed4268 `
  --arm subtype4=<three runs> --arm webfamily=<three runs> `
  --arm-dataset subtype4=CIC_new/outputs/scheme2_runs/20261030_holdout_seta4_xss_slowhttp_hoic_ssh/dataset `
  --arm-dataset webfamily=CIC_new/outputs/scheme2_runs/20261030_holdout_web_family/dataset `
  --out train/runs/seed_sweep_seta4_webfamily
```

Set B-1's LSTM rows come from `train/runs/seed_sweep_3seed/` (ignored by git), which
also holds the cnn, hybrid and xgboost rows that are now archived. Its subtype arm is
the archived three-family build.

Each run is evaluated with `train/evaluate.py --run <run> --split test --out
<run>/test_eval`. The explicit `--out` matters: the default target is
`<run>/evaluations/<timestamp>_test_<hash>`, which `seed_report.py` does not
recognise.

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
classification.

Every number here comes from one architecture. The seed sweep bounds initialization
noise for LSTM only, so nothing in this protocol speaks to cnn, hybrid or xgboost, and
the earlier four-model B-1 comparison is archived rather than extended. That costs one
check: B-1 removes only 3.9% of positives, and a global decision-boundary shift would
look the same as an unseen-family effect in the pooled number. The archived xgboost
run showed exactly that pattern (Web -27.5 and Infilteration -10.2 points, 3 of 3
seeds), so part of the B-1 loss may be a boundary change; with LSTM alone it cannot be
separated out.

`SSH-Bruteforce` contributes 6 test sequences from one host and
`DoS_attacks-SlowHTTPTest` 7 from one host: both are at ceiling under control and
holdout, so they add a held-out family without adding discriminating power.
`SQL_Injection` has 25 test sequences from one host and swings from 29.33% to 81.33%
under Set A, which is a threshold artifact. `Brute_Force_-Web` spans 114 hosts but only
166 sequences and its control recall is 37.55% +/- 11.18, so the -25.10 point B-2 drop
is reliable in sign (3/3 seeds) while its magnitude is host-dependent. Finally, all
three arms were built after the archived results were known, so the choice of arms is
informed by earlier outcomes even though the builds and the seeds are fixed and the
test split was not re-tuned.
