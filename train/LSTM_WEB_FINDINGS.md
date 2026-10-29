# Provisional findings: LSTM baseline and Web detection

Recorded: 2026-10-25. Status: provisional project decision and descriptive findings.

## Working decision

Retain LSTM as the main baseline for subsequent experiments. The objective is
to study the representation and temporal behavior, rather than establish a
universally best model. Keep XGBoost, CNN and Hybrid as diagnostic comparators,
especially for Web detection.

The earlier team single-seed baseline slightly favored LSTM in AP. In the
matched five-seed no-shift comparison, mean AP is 61.25% for LSTM, 61.43% for CNN
and 60.97% for Hybrid. These close results do not establish LSTM superiority.

One useful baseline property is LSTM's low seed variation at the strict global
<=0.1% validation-FPR operating point: recall is 50.73% with a sample SD of
0.40 percentage points, compared with SDs of 0.73 for XGBoost, 2.22 for CNN and
2.30 for Hybrid. This is a metric-specific stability observation on a fixed
split, not a claim that LSTM is most stable on every metric. Its Web recall,
in particular, remains variable across seeds.

## Circular shift

The paired study uses seeds 4266-4270 and rotates ten buckets to
`[5, 6, 7, 8, 9, 0, 1, 2, 3, 4]`, including all input channels and masks.
Each model is retrained with the same rotation applied to training and validation.

| Model | Overall AP change, mean +/- SD (pp) | Web recall before shift | Web recall after shift | Web recall change, mean +/- SD (pp) |
|---|---:|---:|---:|---:|
| XGBoost | -0.040 +/- 0.268 | 44.53% | 45.08% | +0.55 +/- 1.83 |
| LSTM | -0.076 +/- 0.250 | 35.58% | 29.83% | -5.75 +/- 5.48 |
| CNN | +0.116 +/- 0.839 | 30.06% | 32.82% | +2.76 +/- 5.51 |
| Hybrid | -0.002 +/- 1.044 | 32.93% | 36.02% | +3.09 +/- 4.11 |

Web recall uses each run's own global <=1% validation-FPR threshold. LSTM's
Web recall decreases in all five pairs; CNN and Hybrid gains are not consistent
across all seeds. This is a localized observation requiring explanation, rather
than sufficient grounds for replacing the main baseline.

There is no consistent overall AP benefit or harm. Recall changes depend on the
operating point: LSTM's overall recall changes by -1.70 pp at <=1% FPR and
+0.50 pp at <=0.1% FPR. Thus, "shift has no effect" is too broad a conclusion.
The study neither establishes LSTM as the most shift-resistant model nor proves
that temporal order is irrelevant. A circular shift preserves most adjacency,
and retraining measures adaptation to the rotated representation rather than
invariance of an unchanged fitted model.

## Web: strongest descriptive evidence

Under the current 100-second window with ten 10-second buckets:

- 96/181 Web sequences (53.04%) have only one active traffic bucket.
- 127/181 (70.17%) have no consecutive active traffic buckets.
- 171/181 (94.48%) contain malicious flows in only one bucket.

An active traffic bucket can contain benign or malicious traffic. More active
buckets therefore need not mean repeated attack evidence.

The no-shift five-seed results show a clear interaction with occupancy:

| Original active traffic buckets | Web sequences | LSTM mean recall | XGBoost mean recall |
|---|---:|---:|---:|
| One | 96 | 38.75% | 64.79% |
| Two | 19 | 23.16% | 12.63% |
| Three or more | 66 | 34.55% | 24.24% |

These recalls use each model's own global <=1% validation-FPR threshold.
XGBoost's aggregate Web advantage is concentrated in isolated single-bucket
events. LSTM performs better in the observed groups with multiple active
buckets. These groups contain different samples; this is not an intervention
that changes context for the same sample.

Among 19 Web sequences detected by XGBoost in >=4/5 seeds and by LSTM in <=1/5,
17 have only one active bucket. In the earlier past-context availability audit,
17 of these 19 are eligible for a common 100s-to-400s comparison; 14/17 still
have only one active bucket at 400s. In contrast, the eight stable LSTM-only
cases all have multiple active buckets, with a median of 4.5. These observations
support complementary behavior between the two models.

The lack of extra context is subgroup-specific: across all 143 common-eligible
Web anchors, median active buckets increase from one to three at 400s. Window
availability counts do not measure model recall at 400s.

## Interpretation and boundaries

The working hypothesis is that the current representation supplies little
repeated cross-bucket attack evidence for many Web positives. XGBoost may exploit
local feature thresholds more effectively for isolated events, while LSTM's
context aggregation produces different decisions. This mechanism remains
unverified; the present results do not isolate pooling, recurrence, optimization
or threshold effects.

Sparsity alone is insufficient: 54.52% of benign sequences also have only one
active bucket. Bucket content still matters. The previous PCA used learned
LSTM final-head vectors, so its dispersion cannot establish weak temporal
structure in the original traffic. A supplementary embedding-smoothness measure
also cannot rule out other forms of temporal information.

All results above concern the existing real validation cohort: 251,146 sequences,
including 917 positives and 181 Web positives. Fits reuse the same original
training cohort; seeds reuse the same validation samples. These are temporal
stress diagnostics and exploratory validation analyses, not independent held-out
test evidence or significance tests. Checkpoints and operating thresholds were
selected using validation. Historical project work had already inspected the
existing test split; these diagnostics do not score it. Subgroup selection,
small sample counts and possible host overlap limit generalization claims.

For now, retain LSTM and record the isolated-Web/XGBoost interaction as a finding.
A controlled local-feature/readout or context ablation would be needed to test
the proposed mechanism; this note does not initiate further experiments.

## Evidence

- [Paired circular-shift comparison](runs/20261024_223102_671768_circular_multiseed_start5_ef34ba/reports/20261025_003115_565779_paired_multiseed_comparison_f43f00/COMPARISON.md)
- [Seed aggregate metrics](runs/20261024_223102_671768_circular_multiseed_start5_ef34ba/reports/20261025_003115_565779_paired_multiseed_comparison_f43f00/aggregate.csv)
- [Web temporal-support interpretation](runs/20261025_013446_294043_web_temporal_support_diagnostic_7c620b/INTERPRETATION.md)
- [Web sample-level diagnostic](runs/20261025_013446_294043_web_temporal_support_diagnostic_7c620b/web_samples.csv)
- [Earlier window-context availability diagnostic](runs/20261024_124830_784309_window_context_availability_0ce90b/CONTEXT_DIAGNOSTIC.md)
