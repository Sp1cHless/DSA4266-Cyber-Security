# NF-UNSW-NB15-v3 temporal EDA

## Dataset and timestamp integrity

- **Rows / columns:** 2,365,424 / 55
- **Capture-time range:** 2015-01-22T11:49:36.907000+00:00 to 2015-02-18T12:29:24.927000+00:00
- **Calendar span:** 27.03 days
- **Attack prevalence:** 5.40%
- **Invalid start/end timestamps:** 0 / 0
- **Flows ending before they start:** 0
- **Missing cells:** 63,425 across 1 feature(s)
- **Infinite numeric cells:** 181,561 across 2 feature(s)
- **Original-order timestamp inversions:** 335,443
- **Exact duplicate rows (64-bit row hash):** 14,815
- **Duration agrees within 1 ms:** 100.00%

The original CSV row order should not be treated as chronological. All temporal analysis in this workflow sorts or groups by the real flow-start timestamp.

Missing values occur in `SRC_TO_DST_SECOND_BYTES` (63,425); infinite values occur in `SRC_TO_DST_SECOND_BYTES` (59,068), `DST_TO_SRC_SECOND_BYTES` (122,493). These non-finite values are confined to the two per-second byte-rate features in this dataset, but they must be handled using a train-fitted preprocessing rule before modeling. They should not be silently passed into a scaler or estimator.

## Capture continuity

- **Time bucket used for overview plots:** `1min`
- **Gaps exceeding one minute:** 9
- **Gaps exceeding one hour:** 1
- **Largest observed gap:** 37438.06 minutes
- **Capture sessions using a 60-minute gap rule:** 2
- **Session 1 attack prevalence:** 1.54%
- **Session 2 attack prevalence:** 8.96%

Large gaps mean the calendar range is not one continuous monitoring period. The substantial change in attack prevalence between sessions is direct evidence of temporal distribution shift. Refer to `tables/capture_sessions.csv` before defining any temporal split or interpreting uncaptured time as benign traffic.

## Candidate chronological split

- **Train:** before 2015-02-18T05:42:13.382000+00:00 (1,655,796 flows; 3.94% attacks)
- **Validation:** until 2015-02-18T09:08:16.106000+00:00 (354,813 flows; 8.85% attacks)
- **Test:** thereafter (354,815 flows; 8.77% attacks)

Classes absent from each split:

- **train:** none
- **validation:** none
- **test:** none

This split is a feasibility audit, not a finalized modeling decision. If rare attack classes are absent or concentrated in only one period, the team must decide whether to evaluate binary detection, redesign boundaries around capture sessions, or explicitly study unseen-class generalization.

## Plot-by-plot findings

## `class_distribution.png`

Benign traffic accounts for 94.60% of all flows. Exploits and Fuzzers are the largest attack classes, while Worms has only 158 observations. The extreme imbalance means accuracy alone would be misleading; per-class recall, macro-F1, PR-AUC, and the confusion matrix should be reported.

## `traffic_and_attack_rate_over_time.png`

Traffic comes from two roughly 12-hour capture sessions separated by a 26.00-day gap. Attack prevalence rises from 1.54% in session 1 to 8.96% in session 2. The gap must not be interpolated as zero traffic, and this session shift makes a purely random split liable to overstate generalization.

## `attack_categories_over_time.png`

Attack activity is concentrated in temporal bursts rather than being uniformly distributed. Several attack categories are active in nearby time buckets, so randomly placing neighboring flows into both training and testing can leak capture-specific patterns. A chronological or session-aware evaluation is therefore more defensible.

## `duration_and_iat_ecdf.png`

In the reproducible sample, the median flow duration is 28.0 ms for benign traffic and 530.5 ms for attacks. Median source-to-destination IAT is 0.0 versus 50.0, and destination-to-source IAT is 0.0 versus 59.5. These temporal features contain useful separation, but the overlapping ECDFs show that none should be treated as a standalone attack rule.

## `chronological_split_class_composition.png`

All attack classes appear in the candidate train, validation, and test periods, but prevalence shifts from 3.94% in training to 8.85% and 8.77% in validation and test. Worms remains especially sparse (83/38/37 flows), so its class-specific metric will be unstable and should be interpreted with counts or confidence intervals.

## `chronological_split_distribution_shift.png`

The train-to-test Jensen–Shannon divergence is 0.0074 across all traffic and 0.0036 among attack classes. Reporting both prevents the dominant benign class from hiding changes in the attack mixture. Validation and test are very similar overall (0.0012), although their attack mixtures differ more (0.0140), mainly because the Fuzzers share changes by 4.38 percentage points. These values quantify dataset shift; they are descriptive diagnostics rather than universal pass/fail thresholds.

## Files to review before Wednesday

1. `plots/class_distribution.png`
2. `plots/traffic_and_attack_rate_over_time.png`
3. `plots/attack_categories_over_time.png`
4. `plots/duration_and_iat_ecdf.png`
5. `plots/chronological_split_class_composition.png`
6. `plots/chronological_split_distribution_shift.png`
7. `tables/capture_sessions.csv`
8. `tables/chronological_split_class_counts.csv`
9. `tables/chronological_split_distribution_shift.csv`
10. `dataset_summary.json`

No scaling, balancing, feature selection, model fitting, or synthetic timestamp construction is performed here.
