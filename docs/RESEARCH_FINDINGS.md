# Research findings: superimposed traffic and Scheme2

Prepared for repository sharing on 29 October 2026. Research snapshot: 27 October
2026. This summary preserves selected completed conclusions; no experiments were
rerun while preparing it. Full datasets, checkpoints, predictions, local evidence
inventories and maintenance records remain on the research machines.

## Research progression

The first professor-feedback stage led to an event-level design: align source
traffic sequences, combine their original flow records, then aggregate buckets.
Four full-source EDA studies established observation perspective, sequence scale,
feature support and split feasibility. The second feedback specified sequence
construction, bucket aggregation, split and training-only superposition, with
embedding addition as the representation direction. Scheme1, Scheme1 with L2 and
Scheme2 were then compared, and Scheme2 became the shared pipeline.

The initial event-union export has 228 features per bucket; the later Scheme1
comparison uses 137; Scheme2 uses 64 learned coordinates plus an intensity
channel. These are separate representation versions.

## Data and construction findings

- The descriptive source contains 20,115,529 flows, including 2,600,903
  attack-labeled flows. Source grouping is the adopted per-source observation
  unit; raw source identities remain provenance rather than detector inputs.
- Across 50/100/200/500/1000-second descriptive windows, median active-bucket
  count is one. Longer windows add context for some traffic while reducing
  overall occupancy. The adopted initial grid is 100 seconds with ten 10-second
  buckets, retaining internal empty positions.
- Of 5,210 natural hybrid 100-second windows, 4,562 contain Infilteration
  (87.56%). Observed DoS/DDoS/FTP-BruteForce/SSH-Bruteforce have no natural hybrid
  windows at the tested lengths. Background composition is therefore a
  type-dependent research variable.
- The raw-feature audit supports a CIC-specific vocabulary and explicit
  aggregation rules. The old L7 vocabulary groups 39.64% of flows into `other`;
  twelve legacy feature slots are identically zero in this source.
- Eight split candidates led to the adopted global 15-minute block policy,
  coverage reservations, 100-second guards and actual-flow-end exclusions.
  Retained real train/validation/test counts are 765,652/251,146/263,125.
  This evaluates held-out time blocks in the same environment; hosts and
  campaigns can recur across splits.

Scheme2 retains all real training originals and adds 8,093 benign+attack and
8,093 benign+benign examples from different-source real training donors.
The resulting training population is 781,838 sequences. Donor reuse creates
background variants of recorded attacks, not additional independent attacks.
Validation and test retain real samples at their natural prevalence.

Construction and representation definitions are in the
[Scheme2 protocol](../CIC_new/SCHEME2_DATASET.md) and
[input-encoder guide](../CIC_new/representation_learning/README.md).

## Representation comparison and completed real test

Bucket L2 improves Scheme1 validation AP from 3.44% to 10.40% in the temporal
AE plus train-fitted linear-readout experiment. The largest-error 0.1% of
validation sequences contribute 41.66% of reconstruction error before L2 and
0.80% afterward. Amplitude control is a measured contributor to the improvement.

| Historical frozen real-test result | Scheme1 + L2 | Scheme2 |
|---|---:|---:|
| AP | 7.27% | 14.93% |
| ROC-AUC | 0.82768 | 0.84948 |
| Recall using the validation-FPR-1% threshold | 31.02% | 40.54% |
| Actual test FPR at that threshold | 1.03% | 0.99% |

Scheme2's overall validation/test performance supports its selection as the
common pipeline. Controlled dilution with reused training donors additionally
shows that encoding physically merged flows and adding separate embeddings
produce different responses. These synthetic stress results describe selected
weak traffic fragments and are distinct from the real-test table.

## Supervised detector baseline

The four-family screen uses the same frozen Scheme2 data, four-group objective
weights, validation AP selection and global low-FPR thresholds.

| Historical seed-4266 model | Validation AP | Recall at <=1% FPR | Recall at <=0.1% FPR |
|---|---:|---:|---:|
| LSTM | 61.79% | 76.34% | 50.49% |
| CNN + LSTM | 60.86% | 72.96% | 51.04% |
| CNN | 60.18% | 74.26% | 49.07% |
| XGBoost | 50.52% | 69.57% | 37.62% |

LSTM is the working baseline. In the matched five-seed no-shift comparison,
mean AP is 61.25% for LSTM, 61.43% for CNN and 60.97% for Hybrid, showing close
neural-family performance on the fixed cohort. These supervised validation
results are a different experiment from the historical AE/readout test above.
See the [training and evaluation guide](../train/README.md).

## Temporal and attack-specific findings

The circular comparison covers 40 completed fits across four families, five
seeds and two orders: 36 new fits and four reused shifted fits. Mean AP changes
are -0.040/-0.076/+0.116/-0.002 percentage points for
XGBoost/LSTM/CNN/Hybrid. Attack-specific responses are more pronounced: LSTM
Web recall at the global 1% FPR budget changes from 35.58% to 29.83%.

Web often supplies isolated local evidence: 96/181 validation examples have one
active traffic bucket, and 171/181 contain attack flows in only one bucket.
XGBoost's five-seed recall on the single-active-bucket group is 64.79%, versus
38.75% for LSTM; LSTM is stronger in the observed multi-bucket groups. This
supports complementary local and contextual behavior. See
[the detailed Web findings](../train/LSTM_WEB_FINDINGS.md).

The 200s/10s, 200s/20s, 400s/10s and 400s/40s LSTM experiments show that temporal
scale changes both behavior coverage and detection sensitivity. Web recall at
1% reaches 58.12% for 400s/40s on that rebuilt cohort. Cohort sizes and prevalence
differ by duration, so this is a scale-specific observation rather than a paired
effect on identical examples. Frozen context interventions also show that
surrounding benign activity can contribute to a learned attack decision.

Infilteration contains distinct difficulty profiles. Ten of 34 inspected pure
misses have an identical benign counterpart at both pre-encoder aggregate and
Scheme2-input stages. Other cases remain recoverable by different model
families. Later flow-level analysis distinguishes TCP/3389 minority-attack
mixing sensitivity from attack-heavy UDP/53 response changes. These findings
locate concrete representation tradeoffs and different research targets.

## Targeted learning and information experiments

The following runs use the same-runtime chronological seed-4266 reference,
whose AP is 61.5277%. This reference differs from the historical team LSTM.

| Variant | Validation AP | Infilteration recall: 1% / 0.1% FPR | Web recall: 1% / 0.1% FPR |
|---|---:|---:|---:|
| Same-runtime reference | 61.5277% | 72.95% / 36.61% | 25.97% / 3.87% |
| Learned attention pooling | 61.87% | 70.49% / 35.79% | 29.28% / 2.21% |
| Fifteen added distribution statistics | 61.4998% | 70.77% / 31.69% | 36.46% / 6.08% |
| Bucket auxiliary supervision, lambda=0.5 | 61.6398% | 74.04% / 36.07% | 26.52% / 3.87% |

Attention learns nonuniform attack-position weighting. Added distribution
statistics change the Web/Infilteration performance tradeoff despite nearly
equal overall AP; this experiment includes a refitted bucket encoder.
Bucket supervision produces a learnable localization signal: among 135
Infilteration examples with both attack and benign active positions, the
highest-scoring bucket hits an attack position in 116 cases (85.9%), compared
with 54.4% expected under uniform active-bucket selection. Localization and
low-FPR sequence discrimination are distinct measured outcomes.

These are single-new-seed exploratory comparisons on reused training and
inspected validation cohorts. Protocols are documented in the
[attention](../train/LSTM_CONTEXT_ABLATION.md),
[distribution-feature](../train/DISTRIBUTION_FEATURE_ABLATION.md) and
[bucket-supervision](../train/LSTM_BUCKET_SUPERVISION.md) guides.

## Fixed subtype holdout and representation analysis

One fixed configuration removes XSS, HOIC and SlowHTTPTest primary labels
together from training; it is not a rotation through all subtype choices.
The training population becomes 781,190, including 7,877 real positives.
The new LSTM has validation AP 60.3058% versus historical 61.7917%.

Matched visualization uses the same 4,917 validation IDs. Native 64D 20-neighbor
benign fractions stay close for retained Web (23.90% -> 24.48%) and
Infilteration (26.95% -> 27.16%), while selected attack-to-attack neighborhoods
change. Bot's HOIC-neighbor fraction rises to 7.95%; SQL's XSS-neighbor fraction
rises to 18.67% in its 15-example slice. This describes representation changes
between pipelines whose scaler, encoder, donor plan and historical runtime also
differ. Validation includes withheld types for model/threshold selection.

## Evidence retention

Detailed reports, sample-level tables, vector exports, predictions and model
artifacts remain in ignored local run/archive folders. The local
`docs/research_map_20261027/` directory contains the full research tree,
33 evidence groups, a 106-file path/hash index and discussion references.
Some earlier Scheme1 generated datasets and weights were deliberately retired;
their result reports, recorded configurations and small implementation files
remain in the local archive.

The historical Scheme2 test was already inspected. Later exploratory validation,
training-derived probes and oracle stress diagnostics retain those scopes;
future test-informed development requires a fresh holdout for a new independent
final-performance claim. This summary reports saved findings, not new results.
