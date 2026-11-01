# Historical research tree: superimposed traffic and temporal attack detection

Snapshot from 29 October 2026. For the current report-aligned organization, including the completed host study, start with [the research index](../README.md). Organization follows the research logic of the supplied
handwritten sketch: design decisions -> representation comparison -> common
Scheme2 detector study -> temporal, diagnostic and transfer branches.

- [Complete vector tree and detailed result pages](research_tree_complete.pdf)
- [Zoomable complete tree](research_tree_complete.svg)
- [Complete tree image](research_tree_complete.png)
- [Editable Graphviz source](research_tree.dot)

The PDF contains an overview, the full tree, and five detailed pages covering
design/representation, models/time, difficult attacks/interventions, holdout
transfer, and evidence/clarifications. Final project artifacts are in English.

## Research hierarchy

```text
Detect attacks within benign background using local traffic sequences
|
+-- 1. Establish the data and composition design [E01-E05]
|   +-- Observation perspective: source / destination / endpoint / network
|   |   `-- Adopt source-IP behavior; preserve source identity as provenance
|   +-- Scale: compare 50 / 100 / 200 / 500 / 1000s sequences
|   |   `-- Adopt 100s / ten 10s buckets; retain genuine internal sparsity
|   +-- Natural background: hybrid composition varies by attack type
|   |   `-- 4562/5210 hybrid 100s windows contain Infilteration: 87.56%
|   +-- Feature support: 55 raw fields; audit the historical 88 features
|   |   `-- CIC vocabulary, supported statistics and explicit aggregation
|   +-- Split: eight candidates -> protected global 15min blocks
|   |   `-- Coverage reservations, 100s guards and actual-flow-end exclusions
|   `-- Superimposition contract
|       `-- Retain originals; add P B+A and P B+B from real training donors
|
+-- 2. Compare representation and composition choices [E06-E12]
|   +-- Event-union reference: align original flows, merge, reaggregate (228D)
|   +-- Scheme1: exact statistics + field-level category representations (137D)
|   |   `-- Add complete-bucket L2: validation AP 3.44% -> 10.40%
|   +-- Scheme2: frozen whole-bucket vector (64D) + scaled intensity (1D)
|   |   `-- Normalize unit-donor vector sums; preserve total flow count separately
|   +-- Natural subgroup analysis and controlled physical/latent mixtures
|   |   `-- Composition operations have measurably different responses
|   `-- Common detection comparison -> select Scheme2
|       `-- AE/readout real-test AP 14.93%, versus Scheme1+L2 7.27%
|
`-- 3. Study detectors on the shared Scheme2 input [E13]
    +-- Four families: XGBoost / LSTM / temporal CNN / CNN+LSTM
    |   `-- Common data, four-group objective, AP and global FPR budgets
    +-- Working baseline: retain LSTM [E22-E23]
    |   +-- Initial single-seed AP lead; five-seed neural means remain close
    |   `-- Low observed seed SD at the strict-FPR recall operating point
    |
    +-- A. Temporal sensitivity [E20-E26]
    |   +-- Circular rotation by five buckets; channels and masks move together
    |   |   `-- Small overall AP changes; localized type/threshold responses
    |   +-- Past-context availability on common eligible anchors
    |   |   `-- Some traffic gains history; hard isolated Web cases can stay sparse
    |   `-- 200s/10s, 200s/20s, 400s/10s and 400s/40s detector fits
    |       `-- Scale changes context, coverage and type-specific sensitivity
    |
    +-- B. Explain attack differences [E14-E19,E23,E29]
    |   +-- Projector + native-dimensional neighbors + TP/FN and host slices
    |   +-- Web: attack evidence is often isolated in one bucket
    |   |   `-- Local and recurrent models show complementary behavior
    |   +-- Infilteration: observable ambiguity and architecture-dependent recovery
    |   |   +-- Exact benign counterparts already exist before neural encoding
    |   |   +-- Host composition changes the descriptive average
    |   |   `-- TCP/3389 mixing sensitivity differs from attack-heavy UDP/53
    |   `-- Targeted experiments connect diagnosis to learning [E25-E30]
    |       +-- Oracle context interventions -> learned attention weighting
    |       |   `-- Position emphasis and sequence discrimination are distinct
    |       +-- Train-only feature probes -> 15-statistic extension
    |       |   `-- Similar AP conceals a Web/Infilteration tradeoff
    |       `-- Existing bucket labels -> auxiliary temporal supervision
    |           `-- Learnable localization: 116/135 = 85.9% top-bucket hit
    |
    `-- C. Four-family training-holdout transfer [E31-E35]
        +-- A: remove XSS + SlowHTTPTest + HOIC together; retain related types
        |   `-- Three-seed frozen-test pooled recall stays near 99%
        +-- B: remove all seven DoS and DDoS primary labels together
        |   +-- Pooled recall falls by 26.57-31.68 percentage points
        |   `-- Transfer remains traffic-pattern-dependent; high-volume types easier
        +-- Retained-family effects: distinguish ranking from threshold changes
        +-- Fixed-ID embedding comparison after the earlier set-A LSTM fit
        |   `-- Similar retained-type benign mixing; changed attack neighborhoods
        `-- Scope: primary-label training exclusion; validation retains held-out types
```

## Clarifications and additions to the handwritten sketch

| Sketch item | Completed statement in the tree |
|---|---|
| Initial event-level superimposition | Align raw flow sequences, combine records, then recompute bucket statistics. The completed original export is 228D, separate from the later 137D Scheme1. |
| Natural hybrid percentage | At 100s, 4562/5210 hybrid windows contain Infilteration, giving 87.56%. This is an all-membership count, not an exclusive primary-label count. |
| Definition of P | Number of retained real attack-bearing training sequences, including natural hybrids; not the number of attack flows. |
| Purpose of 15-minute blocks | Grouped assignment, coverage and temporal isolation. Internal empty buckets remain; sparsity is measured separately. |
| L2 normalization | Controls a vector's amplitude while preserving relative coordinates. The Scheme1 L2 experiment measures a real gain; it is not per-coordinate variance equalization. |
| Scheme2 flow-count weighting | Active unit donor embeddings enter with equal coefficients, and the combined count is appended separately. The numerical branch can still encode intensity-associated information. |
| Reason to retain LSTM | Initial AP lead, close five-seed neural results, and low variation specifically for strict-FPR recall on the fixed cohort. Architecture suitability is not itself evidence of learned temporal semantics. |
| Circular shift result | Overall AP is relatively stable under the tested refitted rotation, with important type/threshold-specific changes. Rotation preserves most adjacency. |
| Web sparsity | Local attack evidence and readout behavior are concrete research findings. The data can still contain useful content within its isolated active bucket. |
| Infilteration | Separate exact aggregate ambiguity, host composition, minority-flow mixing and attack-heavy representation response. |
| Leave-one-out | One fixed simultaneous subtype-removal set, plus one all-flooding removal set, repeated over three detector seeds. This is not a completed rotation over every subtype. |
| Generalization verdict | Related-type transfer remains high in set A; set B has a consistent pooled loss while retaining substantial detection and marked subtype differences. |

Added branches include the raw-feature audit, donor hierarchy/provenance,
normalization diagnosis, controlled mixture response, completed AE/readout real
test, native-dimensional geometry, exact collisions, context availability,
learned attention, distribution statistics, bucket supervision and retained-type
representation effects.

## Updated team holdout evidence

The source is the committed
[training-holdout protocol](https://github.com/Sp1cHless/DSA4266-Cyber-Security/blob/44ef865be777140b1ba34e7b0abe4ec156ea8708/CIC_new/HOLDOUT_PROTOCOL.md)
and its
[three-seed report implementation](https://github.com/Sp1cHless/DSA4266-Cyber-Security/blob/44ef865be777140b1ba34e7b0abe4ec156ea8708/train/seed_report.py).
This is read from `origin/feature/cic-holdout` without switching or merging the
working branch. Numerical team findings are transcribed from that versioned
protocol; the team's raw three-seed predictions are not present in this checkout
and were not re-scored here.

Set A removes 122 XSS, 13 SlowHTTPTest and 81 HOIC primary-labeled training
sequences, reducing P from 8093 to 7877. The pooled withheld test slice has 134
positives. Related Web/DoS/DDoS types remain in training.

Set B removes all seven DoS/DDoS primary labels: 313 real training positives
(3.9%), reducing P to 7780. Its pooled test slice has 202 positives. Validation
and test populations stay the same as the control.

| Model | A held-out R@1%, mean +/- SD | A paired delta, pp | B held-out R@1%, mean +/- SD | B paired delta, pp |
|---|---:|---:|---:|---:|
| CNN | 99.00 +/- 0.43% | -0.75 +/- 0.75 | 68.32 +/- 0.00% | -31.68 +/- 0.00 |
| Hybrid | 98.76 +/- 1.14% | -1.00 +/- 0.86 | 68.32 +/- 0.00% | -31.68 +/- 0.00 |
| LSTM | 98.76 +/- 1.14% | -0.50 +/- 1.14 | 72.94 +/- 4.00% | -26.57 +/- 3.14 |
| XGBoost | 99.00 +/- 0.43% | +2.74 +/- 0.43 | 68.65 +/- 0.29% | -27.72 +/- 0.50 |

Seeds are 4266, 4267 and 4268. Standard deviations describe detector retraining
variation on fixed arm-specific datasets/encoders; the delta is paired by seed.
The source protocol's displayed values are preserved, including rounding.

| Overall test AP, mean +/- SD | Control | Set A | Set B |
|---|---:|---:|---:|
| CNN | 0.5747 +/- 0.0088 | 0.5806 +/- 0.0041 | 0.5047 +/- 0.0105 |
| Hybrid | 0.5788 +/- 0.0139 | 0.5718 +/- 0.0107 | 0.4868 +/- 0.0058 |
| LSTM | 0.5803 +/- 0.0088 | 0.5555 +/- 0.0189 | 0.5081 +/- 0.0031 |
| XGBoost | 0.4153 +/- 0.0016 | 0.4125 +/- 0.0027 | 0.3204 +/- 0.0052 |

These test AP values belong to the new three-seed study, not the historical
single-seed validation screen or the earlier temporal-AE/readout real test.

The source also reports traffic-pattern differences in the set-B CNN: subtype
AP is 1.000 for GoldenEye, SlowHTTPTest and HOIC; LOIC-HTTP mean AP is 0.545 and
recall is 60/110 = 54.55%; LOIC-UDP AP rounds to 0.000. These observations support
pattern-dependent transfer and motivate intensity-sensitive interpretation;
volume was not independently manipulated in a causal ablation.

Retained-family changes are model-dependent. For set B, XGBoost Web recall
changes 36.55% -> 9.04% and Infilteration 52.99% -> 42.81%, lower in all three
seeds; reported AP also falls. LSTM Web recall changes 37.55% -> 26.31% with
nearly unchanged AP, consistent with a changed operating-point tradeoff.
The earlier CNN-only single-seed Web and strict-XSS observations were retracted
by the team after seed replication; the final tree uses the three-seed version.

The protocol removes primary-labeled training sequences and carries an audit of
non-primary held-out memberships in retained mixed sequences. Validation retains
the withheld labels for checkpoint/threshold selection. Each arm's scaler,
vocabulary and bucket encoder are refitted; detector seeds reuse that arm's
frozen representation. Test is the previously inspected historical population,
evaluated with each run's saved validation thresholds.

## Evidence lookup

| IDs | Contents |
|---|---|
| E01-E05 | Perspective, sequence scale, hybrid structure, features, split and initial event-union export |
| E06-E12 | Representation comparison, L2, physical/latent mixtures, AE/readout test and Scheme2 contract |
| E13-E19 | Four-family screen, Projector, attack/host differences, exact collisions and source-label audit |
| E20-E26 | Raw context availability, circular studies, scale retraining, context interventions and attention |
| E27-E30 | Train-only feature probes, 15-statistic extension, changed-flow cases and bucket supervision |
| E31-E32 | Earlier fixed-holdout LSTM and paired-ID native-dimensional geometry |
| E33 | Preservation/archive records |
| E34 | New committed team holdout protocol and three-seed result tables |
| E35 | New committed seed-report implementation and metric definitions |
| E36 | User's handwritten three-page research sketch |

The local `docs/research_map_20261029/evidence_index.csv` preserves all 106 earlier
indexed files plus the two committed snapshots and the input sketch, with SHA256
checks for 109 source files. No source data, model weights or prior completed
experiment outputs were changed, and no new model was trained.
