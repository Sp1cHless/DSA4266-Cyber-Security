## Scope and files

`bucket_summary.xlsx` summarizes the generated dataset. `summary.json` contains its machine-readable values;
`feature_schema.json` fixes the feature order and training L7 vocabulary; `scaler.json` contains training-only transforms.
`split_notes.json` records block policy, coverage reservations, exclusion counts, actual ratios and per-type retained counts.
These small reports belong in Git. Large shards, event caches, manifests and preview images remain under the ignored dataset output directory.

The summary describes dataset construction, not model accuracy, independent attack episodes, or deployment class priors.

## Summary and Composition

| Metric | Meaning |
|---|---|
| Split | train, validation or test, assigned before augmentation using whole global 15-minute blocks; source-IP 100s windows are retained or excluded intact |
| Sample type | real_benign, real_malicious, synthetic_benign or synthetic_malicious; real_malicious includes natural hybrid windows |
| Sequences | Number of retained 100-second samples; all have at least one flow start |
| Positive sequences | Samples containing at least one malicious flow, regardless of background or attack fraction |
| Positive share | Positive sequences / sequences, not malicious flows / flows |
| Real sequences | Retained original windows, counted once each across the three splits; excludes boundary/interval windows |
| Synthetic sequences | B+A and B+B combinations; only train contains them |
| Bucket positions | Sequences multiplied by ten, including internal empty positions |
| Active buckets | Positions containing at least one recorded flow start |
| Mean occupancy | Active buckets / bucket positions; each position has equal weight |
| Mixed buckets | Positions containing both benign and malicious flow starts; this does not prove packet-level interval overlap |
| Flow presentations | Number of complete flow records included across these samples; synthetic donor reuse is counted again |
| Attack flow presentations | Same accounting, restricted to malicious flow records |
| Original training N | Original nonempty training sequences, before augmentation |
| Original training positives P | Original training windows containing at least one malicious flow; multi-type windows count once |
| Synthetic B+A / Synthetic B+B | Exactly P of each, appended to all original training samples |
| Feature columns (without mask) | 82 fixed statistics + one ratio per training L7 code + one UNK ratio |
| L7 slots including UNK | Vocabulary width V; original code 0 is separate from UNK and empty buckets |
| Buckets per sequence | Ten; each represents ten seconds |
| Scaler-fit active buckets | Original training nonempty positions used to fit transforms; excludes synthetic, validation, test and empty positions |
| Unique source flows / Unique source attack flows | Independent raw-table controls, not augmented flow presentations |
| Excluded original sequences / flows | Raw windows/records omitted by temporal guard and interval rules; present only in the audit cache |

If original train contains N windows and P positives, final train contains N+2P windows:
N-P real benign, P real malicious, P synthetic benign and P synthetic malicious.
Its positive count is 2P and positive share is 2P/(N+2P). No balanced training batch is imposed at this stage.

An empty bucket means no new recorded start, not necessarily no ongoing connection. Whole-flow bytes, packets and exported durations
belong to the flow-start bucket. Long flows may end after their sequence boundary; no packet-level redistribution is inferred.

## Split

This sheet reports original samples **before augmentation**, in train/validation/test/excluded order.
Retained share divides a retained set's count by the sum of the three retained sets; excluded is unavailable.
Original share divides every row by all original nonempty sequences, including exclusions. These four shares sum to one.
Flows, attack flows, positive sequences and sources concern original data only.
The lower table shows each attack's retained window counts and whether a held-out type is absent from train.
Zero held-out coverage is explicitly displayed. Type-level window counts can overlap for multi-type windows.

Coverage reservations protect all training types and then greedily seek feasible validation/test coverage.
The same seed is used throughout. Three deterministic count-only corrections approach 60/20/20 after purge.
No model scores or seed search select the partition. Same sources and campaigns can recur across folds.
Embargo exclusions and interval exclusions in `split_notes.json` can overlap and must not be added together.

## Attack coverage

| Column | Meaning |
|---|---|
| Attack type | Original dataset label; spelling is preserved, including Infilteration |
| Real sequences | Original windows in this split containing this attack; multi-type windows appear in multiple rows |
| Real sources | Distinct source addresses for this attack and split, not independent physical devices or attack episodes |
| Real attack flows | Original records specifically labelled with this attack, not all positive records in these windows |
| Synthetic selected | Number of generated positives for which this type was explicitly selected by the donor sampler |
| Synthetic containing | Number of generated positives actually containing this attack, including incidental types in multi-type donors |

Type/source-balanced donor selection does not create new independent attack sources.
Selected-type counts need not be exactly equal because selection is random; containing counts can overlap.

## Features

| Column | Meaning |
|---|---|
| Index | Zero-based column position in X_raw and X; activity mask is separate |
| Feature | Stable name in feature_schema.json |
| Family / Definition | Group and calculation, including the applicable denominator |
| Log1p | Whether log(1+x) precedes standardization; ratios and normalized entropies are not logged |
| Train mean after log | Mean after the selected transform, fitted only on original train active buckets |
| Train scale | Population standard deviation after transform; training-constant columns use scale 1 |
| Train constant | Transformed training standard deviation below 1e-12; the column remains in the common schema |

The 82 fixed statistics comprise 58 retained legacy statistics and 24 additions. All eight FTP fields and server URG are removed;
the 21 NB15-specific L7 ratios are replaced by the CIC training vocabulary.

- Volume: flow count; sums of in/out bytes and packets. These are whole-flow counters assigned by start time.
- Duration: exported flow-duration mean/min/max, plus mean/max of duration_in and duration_out. Stored zero values are retained.
- Protocol: TCP/UDP/other flow ratios and distinct protocol count.
- L7: original-code distinct count, top1 ratio and normalized entropy, plus vocabulary ratios including UNK.
- Source/destination ports and destination IP: distinct count, reuse=1-distinct/all-flow-count, top1 ratio and normalized entropy.
- TCP flags: each named bit fraction over TCP flows; server URG omitted. No TCP gives zero.
- ICMP: protocol=1 flow ratio and type distinct/top1/entropy within those flows.
- DNS: legacy nonzero query-ID OR query-type OR answer-TTL predicate, then query-type distinct/top1/entropy within that population.
- Packet size: min and mean of per-flow min_ip_pkt_len; mean and max of max_ip_pkt_len. These do not reconstruct pooled packet moments.
- Retransmission: sums of four directional byte/packet retransmission fields.
- Packet-size bins: five exported bin sums divided by their combined sum; zero denominator yields five zeros.
- TCP window: per-flow window mean/max over TCP flows only.
- Start timing: mean/max adjacent sorted start gap and last-first span in milliseconds. Duplicate starts contribute zero gaps; fewer than two starts gives zeros.

Normalized entropy is -sum(p*ln(p))/ln(k), and zero for k<=1. DNS/ICMP top1 ratios use eligible counts;
other category top1 ratios use all bucket flows. Diversity is computed before mapping unseen L7 values into UNK.

TTL, packet-level exported IAT, exporter rate fields, literal IP/port codes, dates, DNS transaction IDs, targets and provenance are not extra model columns.
No embedding or L2 norm is necessary. The unknown L7 slot avoids changing the schema on validation/test.

## Checks

Each row gives Check, Expected, Observed and Passed. Equality must hold before the report can complete.
Checks cover N+2P, 2P positives, raw flow/attack conservation, donor split/source constraints, feasible shifts,
exported sequence counts, finite values, empty-bucket zeros and target/manifest agreement.
Source size and nanosecond mtime are checked before/after processing; this fingerprint is not a full raw-file content hash.
SQLite rowid ownership is unique in the event cache, and the complete-window manifest assigns every original row to one retained split or exclusion.
Checks also cover whole-block assignment, no excluded samples, retained-plus-excluded raw conservation,
all attack types in train, no shared global 100s windows, no adjacent retained source windows crossing sets,
and no retained interval-envelope overlap with another set's window.

## Method and dataset loading

Method records the source, UTC+08 generation time, seed, split ratios, code hashes and processing rules.
The split is not source-disjoint: different nonoverlapping windows from one IP may occur in different sets.
Missing validation/test attack types are reported. Coverage is not a claim of independent campaigns.
Identical complete raw rows have the same source/start grouping key and therefore stay in one original window;
duplicates within that window are not removed, and reverse-flow/session/campaign identity is not inferred.

Final files are `splits/<split>/part_*.npz`. Load with NumPy and `allow_pickle=False`:
`X` is float32 `(samples,10,F)`; `mask` and `y_bucket` are uint8 `(samples,10)`;
`y_sequence` is uint8 `(samples,)`; `sample_id` is int64. Flow-count arrays support reconciliation.
Raw float64 features remain in the corresponding `raw_shards/` files as `X_raw`.
`manifest.sqlite` maps sample IDs to splits, sample types, donor IDs, offsets and shard rows.

The output pool keeps its construction proportions. Batch balancing, loss weighting, epoch length, model fitting,
threshold selection and optional representation experiments are later training decisions.
