# EDA 03: raw-feature audit metrics and findings

Archived report: [03_cic_feature_audit.xlsx](../reports/EDA/03_cic_feature_audit.xlsx).
Entry point: [03_audit_features.py](03_audit_features.py). Frozen full-data run: 2026-10-18.
This audits raw flows and historical definitions; it does not rank bucket features by trained predictive value.
The implemented feature decision is in [../preprocessing_v1.md](../preprocessing_v1.md).

## Findings

All 20,115,529 flows and 55 columns were audited; 52 columns have numeric declarations.
Those numeric columns have no observed SQL NULL, nonfinite value or negative value, which does not prove semantic consistency.
The Bernoulli sample contains 100,117 flows: 87,006 benign and 13,111 attack.

- CIC has 149 L7 codes. The old 20-slot vocabulary sends 7,973,696 flows (39.64%) to other:
  45.28% of benign flows versus 1.66% of attack flows. Refit vocabulary on training donors, not on this full-data EDA.
- l7_7_37_ratio, l7_10_16_ratio and l7_36_ratio have no supporting raw code.
- server_tcp_flag_urg_ratio has no supporting TCP bit.
- ftp_command_ret_code is zero everywhere, making all eight legacy FTP features zero.
- These imply 12 identically zero legacy slots, but 88-12 is not the final dimension because L7 representation is refreshed.
- A zero FTP return-code field does not imply absence of FTP attacks.
- Directional durations, size, retransmission, bins, windows and TTL are not global constants; keep candidate information.
- Among forward/reverse flows with at least two packets, 832,382 (7.51%) / 214,691 (2.04%) have IAT averages outside extrema.
  Causes are unverified; no automatic correction/deletion was made.
- 10,854,621 durations (53.96%) differ from end-start. Preserve both fields rather than overwrite one.
  1,423,139 exported durations exceed 100s, consistent with a start-allocation convention that need not contain complete connection intervals.
- longest_flow_pkt equals max_ip_pkt_len on every original record, confirmed by full-table follow-up.
  TTL min/max correlate strongly in the sample but are not identical.

EDA 02 already includes 100s distribution statistics; no separate length audit is needed.
The final preprocessing keeps source IP / 10s / 100s. Its event superposition does not require category embeddings.

## Exact versus sampled scope

Full-table: counts, NULL/nonfinite/zero/negative frequencies, extrema, floating population moments, category frequencies,
TCP-bit support and diagnostic counts. Counts/extrema are exact; floating moments have rounding error.

Sampled: P50/P90/P99 and numeric redundancy. PCG64 seed 4266 independently selects each row with
p=min(1,100000/N), scanning rowid order. Actual sample size is random, not necessarily 100,000.
This is not every-kth-row sampling or label-balanced sampling. Rare attacks may have few observations.

No vocabulary, scaler, clipping threshold, selector or model is fitted here.
Training-fitted preprocessing must only use original training donors after splitting.

## Overview

Metric names identify the quantities; Value contains their numeric values.

| Metric | Meaning |
|---|---|
| Raw flows / Benign flows / Attack flows | Full record count and raw label=0/1 counts |
| Raw fields / Numeric raw fields | Table columns and numerically declared columns, including metadata/target columns |
| Legacy model features | Historical feature_names length |
| Observed L7 categories | Distinct full-data nonnull L7 codes |
| Flows outside legacy L7 vocabulary | Flows with codes absent from old kept_l7_categories |
| Outside legacy L7 / all flows | Previous count / all flows |
| Outside L7 / benign flows | Benign outside-vocabulary count / all benign flows |
| Outside L7 / attack flows | Attack outside-vocabulary count / all attack flows |
| Legacy L7 categories with no flow support | Fixed legacy L7 ratio slots whose code never occurs |
| Full-table constant numeric fields | Numeric columns with finite observations and min=max; not directly a count of removable model dimensions |
| Bernoulli sample rows / Sample benign rows / Sample attack rows | Actual sample and its label composition |

## Profiles

Rows are Field x Population, not bucket features.

| Column | Meaning |
|---|---|
| Field | Original database name, including metadata inspected only for quality |
| Population | all, benign, attack, or field-specific eligible |
| Rows | Population flow count including missing entries |
| Finite | Finite numeric observations |
| SQL nulls | Actual SQLite NULL count |
| Nonfinite nonnull | Non-NULL NaN/infinity count; SQLite commonly converts NaN to NULL |
| Zero values | Finite observations exactly zero |
| Zero / finite | Zero values / Finite; unavailable if no finite values |
| Negative | Finite values below zero, without assuming whether they are valid sentinels |
| Minimum / Maximum | Full-population finite extrema |
| Mean | Full-population finite arithmetic mean, including zeros |
| Population std | Standard deviation with denominator n, merged from stable batch moments |
| Sample finite | Sample observations used for this field/population's quantiles |
| Sample P50 / Sample P90 / Sample P99 | Linear-interpolated sample median/90th/99th percentiles, not exact full-data quantiles |
| Constant | At least one finite value and min=max in this population |

Rows=Finite+SQL nulls+Nonfinite nonnull. all counts equal benign+attack counts.
eligible overlaps these label populations and must not be added to them.

Eligibility: TCP flags/windows use protocol=6; ICMP uses protocol=1;
DNS uses nonzero query ID OR query type OR answer TTL; FTP uses nonzero return code;
forward/reverse IAT uses in_pkts>=2/out_pkts>=2.
These diagnostic conditions do not guarantee exporter availability. DNS/FTP are heuristics.
eligible rows have full-data moments but no sampled quantiles; unsatisfied conditions produce no eligible row.
Zero may mean a real zero, inapplicability, unavailable extraction or sentinel; no blanket zero-to-missing conversion is made.

## Categories

| Column | Meaning |
|---|---|
| Field | Original category field, or one of the three flag masks |
| Value / bit | Stored code, or TCP bit name; code magnitude is not a metric distance |
| Flows | Flows with that category, or TCP flows containing the bit |
| Benign / Attack | Those occurrences partitioned by raw label |
| Share denominator | All raw flows for categories; all TCP flows for bits |
| Flow share | Flows / Share denominator |
| Attack / value flows | Attack / Flows, not attack recall |
| In legacy L7 | Old-vocabulary membership, only applicable to l7_proto |
| Coverage | Full category display versus top 50 ports |

Several bits can occur in one flow, so bit shares do not sum to one.
Category zero values are retained; ICMP type 0 is valid.
All counts are full-table. Excel shows top 50 source/destination ports but CSV/JSON retain every value.
categories.csv uses flow_fraction/attack_fraction for the two shares above; flags.csv uses tcp_fraction.

## Legacy 88

Each row corresponds to one historical feature.

| Column | Meaning |
|---|---|
| Feature / Family | Stable historical name and logical group |
| Raw dependencies | Read raw fields; flow_count depends on records |
| Merge rule | Required recomputation/intermediate data, not simple vector addition |
| Supporting flows | Fixed L7-code, TCP-bit or relevant nonzero FTP-code occurrences; blank means not applicable |
| Recommendation | Baseline/categorical replacement/unsupported-slot advice, not a fitted selector |
| Interpretation | Raw support is not full bucket variance or predictive importance |

Duration mean is a per-flow mean. TCP bit denominators are TCP flows.
Protocol/L7 ratios use all flows; DNS/ICMP type top1 use eligible flows.
Entropy=-sum(p*ln(p))/ln(k), zero for k<=1; port/IP reuse=1-nunique/flow_count.
Changing k on merge means entropies cannot be averaged.

## Aggregation

| Column | Meaning |
|---|---|
| Raw field / SQLite type | Original field and declared storage type |
| Family / Unit | Logical group and unit; unverified units explicitly say verify |
| Legacy dependency | Whether the old implementation reads it, not necessarily as a direct feature |
| Proposed role | Metadata/target, baseline candidate source, or additional candidate source |
| Constant | Raw full-table constant status, not model usefulness |
| Aggregation proposal | Suggested mergeable information or event-level recomputation |
| Caveat | Applicability, exporter semantics or fingerprint limitations |

Counters can be summed; means/std require counts and moments; categories require frequency maps.
Distinct/top1/reuse/entropy/ratios are recomputed after merging.
Quantiles generally cannot be exactly combined from donor quantiles alone.
Per-flow packet IAT summaries do not reconstruct packet-level IAT after merging.
New inter-flow start gaps must be recalculated from sorted merged starts.

Raw IP/time/labels remain provenance. Current counters are start-allocated, not interval-measured.
The registry is an information-preservation proposal, not a mandatory list of all model inputs.

## Redundancy

| Column | Meaning |
|---|---|
| Left field / Right field | Numeric scalar candidates, excluding dates, IDs, targets, category codes and flag integers |
| Paired sample | Both values finite and nonnegative |
| Pearson log1p | Pearson correlation after independent log(1+x) transforms |
| Absolute correlation | Absolute coefficient, used for sorting |
| Equal raw values | Fraction with exactly equal untransformed values in the paired sample |

Pairs with fewer than three observations or a constant transformed member are omitted.
Shared zeros can inflate equality. Pooled correlations may reflect source/label composition.
No significance test, causal inference or final feature removal is performed.
High raw correlation need not imply post-bucket interchangeability.
Protocol ratios sum to one; reuse is determined by nunique/count; these structural redundancies are separate from empirical correlation.

## Diagnostics

Columns: Check, Affected rows, Eligible rows, Affected share, Interpretation.
Affected share=Affected rows/Eligible rows, or n.a. for zero denominator.

- end_before_start: invalid negative endpoint span.
- duration_differs_from_endpoint_span: exported duration differs from end-start; semantic diagnostic, not an automatic cleaning rule.
- duration_over_10s / duration_over_100s: exported duration longer than bucket/window.
- invalid_binary_label: label outside 0/1; the script stops on such data.
- ttl_min_over_max, ip_packet_min_over_max, flow_packet_shortest_over_longest: reversed stored extrema.
- direction-specific iat_min_over_max / iat_avg_outside_extrema: checked for at least two directional packets.
- *_out_of_range: ports outside 0..65535 or TTL/protocol outside 0..255.
- longest_packet_differs_from_max_ip_length: full-table duplicate-source check following sample equality, not an invalid-record condition.

## Method and outputs

Item / Definition or value record source paths, generation time, size/nanosecond mtime, code/reference SHA256,
sampling parameters, runtimes, populations and text-field NULL/empty/distinct counts.
IP distinct counts are address codes, not confirmed devices.

Source references: [UQ dataset documentation](https://staff.itee.uq.edu.au/marius/NIDS_datasets/) and
[nProbe field documentation](https://www.ntop.org/guides/nprobe/flow_information_elements.html).
Current exporter docs do not prove every behavior of the dataset's exporter version.
IAT units and second_bytes semantics were not independently resolved; no forced conversion was made.

results.json contains all values/provenance. profiles.csv is full-data quality/moments; quantiles.csv is sampled.
text_profiles.csv describes text quality. categories.csv contains every category and flags.csv every bit.
legacy.csv/registry.csv and aggregation_registry.json hold descriptive candidate rules.
diagnostics.csv/correlations.csv contain complete diagnostic/pair results.
run_status.json becomes complete only after successful analysis and Excel.
Source access is read-only; render-only reexports saved results, and fresh runs do not overwrite historical output.
