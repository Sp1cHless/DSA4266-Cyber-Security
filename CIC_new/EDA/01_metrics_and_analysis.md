# EDA 01: perspectives, metrics and findings

Archived workbook: [01_cic_perspectives.xlsx](../reports/EDA/01_cic_perspectives.xlsx).
Frozen source: NF-CICIDS2018-v3, 20,115,529 flows and 2,600,903 attack flows, analyzed 2026-10-18.

## Perspectives

- source: group flows by original source address;
- destination: group by original destination address;
- endpoint: count each distinct endpoint once per flow; a self-flow contributes once;
- network: all records form one reference entity.

Source/destination memberships each reconcile to raw totals. Endpoint memberships can approach twice raw totals.
Do not add different views as independent datasets. Distinct addresses do not prove distinct physical devices.

## Core metrics

An active bucket is an entity's 10s interval with at least one recorded start. Wholly unobserved buckets are not enumerated.
One-flow buckets are active. All flow counts and exported volume are assigned by start, not redistributed by packet time.
Entity/bucket classes are benign_only, attack_only and hybrid according to observed labels.
Hybrid means both labels occur within that entity's history or bucket, depending on the table; inspect the table's unit.

Overview compares entity counts, active buckets, membership flows, positive/hybrid composition, one-flow density and source concentration.
Entity classes partitions entities into the three classes; hybrid entities need not have hybrid buckets.
Bucket density gives per-active-bucket flow distributions. Daily restricts active buckets to a UTC date.
Attack types reports original type flow totals and distinct endpoints. Top entities contains bounded examples by flow and attack-flow ranks.
Method records allocation, self-flow accounting, denominator definitions, database fingerprints and code version.

## Full entity CSV fields

| Field | Definition |
|---|---|
| perspective / entity | Grouping view and address/reference key |
| activity_class | Entire observed history's label composition |
| flows / benign_flows / attack_flows | Membership counts and their binary-label partition |
| attack_fraction | attack_flows / flows |
| active_days_utc | Distinct dates with starts |
| first_start_utc / last_start_utc | First/last recorded starts, UTC |
| calendar_span_s | Last-first start span; gaps are included |
| active_buckets | Recorded entity x10s positions |
| benign_only_buckets / attack_only_buckets / hybrid_buckets | Active positions partitioned by label composition |
| hybrid_bucket_fraction | hybrid_buckets / active_buckets |
| flows_per_active_bucket | flows / active_buckets |
| longest_active_run_buckets | Longest adjacent active-position run, reset at UTC date boundaries |
| max_unobserved_gap_within_day_s | Largest within-date gap between recorded active positions; not confirmed silence |
| total_bytes / total_packets | Sums of complete exported directional counters |
| max_flow_duration_ms | Maximum stored exported duration, not reconstructed from endpoints |
| outgoing_flows / incoming_flows | Directional membership diagnostics for the chosen view |

Distribution statistics use count n, arithmetic mean, linear-interpolated P25/P50/P75/P90/P99 and maximum.
Top-k concentration is the share of the relevant count contributed by the k highest-ranked entities.
Attack-source counts by type are not additive because an entity may contribute several types.
The full CSVs include all entities; bounded Excel top lists are examples, not complete source coverage.

## Findings

Source has 181,876 entities: 180,718 benign-only, 195 attack-only and 963 hybrid.
Destination has 29,036 entities: 27,650 benign-only, 9 attack-only and 1,377 hybrid.
Source has 3,657,936 active bins with median two flows; destination has 4,833,758 with median one.
The source top ten contribute 62.93% of attack flows, versus destination top ten 98.63%.
This motivates source IP for the current per-source modeling target, without proving source-ID generalization.

Endpoint has 205,801 entities and 40,122,932 memberships after self-flow accounting.
There are 108,126 self-flows, all benign. Special addresses such as 0.0.0.0 were retained in descriptive statistics.
Multiple flows can start in the same millisecond; the maximum observed at one start millisecond is 126.
10,854,621 exported durations differ from endpoint span; this is a semantic discrepancy, not an automatic deletion rule.

Follow-up sequence analysis is [02_metrics_and_analysis.md](02_metrics_and_analysis.md).
