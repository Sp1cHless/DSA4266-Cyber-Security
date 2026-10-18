# EDA 02: sequence-length metrics and findings

Archived report: [02_cic_sequence_lengths.xlsx](../reports/EDA/02_cic_sequence_lengths.xlsx).
This guide describes the full NF-CICIDS2018-v3 run of 2026-10-18. Later reruns have their own outputs.
The adopted preprocessing sequence length is now **100 seconds**.

## Basic definitions

A flow is one original communication record; simultaneous starts are allowed.
A source bucket is a ten-second half-open interval allocated by flow start.
An active bucket has at least one recorded start. An empty bucket has none, but may still contain an ongoing connection.
A window/sequence is a fixed-length set of bucket positions for one source.
Wholly empty windows are omitted; internal empty positions remain.

| Activity class | Definition |
|---|---|
| benign_only | At least one flow, all label 0 |
| attack_only | At least one flow, all label 1; internal empty buckets allowed |
| hybrid | Both labels somewhere in the original window; not necessarily simultaneous |
| all | Pooled statistics of the three mutually exclusive classes, not a fourth class |

No augmentation has occurred in this EDA. Natural hybrid is not synonymous with synthetic.

Windows reset at each UTC midnight; stride=length and no overlap within a length.
Different lengths repartition the same flows and must not be added as independent data.
500/1000s do not divide the day; their last 400 seconds are excluded and reconciled.
No capture-session ground truth is assumed. Global no-start bins are flagged, not filled.

### Worked 100s example

```text
Bucket:         0  1  2  3  4  5  6  7  8  9
Class:          B  .  M  H  .  .  B  .  .  .
Benign starts:  1,2,3s; 31,32s; 64s
Attack starts:  25,26s; 33s
```

Nine flows: six benign and three attack. Four active buckets give occupancy 40%.
There is one mixed bucket, a longest empty run of 30s, longest active run 20s, and flow-start span 63s.
The activity mask is 0x4d: bits 0,2,3,6. None of these start statistics establishes packet-level interval overlap.

## Overview

For one length, let N=length/10, W=nonempty windows, W_B/W_M/W_H be class counts,
and F_i/M_i/K_i be total flows, attack flows and active positions in window i.

| Metric | Definition |
|---|---|
| Buckets / window | N, including empty positions |
| Stride (s) | Distance between candidate window starts; equals length |
| Nonempty windows | W; even a one-flow window is retained |
| Source IPs represented | Distinct sources contributing retained windows |
| Benign-only windows | W_B |
| Attack-only windows | W_M |
| Hybrid windows | W_H |
| Attack-bearing window share | (W_M+W_H)/W, a sample-level positive fraction |
| Hybrid / attack-bearing windows | W_H/(W_M+W_H); missing when no positive window |
| Hybrid with a mixed bucket | Hybrid windows having at least one position with both labels |
| Hybrid across separate buckets only | Hybrid windows with both labels, but never in the same position |
| Mean bucket occupancy | mean(K_i/N), also sum(K_i)/(W*N) for one length |
| Median bucket occupancy | Median K_i/N |
| P90 bucket occupancy | 90th percentile of K_i/N |
| One-active-bucket window share | Count(K_i=1)/W; does not mean one flow |
| Fully active window share | Count(K_i=N)/W; does not imply activity every millisecond |
| Median flows / window | Median F_i |
| P90 flows / window | 90th percentile F_i |
| Median longest empty run (s) | Median of each window's longest consecutive empty run times ten |
| Median flow-start span (s) | Median of last start minus first start, not connection duration |
| Raw flow coverage | Retained distinct original flows / original flows |
| Window share: global no-start bins | Windows containing at least one bin with no start anywhere in the dataset / W |
| Top 10 source window share | Top ten sources by window count / W |
| Top 10 source attack-window share | Independently ranked top ten sources by positive-window count / all positive windows |

The two hybrid subtypes sum to W_H. Mixed start buckets do not prove overlapping connection intervals;
long connections starting in separate buckets may overlap.

Do not confuse positive-window fraction, mean within-window attack fraction and pooled attack-flow fraction.
For two windows containing (1 attack,99 benign) and (0 attack,900 benign), these are 50%, 0.5% and 0.1%.

## Classes

Each row is length x activity class. Windows are disjoint by class; sources may recur across classes and are not additive.

| Column | Definition |
|---|---|
| Length (s) | Window length |
| Activity class | Population used |
| Windows | Number of windows in this class |
| Source IPs | Distinct sources contributing this class |
| Flow starts | Complete original records allocated to these windows |
| Attack flows | Their label-1 records |
| Active buckets | Sum of active positions across these windows |
| Mixed-bucket hybrid windows | Hybrid subtype with at least one mixed position |
| Separate-bucket hybrid windows | Other hybrid subtype |
| Global gap windows | Count of windows containing a global no-start position |

At 100s, 5,210 hybrid windows have 26,050 active positions: mean five positions and occupancy 50%.

## Distributions

Each row is length x class x Metric. The pooled all population is dominated by benign windows.
Columns: Length (s), Activity class, Metric, Unit, Windows, Mean, P25, Median, P75, P90, P99, Maximum.
Windows is the sample count; Mean is an unweighted window mean.
Quantiles are exact empirical linear interpolations at sorted position (n-1)*q; fractional flow-count quantiles are valid.
Empty classes have zero count and unavailable quantiles, not fabricated zero observations.

| Metric | Unit | Per-window calculation |
|---|---|---|
| flows | flows | F_i |
| active_buckets | buckets | K_i |
| occupancy | fraction | K_i/N |
| longest_empty_run_s | seconds | Longest consecutive empty positions x10, including leading/trailing gaps |
| longest_active_run_s | seconds | Longest consecutive active positions x10, irrespective of label |
| flow_start_span_s | seconds | (last start-first start)/1000 |
| attack_flows | flows | M_i |
| attack_fraction | fraction | M_i/F_i |
| total_bytes | bytes | Sum of whole-flow in_bytes+out_bytes |
| total_packets | packets | Sum of whole-flow in_pkts+out_pkts |

Complete exported counters belong to start windows, not to packet-level observed traffic in those intervals.

## Occupancy

| Column | Definition |
|---|---|
| Active buckets | Exact K represented by this row |
| Total buckets | Positions per window N, not a pooled total |
| Active share | K/N |
| Windows | Number of windows having that K |
| Share within class | Row Windows / all windows of this length/class |

Length (s) and Activity class identify the population. Unobserved occupancy bands are absent, not missing data.

## Attack types

Labels preserve original spelling, including Infilteration.
A multi-type window may appear in several attack rows, so window counts across types are not additive.

| Column | Definition |
|---|---|
| Attack type | Original attack name |
| Windows | Windows containing this attack |
| Source IPs | Distinct retained sources for this attack |
| Attack-only windows | These windows with no benign flows, possibly other attacks |
| Hybrid windows | These windows also containing benign flows |
| Attack flows retained | Original records specifically of this attack in retained windows |
| Raw attack flows | All original records of this attack |
| Flow coverage | Attack flows retained / Raw attack flows |

For each row Attack-only windows+Hybrid windows=Windows.
Attack-flow counts are additive across original single-label attack names.
100% flow coverage does not establish many independent sources or episodes.

## Daily and Coverage

Daily rows use Length (s), UTC date, Windows, Source IPs, Benign-only, Attack-only, Hybrid, Flow starts,
Attack flows, Mean occupancy and Global gap windows with the same definitions restricted to one UTC date.
Sources recurring on several dates are not additive. An absent date is not verified zero traffic.

| Coverage column | Definition |
|---|---|
| Length (s), Total buckets, Windows | Length, positions per window, retained nonempty-window count |
| One-active-bucket windows | Count with K=1 |
| Fully active windows | Count with K=N |
| Global gap windows | Count with any global no-start position |
| Flows retained | Distinct original records retained |
| Raw flows | 20,115,529 original records in this frozen run |
| Tail flows excluded | Records in incomplete UTC day-end windows |
| Attack flows retained | Retained malicious records |
| Raw attack flows | 2,600,903 in this run |
| Tail attacks excluded | Malicious records excluded by day-end tails |
| Flow coverage | Flows retained / Raw flows; four decimal percentage precision |
| UTC day tail (s) | 86400 modulo length |
| Active buckets retained | Distinct source x10s active bins retained |
| Tail buckets excluded | Source x10s active bins excluded |

Retained+excluded equals raw controls for flows, attacks and EDA 01 source active bins.
Omitting wholly empty windows loses no flow starts.
500/1000s each exclude 450 benign flows and 263 active bins, no attacks: coverage 99.9978%, not exactly 100%.

## Method

Item and Definition or value document dataset, generation time, grouping, bucket, lengths, stride, UTC alignment,
day-end tails, capture-boundary assumptions, empty-bin policy, classes, hybrid definitions, overlap caveats,
run/span definitions, quantiles, pooled populations, source-count scope, attack names, coverage, catalogs and mask.
Source cache/Raw SQLite locate inputs. Raw/Cache mtime fingerprints are size/mtime provenance, not content hashes.
Analysis script SHA256 and EDA 01 analysis SHA256 identify code versions, not database hashes.
Runtime (s) and Python / SQLite record execution. Training confirms this EDA fits no model, threshold or split.

## Measured findings

| Length | Nonempty windows | Positive windows | Mean active bins | Mean occupancy | Median flows | Median start span |
|---|---:|---:|---:|---:|---:|---:|
| 50s | 1,960,841 | 16,225 | 1.87 | 37.31% | 2 | 3.70s |
| 100s | 1,466,665 | 10,918 | 2.49 | 24.94% | 3 | 4.10s |
| 200s | 1,114,680 | 7,304 | 3.28 | 16.41% | 3 | 4.12s |
| 500s | 800,775 | 4,328 | 4.57 | 9.14% | 3 | 4.16s |
| 1000s | 650,256 | 3,148 | 5.62 | 5.62% | 3 | 4.14s |

Median active-bin count is one at every length. Increasing length does not resolve widespread sparsity.
At 100s, benign/attack-only/hybrid mean occupancy is 24.77%/46.71%/50.00%.
Longer context is useful for some hybrid activity: its median active bins rise 3/4/7/13/22.
Hybrid median start span rises 31.23/70.72/157.65/431.18/892.12s.
Its median attack fraction falls from 50% at 100s to about 16.75% at 1000s.

Hybrid / positive-window fractions rise 39.98%,47.72%,55.08%,63.10%,66.07%;
attack flows inside hybrid / all attack flows are only 6.46%,7.36%,8.26%,9.89%,11.61%.
Window percentages and flow percentages answer different questions.
At 100s, 4,562 of 5,210 hybrid windows contain Infilteration (87.56%).
DoS/DDoS/FTP-BruteForce/SSH-Bruteforce have no natural hybrid windows at any tested length.
Several attack types have only one source. Hulk has 17/9/2 windows at 50/100/1000s but one source;
GoldenEye 23/14/4; FTP 117/59/7; SSH 111/56/7.
Larger or augmented window counts do not create independent attack sources.

100s was selected as a practical local-context baseline, not an empirically optimal model length.
Keep sparse samples and distinguish density; use B+B alongside B+A augmentation, and split original donors before augmentation.

## Full catalog fields

`windows_<length>s.csv.gz` is complete. Its identifiers are source_ip plus start_ms at one length.
start_ms/end_ms_exclusive are UTC epoch-ms half-open boundaries.
activity_class, flows/benign_flows/attack_flows, attack_fraction, active_buckets/total_buckets/occupancy use definitions above.
bucket_mask_hex has bit 0 first; e.g. positions 0,2,3,6 give 0x4d.
hybrid_buckets counts mixed positions; attack_buckets counts positions with any attack start.
longest_empty_run_s/longest_active_run_s include boundary positions.
first_flow_offset_s/last_flow_offset_s are start offsets from the left edge; flow_start_span_s is their difference.
first_attack_offset_s/last_attack_offset_s are analogous and unavailable in benign-only windows.
global_active_start_buckets counts network bins with any start inside the interval; it is not sensor uptime.
total_bytes/total_packets use complete flow counters. max_flow_duration_ms preserves the exported duration.
attack_types is the sorted semicolon-separated original attack-name set.
