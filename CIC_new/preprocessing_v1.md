# CIC preprocessing: feature v1 and split policy v2

The implementation is `build_buckets.py`. Scope ends at model-ready bucket data, before model/loss/batch/epoch decisions.
Only CIC is used and the original SQLite remains read-only.

## Pipeline

1. Group original flows into source-IP 100s windows, UTC-midnight aligned, stride 100s.
2. Assign whole global 15min blocks, target **60% train / 20% validation / 20% test**, seed **4266**; purge boundary/interval windows.
3. Keep every surviving original window. Count retained original training windows N and attack-bearing training windows P.
4. Generate **P B+A** and **P B+B** from original train only; append them to the N originals.
5. Aggregate every real/synthetic window into ten 10s buckets with features, masks and labels.
6. Fit vocabulary/scaling on original train only and reuse for all other samples.
7. Save normalized/raw shards, schema, transforms, provenance, final Excel and English explanations.

Wholly empty windows are omitted; internal empty positions and single-flow/one-active-bucket windows are retained.
100s divides a UTC day, so there is no incomplete day-end tail.

## Split and targets

The sequence key is `(source IP,100s window start)`, not a flow row.
The key is `floor(start_ms/100000)*100000` for this nonnegative UTC dataset.
Computing a grouping key is not feature fitting; right-boundary starts belong to the next window.

The assignment unit is a global UTC-aligned 900s block: all sources in that block share an assigned fold.
Start from a seeded group allocation. Process rare-block-support attack types first, reserving a training witness and
its surrounding blocks so that it survives purging. Reserve two pure-benign donor sources in training.
Greedily reserve feasible validation/test witnesses without breaking previous reservations, alternating the first held-out fold by type.
Prefer already retained initial members, then larger type-flow support, then move cost, seeded rank and stable sequence index.
Labels are used only to design split coverage; no model score or feature distribution chooses a partition.

Allocate remaining whole blocks by largest sequence count first, with seeded tie order and a 60/20/20 count target.
Apply three deterministic retained-ratio corrections and keep the smallest squared deviation from 60/20/20 among those four assignments.
All comparisons use counts only and retain the same coverage reservations and seed. Exact proportions are not forced.

At every change in assigned global fold, exclude whole windows whose support intersects a 100s guard on either side.
Support is [window start, max(window end, maximum actual flow end + 1ms)). Also exclude a window if its actual
[first flow start, maximum actual flow end] envelope intersects another assigned fold's 100s window.
Use actual end timestamps, not exported flow duration. Exclusions are not reassigned.
Recheck all training types and coverage witnesses after purge. A type in only one feasible block remains train-only.
Excluded windows remain in the manifest/event cache for auditing but are absent from all samples, feature fitting and donors.

The full-data split-only preflight yields 765,652 / 251,146 / 263,125 retained originals and 14 / 13 / 12 attack types.
186,742 windows are excluded; counts and coverage are recorded in `reports/split_policy_v2_preflight.json`.

Different nonoverlapping windows from one IP may occur across sets. This is not a claim of unseen-source generalization.
Attack campaigns may recur: this is held-out-time-block detection in the same environment, not held-out-campaign evaluation.
No original rowid or complete window is shared across retained splits. Identical complete raw rows have the same source/start
key and therefore cannot cross sets. Within-window duplicates are retained; session identity is not inferred.
Bucket label is the maximum flow label; empty bucket label/mask are both zero.
Sequence label is the maximum bucket label. Real positives include attack-only and natural hybrid windows.
Attack names and labels stay outside the inputs.

## Features

Use **82 fixed statistics plus V L7 ratios**, where V includes all original-training L7 codes and a separate UNK.
X has shape `(samples,10,82+V)`; mask is separate. Concatenating mask gives width `83+V`.
No mandatory embedding, fixed 115-dimensional input or L2 stage is used.

### Retained 58 statistics

| Group | Count | Definition |
|---|---:|---|
| Volume/duration | 8 | flow_count; in/out byte/packet sums; exported duration mean/min/max |
| Protocol | 4 | TCP/UDP/other ratios and distinct count |
| L7 diversity | 3 | original-code nunique/top1/normalized entropy |
| Source/destination ports | 8 | each: nunique/reuse/top1/entropy |
| TCP flags | 23 | three eight-bit groups excluding server URG; TCP denominator |
| ICMP | 4 | protocol=1 ratio; eligible type nunique/top1/entropy |
| DNS | 4 | legacy presence ratio; eligible query-type nunique/top1/entropy |
| Destination IP | 4 | nunique/reuse/top1/entropy, not literal address |
| **Total** | **58** | 88 minus eight FTP fields, server URG and 21 old L7 slots |

The three unsupported old L7 categories were already among those 21 slots; do not subtract them again.

### Added 24 statistics

| Source | Statistics | Count |
|---|---|---:|
| duration_in/out | each mean/max, ms; all flows, stored zeros retained | 4 |
| min_ip_pkt_len / max_ip_pkt_len | former min/mean; latter mean/max, bytes | 4 |
| Four in/out byte/packet retransmission fields | sum of each | 4 |
| Five num_pkts_* bins | each sum / combined five-bin sum; zero denominator gives zeros | 5 |
| tcp_win_max_in/out | each TCP-only mean/max; no TCP gives zero | 4 |
| Sorted bucket starts | adjacent gap mean/max and last-first span, ms | 3 |
| **Total** | | **24** |

Packet bins are not assumed exhaustive. Per-flow length extrema do not recover pooled packet moments.
Duplicate starts contribute zero gaps; fewer than two starts gives zero timing values.
Keep max_ip_pkt_len and omit its audited duplicate longest_flow_pkt.
Preserve exported duration rather than silently replacing it with end-start.

## Categories

Superposition merges records, not category codes. TCP/UDP plus TCP yields TCP ratio 2/3, UDP ratio 1/3.
Counts can be combined, but ratios/nunique/top1/reuse/entropy are recalculated.
Sub-bucket shifts change membership, so event-level reaggregation is the reference.

- L7: original-training codes plus UNK. Code 0 is ordinary, not an empty/unknown token.
- L7 ratios: code-flow count / all bucket flows; active ratios sum to 1, empty ratios to 0.
- L7 diversity: original codes before collapsing unknown values to UNK.
- DNS eligibility: query ID nonzero OR query type nonzero OR answer TTL nonzero; a legacy heuristic, not protocol ground truth.
- ICMP eligibility: protocol=1; type 0 is valid.
- DNS/ICMP top1 denominators: eligible flows. Other top1 denominators: all bucket flows.
- Normalized entropy: `-sum(p*ln(p))/ln(k)` for k>1, otherwise zero.
- Port/destination reuse: `1-nunique/flow_count`.

IP literals, dates, DNS transaction IDs, targets and synthetic/donor metadata are not inputs.
DNS IDs are used only for the presence predicate. Port IDs are equality categories, not continuous magnitudes.
TTL, exported packet IAT and exporter rates are omitted. TCP flags use bit ratios.
Embedding may be studied later, but is not required for this event-level generator.

## Scaling and empty buckets

Log1p magnitudes/counts/nunique/durations/gaps, not ratios or normalized entropies.
Fit transformed mean/population std using **original train active buckets only**.
Std below 1e-12 uses scale 1; retain the common column. Never fit separately by target.
Reuse the transform for all sets and synthetic samples, then restore empty features to zero.
Mask=1 means at least one start; no new start does not prove no ongoing connection.
Whole-flow counters belong to the start bucket, not measured ten-second link traffic.

## Superposition

B is real benign-only; A contains at least one attack and can already be hybrid.
Synthetic positives use B+A, negatives B1+B2.
Donors are distinct original training windows with **different source IPs**.
All retained originals, including pure malicious windows, remain in the final pool.

1. Subtract original window start from donor flow start/end timestamps.
2. Keep the first benign donor fixed.
3. Shift every start/end in the second donor by one common integer-ms delta.
4. Sample delta uniformly from `[-first_start_ms,99999-last_start_ms]`.
5. Concatenate, sort starts and reaggregate ten buckets.

All starts stay in [0,100000), with no cropping or circular shift.
Long ends may extend past 100s. Preserve exported counters/durations and every natural-hybrid flow.
Equal start timestamps do not cause deduplication.

Use a new synthetic ID and group by `(sample_id,local_bucket_index)`, not original source IP.
The combination represents a virtual source. Real row/source/window IDs and offset stay in provenance.

Do not fill calendar gaps or require full occupancy.
Choose B uniformly among available active-bucket bands 1, 2-4, 5-10, then sources, then windows.
Exclude the other donor's source without unbounded rejection. Record occupancy/mixed buckets without filtering them.

## Counts and provenance

| Sample type | Count |
|---|---:|
| real_benign | N-P |
| real_malicious | P |
| synthetic_benign | P |
| synthetic_malicious | P |
| **Final train** | **N+2P** |

Positive count is 2P, positive share `2P/(N+2P)`. P counts multi-type original windows once.
Positive synthesis selects type/source/window uniformly at each level, then B.
Reuse is allowed; selected-type counts and actual memberships differ with multi-type A.
New backgrounds are not independent newly captured attacks.

Append to derived train only. Do not edit raw data, use held-out/synthetic donors or update P after augmentation.
Persist sample type, donor IDs, offset, requested attack, counts, shard and row.

## Completion

Final shards contain float32 X, uint8 mask/targets, int64 IDs and flow counts.
Raw shards retain float64 X_raw; manifest/event cache retain provenance.
Small Excel/English explanation/JSON/schema/scaler reports are Git-visible.

Check flow conservation, original split ownership, N+2P and 2P counts, train-only different-source donors,
valid shifts, ten positions per sample, finite values, empty zeros and matching shard/manifest/targets.
Source size/mtime must not change. See [README.md](README.md) for stage-level resume.
