# CIC split feasibility audit

This EDA compares candidate 60/20/20 partitions of the original source-IP 100s sequences.
It does not choose a final split, edit the bucket builder, train a model, fit transforms, or synthesize data.
The full dataset is described to design an evaluation protocol; test scores are never used to select a seed or partition.
This is the pre-v2 comparison snapshot. Production now uses the separately approved coverage-constrained 15min
policy in `CIC_new/split_policy.py`; its full-data preflight is `CIC_new/reports/split_policy_v2_preflight.json`.
The production policy is distinct from the unconstrained `blocks_15m` candidate below.

## Full-data findings: 18 October 2026

The run reconciled **20,115,529 flows**, including **2,600,903 malicious flows**, against **1,466,665** nonempty 100s source windows.
The data has **11 observed UTC dates**. At a 300s gap there are **22 candidate attack episodes** across 14 types;
**11 types have only one episode**. At 60s/900s gaps there are 444/17 episodes, showing substantial sensitivity to the gap convention.
For example, sparse Web/XSS starts are fragmented at 60s but form long activities at 300s. Episode counts should not be interpreted as confirmed campaigns.

| Candidate | Attack types in train / validation / test | Cross-set 300s episodes | Excluded sequences |
|---|---|---:|---:|
| random 100s baseline | 14 / 14 / 14 | 18 | 0 |
| chronological | 12 / 1 / 2 | 1 | 2,208 |
| chronological whole dates | 12 / 1 / 2 | 0 | 376 |
| shuffled whole dates | 9 / 2 / 4 | 0 | 185 |
| global 15min blocks | 13 / 11 / 8 | 11 | 219,861 |
| global 30min blocks | 12 / 10 / 10 | 12 | 118,965 |
| global 60min blocks | 12 / 9 / 6 | 10 | 55,842 |
| episode-guarded 30min blocks | 12 / 2 / 2 | 0 | 49,225 |

The random baseline shares 3,849 global 100s timestamps across sets and splits 394,724 adjacent same-source window pairs.
Its 18 cross-set 300s episodes also all touch both train and test. These are measured dependence risks, not proof of duplicate packets.

Chronological windows and chronological dates leave **Bot and Infilteration unseen in train but present in test**.
Only Bot is strictly test-only; Infilteration also occurs in validation. The chronological window cut splits one Infilteration episode
between validation and test even after a 100s boundary guard. A finite guard does not ensure episode independence.

The guarded 30min candidate has zero cross-set episodes at all three gap thresholds in this run, and zero shared global timestamps,
adjacent cross-set source pairs or retained interval-envelope overlaps. It retains 1,417,440 sequences:
872,650 train / 306,126 validation / 238,664 test (61.56% / 21.60% / 16.84% of retained sequences).
However, its validation attacks are **Bot and LOIC-UDP**, neither present in training; test attacks are only **LOIC-HTTP and Infilteration**.
LOIC-HTTP has 66 train windows and 384 test windows; its largest episode, 99.23% of that type's flows, is in test.
Thus, zero unseen test types does not imply balanced attack coverage or a representative supervised training/evaluation protocol.

No candidate is selected automatically. A final protocol must state whether it targets unseen activities/types or within-activity detection.
An activity-held-out benchmark needs per-type coverage and unseen-type results reported explicitly; a broad within-activity benchmark
must acknowledge shared campaign patterns even with global block separation and purging. A second benchmark or grouped cross-validation
may be needed, but repeating folds cannot create independent activities for single-episode types.

## Inputs and outputs

Run `python CIC_new/EDA/04_audit_splits.py` from the repository root with NumPy installed.
The latest completed EDA 02 100s catalog is reused. `--sequences-dir`, `--db`, `--output`,
`--reports-dir`, `--seed`, `--embargo-seconds`, `--node`, and `--node-modules` override defaults.
The raw database fingerprint must match EDA 02. One read-only scan reconciles each sequence's flow counts,
attack counts, attack memberships, and first/last starts, and records actual maximum flow ends.

Heavy CSVs, manifests and previews are in ignored `EDA/outputs/04_splits_<timestamp>/`.
The final Excel, `results.json` and this guide are copied to Git-visible `CIC_new/reports/EDA/04_splits_<timestamp>/`.
`--render-only <existing output>` rebuilds the report without a raw rescan; a saved `report_config.json` is required.

## Candidate methods

| Candidate | Assignment unit and order |
|---|---|
| `random_100s` | Pre-v2 builder reference: complete source/IP windows, primary-attack stratification, seed 4266. No purge. |
| `chronological` | All sources at one global 100s timestamp stay together. Time-ordered cuts nearest cumulative 60% and 80% sequence mass. |
| `day_chronological` | Whole observed UTC dates, in chronological order. |
| `day_shuffled` | Whole observed UTC dates, in one seeded random order. |
| `blocks_15m`, `blocks_30m`, `blocks_60m` | Whole global UTC-aligned time blocks, in one seeded random order. All sources in a block stay together. |
| `guarded_30m` | Start with global 30min blocks; union every block spanned by each 300s-gap candidate attack episode. Shuffle the resulting indivisible groups. |

For grouped schemes, shuffle sorted group IDs using `numpy.random.default_rng(seed)` and choose two cuts nearest
cumulative 60% and 80% original-sequence counts, while leaving at least one group per set.
This is not a search for optimal category coverage and does not force all attack types into every set.
Final proportions change after exclusions. A date/block group can be large; exact 60/20/20 is not guaranteed.

The random reference reproduces the builder's stratum rounding. Primary attack is the type with most malicious flows,
ties broken by name; fewer than three windows in a primary stratum are train-only.

## Candidate attack episodes

All malicious flows of one attack type are combined across source IPs into observed 10s activity bins.
Each bin stores exact first/last starts, flow count and source-window memberships.
Start a new episode if the next bin's first malicious start is **more than** the chosen gap after the previous
activity's last start, or if the UTC date changes. Gaps of 60, 300 and 900 seconds are compared.
The detailed Episodes sheet uses 300 seconds.

These are **candidate contiguous activities**, not verified independent campaigns. Different episodes may share
tools, hosts or scripts. One long episode may contain multiple real activities. Day boundaries are a convention,
not proof that an attack could not continue overnight. Counts are sensitivity diagnostics, not independent-sample counts.
An attack sequence can belong to several episodes at a short threshold or to multiple attack types.

## Boundary exclusions and interval checks

All grouped candidates receive the same conservative exclusions; the random reference is deliberately unpurged.

1. Find every change in global fold assignment in the ordered observed 100s windows.
2. Place a default **100s guard on each side** of that transition. Exclude a complete sequence if its support intersects
   the guard band. Support begins at the window start and ends at the later of window end or maximum actual flow end + 1ms.
3. Also exclude a sequence if its inclusive interval envelope from its first actual flow start to its maximum actual
   flow end intersects any 100s window originally assigned to another fold.
4. Recompute isolation diagnostics on retained sequences. Excluded sequences are not reassigned elsewhere.

These rules are conservative: an envelope contains gaps, and another fold's full window need not have a flow at
the exact overlapping instant. An overlap is a dependence risk, not proof of shared packets or duplicate records.
Actual `flow_end_milliseconds` is used; exported duration is not assumed equal to end minus start.
The raw interval rule runs even when `--embargo-seconds 0` is specified.

## Workbook metrics

### Comparison

Each row is one candidate. Train/validation/test/excluded shares have **all original nonempty sequences** as denominator.
The four shares sum to one. Shares, type counts and positive rates are live Excel formulas backed by detail tables.

- **Test types unseen in train**: number of types with test windows > 0 and train windows = 0, regardless of validation.
- **Validation types unseen**: types with validation windows > 0 and train windows = 0.
- **Train/validation/test types**: distinct types with any retained windows in the named set. Low counts restrict what evaluation can establish.
- **Types in all three**: types with at least one retained window in each set; this does not establish independent activities.
- **Cross episodes: 300s**: candidate 300s-gap episodes with retained members in at least two sets.

### Composition and Daily

- **Sequences**: original nonempty source-IP 100s windows assigned to that set, or excluded.
- **Positive sequences**: windows with any malicious flow; includes natural hybrid windows.
- **Positive share**: positive sequences / sequences within that row; unavailable for an empty set.
- **Flows / attack flows**: raw flow starts in those windows; each flow is counted once per candidate.
- **Sources / UTC dates**: distinct sources/dates in that row; not additive across sets.
- **Original share**: sequences / all original sequences, including exclusions in the denominator.
- Daily reports the same additive counts by the window's UTC start date. Empty dates are not invented.

### Attack coverage

One row per candidate and attack type, using **all type memberships**, not just the primary stratum.
Workbook flag columns use numeric 1=true and 0=false; JSON preserves Boolean values.

- **Train/validation/test/excluded windows**: windows containing at least one flow of that type.
- **Flows**: exact malicious flow counts of that type, not all flows in its windows.
- **Sources**: distinct malicious source IPs for that type within the set.
- **Episodes**: candidate 300s-gap episodes of that type touching the set; an episode can count in multiple sets.
- **Test unseen in train**: test > 0 and train = 0.
- **Test only**: test > 0 and both train and validation = 0.
- **Validation unseen**: validation > 0 and train = 0.
- **All three sets**: train, validation and test each contain that type.

Type-level window counts are not additive because a sequence can contain multiple attack types.
Unseen types remain valid binary-detection evaluation cases, but must be reported separately from seen-type cases.

### Activity and Episodes

- **Gap threshold (s)**: maximum inter-bin attack-start inactivity tolerated within an episode.
- **Candidate episodes**: episode count for that type and threshold, across observed dates.
- **UTC dates / sources / 100s windows**: distinct coverage for that type; window counts do not grow by counting memberships repeatedly.
- **First/last start (UTC)**: earliest/latest malicious flow start; not connection end.
- **Largest episode flow share**: largest episode's malicious flows / all malicious flows of that type.
- **Episode ID**: stable within this run's 300s definition, assigned in attack-name/time order.
- **Start span (s)**: last malicious start minus first; not connection duration.
- **Active 10s bins**: global type/time bins with at least one malicious start, irrespective of source count.

### Isolation

- **Shared global 100s**: global 100s timestamps containing retained source windows in multiple sets; measures same-time cross-set exposure.
- **Adjacent source pairs crossing sets**: consecutive 100s windows of the same source with both retained but assigned to different sets.
  Windows separated by missing periods are not adjacent for this metric.
- **Interval overlap windows**: retained sequence envelopes intersecting another set's retained 100s window; conservative temporal check.
- **Shared train/test sources**: unique source IPs in both sets. This is descriptive and not itself proof of leakage.
- **Cross episodes: 60/300/900s**: episodes under each threshold with retained members in multiple sets.
- **Train/test episodes: 300s**: episodes touching both train and test specifically.
- **Retained/excluded sequences**: mutually exclusive counts summing to the original total.
- **Embargo/interval exclusions**: each exclusion rule's count before union. They may overlap; do not add them.
- **Fold transitions**: changes between successive observed global window assignments before exclusions; unavailable for the random baseline.

Additional machine-readable diagnostics include `affected_attack_windows_<gap>s`: distinct retained attack windows
belonging to any episode that crosses sets under that gap. `episode_splits.csv` gives the 300s episode/set memberships.

## Candidate manifests and limits

`candidate_assignments.npz` is loadable with `numpy.load(..., allow_pickle=False)`.
Each row is keyed by `(source_names[source_id], start_ms)`. Candidate arrays use
0=train, 1=validation, 2=test, 3=excluded; `<candidate>_before_purge` preserves its original assignment.
`first_ms` and `max_end_ms` retain the verified interval envelope. Array positions are local catalog indices,
not promised to be builder sequence IDs; join by source and start when integrating a selected split.

Raw rows map to exactly one original source/start window. This EDA does **not** audit exact full-row duplicate
content or bidirectional/session identity; row disjointness alone does not establish content independence.
No candidate is certified leakage-free. These candidate manifests are exploratory and are not the production split.
The approved production policy separately fixes grouping, coverage reservations and exclusion rules in `CIC_new/split_policy.py`.
It evaluates held-out time blocks in the same environment and explicitly allows shared source/campaign patterns.
