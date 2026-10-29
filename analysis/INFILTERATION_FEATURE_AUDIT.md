# Infilteration feature audit

Run from the repository root with the existing environment:

```powershell
.\.venv-cnn\Scripts\python.exe -u analysis/audit_infilteration_features.py
```

The script reads the original completed 100s/10s preparation, its staged flow
row references and the source SQLite database in read-only mode. It creates a
unique output directory in `train/runs/`; no completed artifacts are overwritten.
Only selected train flows and previously identified validation collision-pair
flows are fetched from the source database. Test flow records are not queried.

## Diagnostic cohort

All real training sequences whose primary label is `Infilteration` are included.
Benign controls combine 6,000 fixed-seed random real training sequences with
same-source-host, same-active-bucket-count, same-floor-log2-flow-count matches.
Each positive draws at most two matched controls; repeated controls are deduplicated.
This is an artificial diagnostic population, not natural deployment prevalence.

Original numeric aggregates and protocol/L7/service histograms are rebuilt on
the unchanged ten-bucket grid. This pre-encoder representation is the reference
for the probes; it is not the LSTM or frozen Scheme2 embedding representation.

## Fixed candidate families

- `iat`: eight directional packet-interarrival fields, summarized by mean, max
  and positive fraction within each bucket. Zero semantics remain unverified.
- `rates`: four directional second-bytes/throughput fields with the same summaries.
- `ttl`: minimum/maximum TTL summaries, evaluated separately for shortcut sensitivity.
- `distribution`: per-flow volume/duration dispersion and quantiles, zero-reply
  and short-duration fractions, directional byte share, bytes per packet,
  destination-port-pair diversity, and omitted longest/shortest packet lengths.
- `categories`: predefined DNS query-type and ICMP IPv4-type fractions, retaining
  category meaning that diversity statistics alone cannot preserve.

Destination identities are used only for within-sequence equality/diversity,
never as numeric model inputs. Source identities are used only for matching and
group splits. Absolute timestamps, row IDs, labels and attack names are excluded
from the feature matrices. Completed-flow summaries retain the original offline
information-availability assumption; they do not establish real-time detection.

## Train-only screening

Fixed seed-4266 GroupShuffleSplit screens hold out either 15-minute start-time
blocks or entire source hosts. The time-block screening split has no additional
internal embargo/envelope purge and can retain host overlap. The host split
has no source-host overlap. Both reuse the original training data and are
development diagnostics rather than independent held-out validation results.

All probes are fixed ExtraTrees classifiers: 64 trees, depth 12, minimum leaf
8, balanced class weights, square-root feature subsampling, seed 4266, four CPU
threads. Old aggregates, each added family, robust combined candidates and all
candidates use the same settings. Extra dimensions can affect fitting behavior;
an AUC difference does not isolate a causal feature contribution.

Univariate directions and rankings use only the internal fitting subset.
Original category vocabularies were fitted on the full original real training
set and are retained, including for internal probe holdouts. No encoder, LSTM,
synthetic augmentation or operating threshold is trained here.

## Validation-selected collision diagnostic

The earlier pure-Infilteration audit identified ten missed and ten detected
samples with exact benign counterparts. One previously selected counterpart
per sample is checked under the additional features. This separate analysis
reuses validation selections, does not select probe features, and does not
re-search the entire benign validation population. Pair separation is not a
claim that a sample is now detectable at a fixed FPR.

## Outputs

- `REPORT.md`: probe AUCs, field availability, collision-pair diagnostics and limits.
- `training_cohort.csv` and `internal_partitions.csv`: exact sampling and grouping.
- `omitted_field_quality.csv`: nonfinite/negative/zero/positive counts and ranges.
- `univariate_screening.csv`: fit-directed train-only holdout feature diagnostics.
- `probe_results.csv`: fixed-family results on both internal group splits.
- `training_feature_cache.npz`: selected pre-encoder and candidate feature matrices.
- `candidate_definitions.json`, `provenance.json`, `status.json`: definitions,
  source/software identities, limitations and completion state.

Focused artificial checks:

```powershell
.\.venv-cnn\Scripts\python.exe -m unittest train.tests.test_infilteration_features -v
```
