# NF-UNSW-NB15-v3 temporal EDA

This workflow performs a direction-neutral audit of the real temporal fields in
NF-UNSW-NB15-v3. It does not fit models or alter the existing UNSW EDA script.

## Input

Download **NF-UNSW-NB15-v3** from the University of Queensland ML-Based NIDS
Datasets page. The required files are:

- `NF-UNSW-NB15-v3.csv`
- `NetFlow_v3_Features.csv`

The large CSV is deliberately excluded from Git by `.gitignore`.

## Run

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-temporal-eda.txt

.venv/bin/python eda/eda_nf_unsw_nb15_v3.py \
  --input datasets/nf_unsw_nb15_v3/NF-UNSW-NB15-v3.csv \
  --feature-dictionary datasets/nf_unsw_nb15_v3/NetFlow_v3_Features.csv \
  --output-dir output/temporal_eda
```

The default implementation reads the complete CSV in 200,000-row chunks and
uses a small deterministic sample only for expensive ECDF plots. Counts,
timestamp validation, time aggregation, gaps, sessions, and chronological split
tables use the full dataset.

Use `--bucket 5min` or another pandas time frequency to override the automatic
plotting resolution. Use `--session-gap-minutes` to change the definition of a
new capture session.

## Main outputs

- `temporal_eda_report.md`: concise findings and cautions
- `dataset_summary.json`: machine-readable audit
- `tables/schema_and_quality.csv`: feature schema and data-quality checks
- `tables/capture_sessions.csv`: discontinuous capture periods
- `tables/chronological_split_class_counts.csv`: candidate split feasibility
- `tables/chronological_split_distribution_shift.csv`: pairwise split-drift metrics
- `plots/class_distribution.png`
- `plots/traffic_and_attack_rate_over_time.png`
- `plots/attack_categories_over_time.png`
- `plots/duration_and_iat_ecdf.png`
- `plots/chronological_split_class_composition.png`
- `plots/chronological_split_distribution_shift.png`

The generated report includes the insight and modeling implication for every
plot, so a second plot-specific README is intentionally not generated. Detailed
minute-level aggregation tables are kept in memory for plotting rather than
written as redundant multi-megabyte CSV files. Generated metadata records only
the input filename; it never stores a contributor's absolute filesystem path or
username.

The proposed chronological split is exploratory. It should be finalized only
after checking class coverage and agreeing on the eventual modeling question.
