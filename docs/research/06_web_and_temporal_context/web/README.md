# Sparse Web Evidence

## Finding

Of 181 Web validation sequences, 171 contain attack flows in one bucket and 96 have only one active traffic bucket. XGBoost has higher mean recall on the single-active-bucket group; LSTM is stronger on the two-bucket and three-or-more groups. Local and recurrent models show complementary detection profiles.

## Evidence scope

Observational groups with the original masks and five original-order seeds. Each fit uses its own global validation threshold. Occupancy is not an experimentally randomized factor.

## Compact results

- [cohort_profiles.csv](results/cohort_profiles.csv)
- [past_context_comparison.csv](results/past_context_comparison.csv)
- [web_recall_by_occupancy.csv](results/web_recall_by_occupancy.csv)
- [web_recall_per_seed.csv](results/web_recall_per_seed.csv)

## Code and protocol entry points

- [analysis/diagnose_web_temporal_structure.py](<../../../../analysis/diagnose_web_temporal_structure.py>)
- [train/LSTM_WEB_FINDINGS.md](<../../../../train/LSTM_WEB_FINDINGS.md>)

## Local evidence

Completed runs and historical working material are consolidated in the ignored
`archive/` directory. Restore the matching inputs before rerunning a diagnostic
that requires saved arrays or checkpoints. See the [research index](../../README.md) for the result catalog and evidence boundaries.
