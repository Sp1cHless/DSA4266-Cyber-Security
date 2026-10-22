# Controlled background-flow dilution

Historical stress diagnostic. Scheme2 is now selected. Retired Scheme1 outputs
must be rebuilt before rerunning. This diagnostic does not replace the completed
real-test comparison in [the archive](reports/superimposition_comparison_20261022/SUMMARY.md).

Run an additional stress diagnosis of the existing frozen Scheme1 L2 and Scheme2
models from the repository root:

```powershell
.\.venv-cnn\Scripts\python.exe -u CIC_new\diagnose_controlled_dilution.py
```

This is a train-derived stress test, not an independent-donor validation/test.
The original models already had access to all real training originals. New pairs
and selected flow subsets cannot retrospectively make their donors unseen. No
retraining, probe fitting, threshold selection or validation/test flow reads occur.
Existing validation reports/predictions are read to identify and verify references.

## Controlled construction

- Sample up to 20 original training sequences per attack type, balancing source
  hosts before reusing hosts. For rare types, use all available sequences up to
  that limit. Read exact attack labels from original rows, rather than trusting
  sequence primary labels.
- Fix min(5, available class flows) distinct attack rows per pair. This retains
  low-flow types such as XSS/Web/SQL. The budget is fixed across ratios within a
  pair; budgets can differ across pairs/types.
- Select a different-source benign training donor with enough distinct rows.
  Use nested subsets to achieve exactly 1%, 5%, 10%, 25%, 50% attack flow count.
  Include the same selected fragment at 100% attack (no background) to distinguish
  dilution effects from failure to recognize the weak attack fragment itself.
  Preserve row counters, duration and local time; zero displacement, no duplicated
  rows, no rescaling of per-flow byte counts.
- Replace attacks with the same number of rows from a third benign source for
  equal-total-flow benign controls. These do not match packet or byte totals.
- Exclude pairs already present in the original training donor plan. This does
  not eliminate training-flow reuse.
- Record temporal overlap and local attacked-bucket fractions. Some attack and
  benign flows may occupy different buckets despite a low sequence-level ratio.

## Three routes

1. `scheme1_l2_physical`: aggregate the actual union, build original field
   features, scale, apply per-bucket L2, then frozen temporal AE/readout.
2. `scheme2_physical`: aggregate the same actual union and use the frozen bucket
   encoder, count channel and temporal AE/readout. This is Scheme2's representation
   of realistic mixed traffic.
3. `scheme2_vector_add`: encode donors separately, add/unit-normalize embeddings,
   append the same total-flow channel, then use the same frozen Scheme2 model.
   This is an augmentation-space proxy, not a realistic-traffic test result.

Comparing routes 2/3 exposes differences between E(C(A,B)) and normalize(E(A)+E(B))
without retraining the encoder. Cosine similarity and per-case score/decision
changes describe the mismatch; they do not establish causal augmentation benefit.

## Thresholds and outputs

Both previously fitted real-training benign score thresholds (99th and 99.9th
percentiles) stay fixed. They do not impose identical stress alarm rates. Report
recall together with matched benign alarm rate; do not interpret the latter as
population FPR or compare precision at the artificial stress prevalence.

Unique output: `outputs/diagnostics/<timestamp>_controlled_dilution_<suffix>/`.
Use `--l2-run`, `--pairs-per-attack`, `--seed` or a fresh `--out` to pin inputs.

- `CONTROLLED_DILUTION.md`: coverage, both thresholds and per-type curves/tables.
- `cases.csv`: every ratio, donor, local overlap, cosine, logit and decision.
- `baseline_transitions.csv`: detections retained/lost/gained relative to the
  identical selected fragment without background. Undefined retention is reported
  when the baseline detects no fragment.
- `slices.csv`: all types/ratios/routes/thresholds and overlap-only views.
- `plan.json` / `donor_provenance.json`: original sequence IDs and exact selected
  raw row IDs, including nested backgrounds and matched benign replacements.
- `stress_inputs.npz` / `scores.npz`: reproducible prepared inputs and scores.
- `report.json` / `status.json`: hashes, unchanged-source/model audit and completion.

Raw-flow encodings first reproduce full original donor inputs in both pipelines.
Every stress case checks exact counts/labels/ratios, padding, finite values and
agreement of flow concatenation with Scheme1's statistical composition. No model,
vocabulary, scaler or existing dataset is overwritten.

Small selected attack budgets can remove campaign structure. Increasing benign
flow count also changes volume and possibly overlap. Results characterize these
weak fragments; they are not a complete attack-intensity or unseen-host study.
