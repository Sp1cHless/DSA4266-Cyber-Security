# Paired diagnostics for selecting superimposition

Historical comparison tool. Scheme2 is now selected. Retired Scheme1 outputs
must be rebuilt before rerunning; completed evidence and the final decision are
in [the comparison archive](reports/superimposition_comparison_20261022/SUMMARY.md).

After matched Scheme1, Scheme1 L2 and Scheme2 training/evaluation:

```powershell
.\.venv-cnn\Scripts\python.exe -u CIC_new\diagnose_superimposition.py
```

The latest completed L2 three-way report pins references. --l2-run selects it
explicitly; --out selects fresh output. Default:
outputs/diagnostics/<timestamp>_superimpose_choice_<suffix>/.
Any existing explicit output is refused. No fitting or test access occurs.
These are exploratory real-validation diagnostics after checkpoint selection.

- DECISION_DIAGNOSTICS.md: attack/composition/fraction/length/volume/protocol/DNS
  slices, paired detections and order sensitivity.
- slices.csv: all four thresholds and attack x composition/fraction tables,
  including example and source-host counts.
- paired_errors.csv: both, L2 only, Scheme2 only, or neither.
- cases.csv: every attack and union of false alarms at the 1% point, retaining
  all four threshold decisions. Cross-model logits are not calibrated confidence;
  benign percentiles are descriptive.
- positive_hosts.csv: recall by attack source host.
- paired_host_bootstrap.json: recall differences resampled by source host.
- temporal_order.csv: original/shuffled/reversed active content, preserving
  masks/padding.
- report.json/status.json: results, hashes, provenance and completion.

Thresholds are exploratory validation-ROC FPR budgets 1%/0.1%, plus previously
fixed train-benign 99%/99.9% quantiles. Subgroups do not refit thresholds.

Order diagnosis uses all validation attacks and up to 4096 fixed random benign
samples, with identical samples/permutations across models. Frozen readouts and
thresholds are reused. Available GPU inference at batch 1024 must reproduce the
original scores before perturbation; CPU fallback must pass the same check.
--skip-temporal profiles saved scores only; --benign-limit changes the sample.
The selected set has artificial prevalence; interpret group-specific rates and changes.

Profiles describe behavior, not causal feature importance or encoder additivity.
Sequence attack-flow fraction is not packet/byte severity or local bucket dilution.
Host-bootstrap intervals condition on fixed models/thresholds, not training-seed
uncertainty. Controlled dilution studies must explicitly disclose training-donor reuse.
