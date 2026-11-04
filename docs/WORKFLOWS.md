# Final Report to Repository Workflows

The repository contains independent workflows matching `DSA4266_Group3.pdf`.
An identical tensor shape does not imply an identical preprocessing or runtime
version. This organization does not combine the two preprocessing chains.

| Final report section | Study | Canonical code | Evaluation |
|---|---|---|---|
| III-IV | Source-host construction, bucket encoder and augmentation | Root `CIC_new/` | Training-only fitting; original real validation/test |
| V-A/B | Four-model comparison, circular shift and attack profiles | Root `train/` and `analysis/` | Five seeds 4266-4270; fixed real validation |
| V-C | Sparse Web evidence and temporal-scale study | Root temporal drivers in `CIC_new/` and `train/` | Availability anchors and separately rebuilt single-seed cohorts |
| VI | Control and A/B-1/B-2 training exclusions | [experiments/holdout](../experiments/holdout/README.md) | Fixed 100s/10s pipeline; LSTM; seeds 4266-4268; frozen historical-test comparisons |
| VII | Host-information and input-ambiguity diagnostics | Root host-adversarial and raw-flow audit code | Reused-training probes and inspected-validation diagnostics |

Root preprocessing/detector code retains the cleaned embedding-superposition
version. The holdout workflow retains its committed implementation in a separate
source root, including its matching common runtime, model configurations and
tests. Source duplication here preserves experiment versions; it is not a claim
that two implementations can be substituted within a completed run.

Use [the research index](research/README.md) for compact results and
[the code guide](code/README.md) for source inventories. Report tables are saved
with explicit transcription provenance. Raw arrays and complete historical runs
are local archives, and datasets are not bundled into Git.
