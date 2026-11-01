# Context Interventions and Attention Pooling

Report sections: 6.3. Organized on 1 November 2026 from completed project records.

## Finding

Learned attention increases attack-bearing-bucket weight in context-bearing Web cases, while sequence discrimination changes differently across occupancy groups. Same-runtime AP changes from 61.53% to 61.87%; Web recall changes from 25.97% to 29.28% at the 1% budget and from 3.87% to 2.21% at 0.1%.

## Evidence scope

A single retrained seed-4266 attention model versus the same-runtime reference. Label-based frozen context interventions are oracle diagnostics with unchanged thresholds. The full sequence metrics are consolidated in Table V1; the duplicate performance chart remains local.

## Presentation

The current report uses concise prose or a consolidated table for this study. Earlier standalone figures and insertion drafts remain local rather than duplicating the main evidence.

## Compact results

- [figure_provenance.json](<results/figure_provenance.json>)

The shared intervention comparison is [Table V1](<../results/table_v1_targeted_comparison.csv>).

## Code and protocol entry points

- [train/LSTM_CONTEXT_ABLATION.md](<../../../../train/LSTM_CONTEXT_ABLATION.md>)
- [train/run_lstm_context_ablation.py](<../../../../train/run_lstm_context_ablation.py>)
- [analysis/diagnose_lstm_context.py](<../../../../analysis/diagnose_lstm_context.py>)

## Local supporting records

`local/` preserves the original dated insertion drafts, plotting workspaces, source screenshots and superseded displays for this topic. It is ignored by Git. These files are historical support; the current DOCX determines final section placement. Full experiment runs, arrays and checkpoints retain their original paths under the ignored runtime directories. Source paths and hashes in captured provenance describe the original records, not bundled datasets.

## Saved evidence locations

These original records retain their existing paths; full runs and private evidence maps are intentionally local.

- `train/runs/20261025_174735_463920_lstm_context_interventions_956fa8/FINDINGS.md`
- `train/runs/20261025_175604_000150_lstm_attention_pooling_seed4266_215706/RESULTS.md`
- `train/runs/20261025_175604_000150_lstm_attention_pooling_seed4266_215706/attack_comparison.csv`
- `train/runs/20261025_175604_000150_lstm_attention_pooling_seed4266_215706/comparison.json`
