# Context Interventions and Attention Pooling

## Finding

Learned attention increases attack-bearing-bucket weight in context-bearing Web cases, while sequence discrimination changes differently across occupancy groups. Same-runtime AP changes from 61.53% to 61.87%; Web recall changes from 25.97% to 29.28% at the 1% budget and from 3.87% to 2.21% at 0.1%.

## Evidence scope

A single retrained seed-4266 attention model versus the same-runtime reference. Label-based frozen context interventions are oracle diagnostics with unchanged thresholds. The full sequence metrics are consolidated in Table V1; the duplicate performance chart remains local.

## Compact results

- [attack_comparison.csv](results/attack_comparison.csv)

[Shared single-seed comparison](../results/table_v1_targeted_comparison.csv).

## Code and protocol entry points

- [train/LSTM_CONTEXT_ABLATION.md](<../../../../train/LSTM_CONTEXT_ABLATION.md>)
- [train/run_lstm_context_ablation.py](<../../../../train/run_lstm_context_ablation.py>)
- [analysis/diagnose_lstm_context.py](<../../../../analysis/diagnose_lstm_context.py>)

## Local evidence

Completed runs and historical working material are consolidated in the ignored
`archive/` directory. Restore the matching inputs before rerunning a diagnostic
that requires saved arrays or checkpoints. See the [research index](../../README.md) for the result catalog and evidence boundaries.
