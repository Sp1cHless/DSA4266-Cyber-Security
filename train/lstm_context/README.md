# Attention Pooling Extension

[model.py](model.py) subclasses the original LSTM and replaces uniform masked
mean pooling with learned attention. Masked maximum pooling and the final state
are retained. Zero-initialized attention logits start with uniform weighting.

The experiment entry point is [run_lstm_context_ablation.py](../run_lstm_context_ablation.py),
which validates a completed same-runtime reference before fitting a fresh model.
The model is selected through the shared neural runner with name `lstm_context`.
The frozen oracle context diagnostics are a separate script in
`analysis/diagnose_lstm_context.py`.

Read [LSTM_CONTEXT_ABLATION.md](../LSTM_CONTEXT_ABLATION.md) for the method and
[the study page](../../docs/research/07_targeted_studies/attention_pooling/README.md)
for completed findings. Full runs remain local; this directory contains the
implementation, not a baseline checkpoint or an independent test result.
