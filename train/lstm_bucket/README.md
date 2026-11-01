# Auxiliary Bucket Supervision Extension

| File | Role |
|---|---|
| [model.py](model.py) | Original sequence classifier plus one linear head for each recurrent state |
| [runner.py](runner.py) | Verified bucket targets, active-position auxiliary loss, sequence training and post-selection bucket diagnostics |
| [config.json](config.json) | Fixed architecture/training settings and auxiliary loss weight |
| [train.py](train.py) | Direct shared-runtime entry point under model name `lstm_bucket` |

Use [run_lstm_bucket_ablation.py](../run_lstm_bucket_ablation.py) for the recorded
matched-reference comparison. It wraps the direct training entry point with
dataset/reference checks and a paired result summary. These entry points have
different responsibilities.

Bucket labels are targets only. Sequence validation AP selects the checkpoint;
bucket localization metrics are descriptive diagnostics and do not replace the
sequence score or threshold policy. The input encoder remains frozen.

Read [LSTM_BUCKET_SUPERVISION.md](../LSTM_BUCKET_SUPERVISION.md) for the protocol
and [the study page](../../docs/research/07_targeted_studies/bucket_supervision/README.md)
for completed results. Generated arrays and weights remain under ignored run
directories.
