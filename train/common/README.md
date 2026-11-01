# Shared Detector Runtime

This package is shared by the four baseline families and selected extensions.
Its responsibilities differ from the input bucket encoder in `CIC_new/`.

| Module | Responsibility |
|---|---|
| [data.py](data.py) | Validate delivered Scheme2 shards, construct disposable caches, split views, four-group weights and flattened tree inputs |
| [experiment.py](experiment.py) | Common training CLI, adapter/device selection, run metadata and evaluation exports |
| [torch_runner.py](torch_runner.py) | Neural model dispatch, deterministic setup, optimization, AP checkpointing, prediction and loading |
| [networks.py](networks.py) | Shared sequence inputs, mask-aware pooling and circular rotation |
| [metrics.py](metrics.py) | Score checks, global validation thresholds and common unweighted evaluation metrics |
| [io.py](io.py) | JSON, hashes, runtime records and fresh run-directory creation |
| [temporal.py](temporal.py) | Input-order policy descriptions and rotation validation |
| [temporal_experiment.py](temporal_experiment.py) | Rebuilt-window experiment orchestration, matched-reference checks and subtype summaries |
| [temporal_storage.py](temporal_storage.py) | Temporal-stage storage estimation and reviewed disposable-cache handling |
| [remove_temporal_cache.ps1](remove_temporal_cache.ps1) | Windows cache removal with owner/path checks; called by the storage helper |

In `io.py`, `ROOT` denotes the **train directory** and `PROJECT` denotes the
repository root. They are different path anchors. Neural models are loaded
dynamically from `train.<name>.model`; XGBoost and bucket supervision have their
own adapters. The host-adversarial experiment uses a separate specialized loop.

See the [training protocol](../README.md) and the
[complete implementation map](../../docs/code/README.md). Keep inputs, outputs
and immutable completed experiments separate; caches are disposable, source
datasets and model results are not.
