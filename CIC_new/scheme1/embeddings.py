"""Independent masked-field prediction learns only L7/source/destination tables.

The prediction MLP is discarded. This is neither the final AE nor a whole-bucket
embedding. Sparse weighted bags avoid allocating a bucket-by-65k port matrix.
"""
from __future__ import annotations

import logging
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .statistics import LEARNED
from .storage import arrays, digest, identity, read_json, write_json

LOG = logging.getLogger(__name__)


def sparse_batch(pool, field, rows, device):
    boundaries = pool[field + "_offsets"]
    indices, weights, lengths = [], [], []
    for row in rows:
        lo, hi = boundaries[row:row + 2]
        indices.append(pool[field + "_indices"][lo:hi])
        weights.append(pool[field + "_weights"][lo:hi])
        lengths.append(hi - lo)
    return (torch.as_tensor(np.concatenate(indices), dtype=torch.long, device=device),
            torch.as_tensor(np.concatenate(([0], np.cumsum(lengths))), dtype=torch.long, device=device),
            torch.as_tensor(np.concatenate(weights), dtype=torch.float32, device=device))


class CategoryLearner(nn.Module):
    def __init__(self, slots, dimensions, fixed_width):
        super().__init__()
        self.tables = nn.ModuleDict({field: nn.EmbeddingBag(slots[field], dimensions[field],
                                    mode="sum", padding_idx=0, include_last_offset=True) for field in LEARNED})
        self.context = nn.Sequential(nn.Linear(82 + fixed_width + sum(dimensions.values()) + 3, 128),
                                     nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.heads = nn.ModuleDict({field: nn.Linear(64, dimensions[field]) for field in LEARNED})
        self.bias = nn.ParameterDict({field: nn.Parameter(torch.zeros(slots[field])) for field in LEARNED})

    def forward(self, numeric, fixed, bags, hidden):
        vectors = []
        flags = numeric.new_zeros((len(numeric), 3))
        for j, field in enumerate(LEARNED):
            vector = self.tables[field](bags[field][0], bags[field][1], per_sample_weights=bags[field][2])
            if field == hidden:
                vector = vector * 0
                flags[:, j] = 1
            vectors.append(vector)
        context = self.context(torch.cat((numeric, fixed, *vectors, flags), dim=1))
        # Tied input/output tables give the hidden category a direct learning signal.
        return F.linear(self.heads[hidden](context), self.tables[hidden].weight, self.bias[hidden])


def batch_loss(model, pool, rows, field, device):
    bags = {f: sparse_batch(pool, f, rows, device) for f in LEARNED}
    numeric = torch.as_tensor(pool["numeric"][rows], device=device)
    fixed = torch.as_tensor(pool["fixed"][rows], device=device)
    logits = model(numeric, fixed, bags, field)
    indices, offsets, weights = bags[field]
    owners = torch.repeat_interleave(torch.arange(len(rows), device=device), offsets[1:] - offsets[:-1])
    loss = numeric.new_zeros(len(rows))
    loss.index_add_(0, owners, -F.log_softmax(logits, dim=-1)[owners, indices] * weights)
    eligible = offsets[1:] > offsets[:-1]
    if not eligible.any():
        return None, 0
    return loss[eligible].mean(), int(eligible.sum())


def train(prepared, out, *, steps=3000, batch_size=128, validate_every=200,
          l7_dim=8, src_port_dim=8, dst_port_dim=8, seed=4266, device="auto", cpu_threads=2, lr=.001):
    prepared, out = Path(prepared).resolve(), Path(out).resolve()
    manifest = read_json(prepared / "manifest.json")
    if not manifest.get("complete") or manifest.get("version") != "scheme1-prepared-v1":
        raise ValueError("Completed scheme1 preparation required")
    if min(steps, batch_size, validate_every, l7_dim, src_port_dim, dst_port_dim, cpu_threads) < 1 or lr <= 0:
        raise ValueError("Training settings must be positive")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Fresh embedding directory required: {out}")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu")
    torch.set_num_threads(cpu_threads)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    schema = read_json(prepared / "schema.json")
    train_pool = arrays(prepared / "embedding_pools" / "train.npz")
    val_pool = arrays(prepared / "embedding_pools" / "validation.npz")
    eligible = {field: np.flatnonzero(np.diff(train_pool[field + "_offsets"]) > 0) for field in LEARNED}
    if any(not len(rows) for rows in eligible.values()):
        raise ValueError("Training pool must contain L7 and TCP/UDP port observations; increase --train-pool if necessary")
    minimum_steps = 3 * int(np.ceil(len(train_pool["numeric"]) / batch_size))
    requested_steps, steps = steps, max(steps, minimum_steps)
    orders = {field: rng.permutation(rows) for field, rows in eligible.items()}
    cursors = {field: 0 for field in LEARNED}
    dimensions = dict(l7=l7_dim, src_port=src_port_dim, dst_port=dst_port_dim)
    slots = {field: len(schema["vocabularies"][field]) + 1 for field in LEARNED}
    model_args = dict(slots=slots, dimensions=dimensions, fixed_width=train_pool["fixed"].shape[1])
    model = CategoryLearner(**model_args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    out.mkdir(parents=True, exist_ok=True)
    config = dict(complete=False, prepared_identity=identity(prepared), schema_sha256=digest(prepared / "schema.json"),
                  dimensions=dimensions, slots=slots, seed=seed, device=device, steps=steps,
                  requested_steps=requested_steps, minimum_full_pass_steps=minimum_steps, batch_size=batch_size,
                  validate_every=validate_every, lr=lr, cpu_threads=cpu_threads,
                  objective="mask one learned field; predict its original distribution using tied category tables and numerical/fixed/other-field context",
                  fitting_population="real retained training active buckets: uniform reservoir plus all-category witnesses; benign and malicious",
                  validation_population="real validation active buckets; checkpoint selection only", test_used=False,
                  training_buckets=len(train_pool["numeric"]),
                  training_positive_sequence_buckets=int(train_pool["label"].sum()),
                  train_pool_sha256=digest(prepared / "embedding_pools" / "train.npz"),
                  validation_pool_sha256=digest(prepared / "embedding_pools" / "validation.npz"))
    write_json(out / "manifest.json", config)

    @torch.no_grad()
    def evaluate():
        model.eval()
        scores = {}
        for field in LEARNED:
            total, count = 0., 0
            for start in range(0, len(val_pool["numeric"]), batch_size):
                rows = np.arange(start, min(start + batch_size, len(val_pool["numeric"])))
                loss, n = batch_loss(model, val_pool, rows, field, device)
                if n:
                    total += float(loss) * n
                    count += n
            scores[field] = total / count if count else None
        valid = [value for value in scores.values() if value is not None]
        if not valid:
            raise ValueError("No eligible validation categories")
        return float(np.mean(valid)), scores

    initial, _ = evaluate()
    best, history, running = float("inf"), [], []
    coverage = {field: np.zeros(slots[field], bool) for field in LEARNED}
    begun = time.monotonic()
    LOG.info("Category fitting: %s buckets; requested %s steps, running %s to guarantee one full pass",
             len(train_pool["numeric"]), requested_steps, steps)
    for step in range(1, steps + 1):
        model.train()
        field = LEARNED[(step - 1) % len(LEARNED)]
        if cursors[field] == len(orders[field]):
            orders[field] = rng.permutation(eligible[field])
            cursors[field] = 0
        stop = min(cursors[field] + batch_size, len(orders[field]))
        rows = orders[field][cursors[field]:stop]
        cursors[field] = stop
        loss, _ = batch_loss(model, train_pool, rows, field, device)
        if loss is None or not torch.isfinite(loss):
            raise ValueError("Nonfinite category prediction loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for module in model.tables.values():
            module.weight.grad[0] = 0
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        offsets = train_pool[field + "_offsets"]
        for row in rows:
            lo, hi = offsets[row:row + 2]
            coverage[field][train_pool[field + "_indices"][lo:hi]] = True
        running.append(float(loss.detach().cpu()))
        if step % validate_every == 0 or step == steps:
            value, field_scores = evaluate()
            if not np.isfinite(value):
                raise ValueError("Nonfinite category validation loss")
            history.append(dict(step=step, train_loss=float(np.mean(running)), validation_loss=value,
                                validation_by_field=field_scores))
            running.clear()
            if step >= minimum_steps and value < best:
                best = value
                tables = {f: model.tables[f].weight.detach().cpu().numpy().copy() for f in LEARNED}
                for table in tables.values():
                    table[0] = 0  # UNK has no training observations and is a deterministic zero vector.
                np.savez_compressed(out / "embeddings.npz", **tables)
                torch.save(dict(model_args=model_args, model_state=model.state_dict(), step=step), out / "best.pt")
            write_json(out / "history.json", history)
            LOG.info("Category embedding %s/%s: validation %.6f (%s)", step, steps, value, field_scores)
    config.update(complete=True, frozen=True, embedding_sha256=digest(out / "embeddings.npz"),
                  initial_validation_loss=initial, best_validation_loss=best, seconds=time.monotonic() - begun,
                  training_categories_covered={field: int(values[1:].sum()) for field, values in coverage.items()})
    if any(not values[1:].all() for values in coverage.values()):
        raise AssertionError("Some training categories did not receive target observations")
    write_json(out / "manifest.json", config)
    return config


