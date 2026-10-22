"""Train all neural architectures with the same objective and early stopping."""
import importlib
import os
import random
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score

from .io import write_json


def setup(seed, threads, device):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if device == "mps":
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    return device


def build(name, config, device):
    return importlib.import_module(f"train.{name}.model").Model(**config["architecture"]).to(device)


def tensors(split, indices, device):
    return (torch.as_tensor(np.array(split["x"][indices], copy=True), device=device),
            torch.as_tensor(np.array(split["mask"][indices], copy=True), device=device))


@torch.no_grad()
def predict(model, split, device, batch_size=2048):
    model.eval()
    result = np.empty(len(split), dtype=np.float64)
    for start in range(0, len(split), batch_size):
        end = min(start + batch_size, len(split))
        x, mask = tensors(split, slice(start, end), device)
        result[start:end] = torch.sigmoid(model(x, mask)).cpu().numpy()
    return result


def fit(name, training, validation, weights, config, device, threads, seed, out):
    setup(seed, threads, device)
    model = build(name, config, device)
    settings = config["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"])
    rng = np.random.default_rng(seed)
    history = []
    best, waiting, best_epoch = -1.0, 0, 0
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        total_loss = 0.0
        order = rng.permutation(len(training))
        for start in range(0, len(order), settings["batch_size"]):
            ids = order[start:start + settings["batch_size"]]
            x, mask = tensors(training, ids, device)
            y = torch.as_tensor(np.array(training["y"][ids], dtype=np.float32), device=device)
            w = torch.as_tensor(weights[ids], device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, mask)
            loss = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * w).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(ids)
        scores = predict(model, validation, device)
        ap = float(average_precision_score(validation["y"], scores))
        history.append({"epoch": epoch, "weighted_train_bce": total_loss / len(training), "validation_ap": ap})
        print(f"{name} epoch={epoch} loss={total_loss / len(training):.6f} validation_AP={ap:.6f}", flush=True)
        if ap > best:
            best, waiting, best_epoch = ap, 0, epoch
            torch.save({"model_name": name, "architecture": config["architecture"],
                        "state_dict": model.state_dict(), "best_epoch": epoch}, out / "model.pt")
        else:
            waiting += 1
        write_json(out / "history.json", history)
        if waiting >= settings["patience"]:
            break
    saved = torch.load(out / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(saved["state_dict"])
    return model, {"best_epoch": best_epoch, "best_validation_ap": best,
                   "epochs_completed": len(history), "parameters": sum(p.numel() for p in model.parameters())}


def load(name, config, path, device):
    model = build(name, config, device)
    saved = torch.load(path, map_location=device, weights_only=True)
    if saved["model_name"] != name or saved["architecture"] != config["architecture"]:
        raise ValueError("Checkpoint/config mismatch")
    model.load_state_dict(saved["state_dict"])
    return model
