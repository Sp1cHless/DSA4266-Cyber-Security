"""Original optimizer, batches, AP selection and stopping with one auxiliary loss."""
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score

from train.common.io import read_json, write_json, sha256
from train.common.torch_runner import setup, build, tensors, predict, load

REDUCTION = "active_bucket_mean_per_sequence_then_existing_group_weights"


def load_bucket_targets(dataset, split_name, split):
    """Read only this split's targets; do not modify the shared input cache."""
    if split_name not in ("train", "validation"):
        raise ValueError("Bucket-supervision fitting/diagnostics allow train and validation only")
    dataset = Path(dataset)
    manifest = read_json(dataset / "manifest.json")
    records = [r for r in manifest["shards"] if Path(r["path"]).parts[0] == split_name]
    targets = np.empty(split["mask"].shape, np.uint8)
    offset = 0
    for record in records:
        file = dataset / record["path"]
        if file.resolve().parent != (dataset / split_name).resolve():
            raise ValueError("Unexpected target shard path")
        if sha256(file) != record["sha256"]:
            raise ValueError("Bucket target shard changed: " + str(file))
        with np.load(file, allow_pickle=False) as saved:
            labels, mask = saved["y_bucket"], saved["mask"]
            end = offset + len(labels)
            if len(labels) != record["count"] or labels.shape != targets[offset:end].shape:
                raise ValueError("Bucket target shape/count mismatch")
            if not np.isin(labels, (0, 1)).all() or np.any(labels[mask == 0] != 0):
                raise ValueError("Nonbinary bucket target or positive padding label")
            for a, b in ((saved["sample_id"], split["ids"][offset:end]), (mask, split["mask"][offset:end]),
                         (saved["y_sequence"], split["y"][offset:end]), (labels.max(1), split["y"][offset:end])):
                np.testing.assert_array_equal(a, b)
            targets[offset:end] = labels
            offset = end
    if offset != len(split) or not records:
        raise ValueError("Incomplete bucket target coverage")
    return targets, {"split": split_name, "sequences": len(split),
        "active_buckets": int(np.asarray(split["mask"], dtype=np.int64).sum()),
        "positive_buckets": int(targets.sum(dtype=np.int64)), "target_sha256": hashlib.sha256(targets.tobytes()).hexdigest(),
        "ids_sha256": hashlib.sha256(np.asarray(split["ids"]).tobytes()).hexdigest(),
        "shard_hashes_verified": True, "ids_masks_and_sequence_labels_verified": True,
        "labels_are_model_inputs": False, "test_opened": False}


def bucket_bce_per_sequence(logits, labels, mask):
    losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (losses * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


@torch.no_grad()
def export_bucket_diagnostics(model, validation, labels, device, out):
    """Descriptive bucket metrics; never select a checkpoint or threshold here."""
    model.eval()
    scores = np.empty(labels.shape, np.float32)
    total_bce = 0.0
    for start in range(0, len(validation), 2048):
        end = min(start + 2048, len(validation))
        x, mask = tensors(validation, slice(start, end), device)
        _, logits = model.forward_with_buckets(x, mask)
        truth = torch.as_tensor(labels[start:end], dtype=torch.float32, device=device)
        total_bce += float(bucket_bce_per_sequence(logits, truth, mask).sum().cpu())
        scores[start:end] = (torch.sigmoid(logits) * mask).cpu().numpy()
    active = np.asarray(validation["mask"], bool)
    y, p = labels[active], scores[active]
    np.savez_compressed(out / "bucket_predictions.npz", sample_id=validation["ids"],
                        mask=validation["mask"], y_bucket=labels, score=scores)
    write_json(out / "bucket_metrics.json", {"split": "validation", "active_buckets": int(active.sum()),
        "positive_buckets": int(y.sum()), "average_precision": float(average_precision_score(y, p)) if y.sum() else None,
        "sequence_mean_active_bucket_bce": total_bce / len(validation), "used_for_checkpoint_selection": False,
        "used_for_sequence_scores_or_thresholds": False, "test_used": False,
        "limitations": ["Previously inspected validation with correlated buckets, not an independent test result.",
                        "The auxiliary head predicts whether any attack-labeled flow occurs in each active bucket."]})


def fit(name, training, validation, weights, config, device, threads, seed, out):
    if name != "lstm_bucket" or config.get("input_transform", {"name": "identity"}) != {"name": "identity"}:
        raise ValueError("This experiment requires chronological lstm_bucket inputs")
    supervision = config["bucket_supervision"]
    coefficient = float(supervision["weight"])
    if not np.isfinite(coefficient) or coefficient < 0 or supervision["reduction"] != REDUCTION:
        raise ValueError("Invalid auxiliary loss settings")
    setup(seed, threads, device)
    model = build(name, config, device)
    if model.circular_shift_start:
        raise ValueError("Bucket target alignment requires unshifted inputs")
    settings = config["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    labels, target_audit = load_bucket_targets(config["dataset"], "train", training)
    write_json(out / "bucket_target_audit.json", {"train": target_audit, "validation_bucket_labels_used_during_fit": False,
               "test_opened": False, "reduction": REDUCTION, "weight": coefficient})
    rng = np.random.default_rng(seed)
    history = []
    best, waiting, best_epoch = -1.0, 0, 0
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        total_loss = sequence_loss = auxiliary_loss = 0.0
        order = rng.permutation(len(training))
        for start in range(0, len(order), settings["batch_size"]):
            ids = order[start:start + settings["batch_size"]]
            x, mask = tensors(training, ids, device)
            y = torch.as_tensor(np.array(training["y"][ids], dtype=np.float32), device=device)
            buckets = torch.as_tensor(labels[ids], dtype=torch.float32, device=device)
            w = torch.as_tensor(weights[ids], device=device)
            optimizer.zero_grad(set_to_none=True)
            logits, bucket_logits = model.forward_with_buckets(x, mask)
            primary = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
            auxiliary = bucket_bce_per_sequence(bucket_logits, buckets, mask)
            loss = ((primary + coefficient * auxiliary) * w).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(ids)
            sequence_loss += float((primary.detach() * w).mean().cpu()) * len(ids)
            auxiliary_loss += float((auxiliary.detach() * w).mean().cpu()) * len(ids)
        scores = predict(model, validation, device)
        ap = float(average_precision_score(validation["y"], scores))
        history.append({"epoch": epoch, "weighted_total_loss": total_loss / len(training),
                        "weighted_train_bce": sequence_loss / len(training),
                        "weighted_bucket_bce": auxiliary_loss / len(training), "validation_ap": ap})
        print(f"{name} epoch={epoch} loss={total_loss/len(training):.6f} sequence_bce={sequence_loss/len(training):.6f} bucket_bce={auxiliary_loss/len(training):.6f} validation_AP={ap:.6f}", flush=True)
        if ap > best:
            best, waiting, best_epoch = ap, 0, epoch
            torch.save({"model_name": name, "architecture": config["architecture"], "state_dict": model.state_dict(),
                        "best_epoch": epoch, "bucket_supervision": supervision}, out / "model.pt")
        else:
            waiting += 1
        write_json(out / "history.json", history)
        if waiting >= settings["patience"]:
            break
    saved = torch.load(out / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(saved["state_dict"])
    validation_labels, validation_audit = load_bucket_targets(config["dataset"], "validation", validation)
    write_json(out / "bucket_target_audit.json", {"train": target_audit, "validation_after_checkpoint_selection": validation_audit,
               "validation_bucket_labels_used_during_fit": False, "test_opened": False, "reduction": REDUCTION, "weight": coefficient})
    export_bucket_diagnostics(model, validation, validation_labels, device, out)
    return model, {"best_epoch": best_epoch, "best_validation_ap": best, "epochs_completed": len(history),
                   "parameters": sum(p.numel() for p in model.parameters()), "bucket_supervision_weight": coefficient,
                   "bucket_head_parameters": sum(p.numel() for p in model.bucket_head.parameters())}
