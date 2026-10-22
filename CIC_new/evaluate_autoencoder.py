"""Frozen-AE diagnostics and a real-train-only linear probe; never opens test."""
import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from CIC_new.modelling.autoencoder import TemporalAE, sequence_loss
from CIC_new.preprocessing.dataset import ModelInputs

KINDS = ("real_benign", "real_malicious", "synthetic_benign", "synthetic_malicious")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            result.update(block)
    return result.hexdigest()


def binary_metrics(labels, scores, threshold=0.):
    """ROC-AUC and non-interpolated AP, grouping tied scores together."""
    labels = np.asarray(labels, bool)
    scores = np.asarray(scores, np.float64)
    if not len(labels) or len(labels) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Nonempty finite scores matching labels required")
    predicted = scores >= threshold
    tp = int(np.count_nonzero(predicted & labels))
    fp = int(np.count_nonzero(predicted & ~labels))
    fn = int(np.count_nonzero(~predicted & labels))
    tn = int(np.count_nonzero(~predicted & ~labels))
    positives, negatives = tp + fn, tn + fp
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / positives if positives else None
    specificity = tn / negatives if negatives else None
    result = dict(n=len(labels), positives=positives, prevalence=positives / len(labels),
                  tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall,
                  f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                  accuracy=(tp+tn)/len(labels), fpr=fp/negatives if negatives else None,
                  balanced_accuracy=(recall+specificity)/2 if positives and negatives else None,
                  threshold_logit=float(threshold), roc_auc=None, average_precision=None)
    if positives and negatives:
        order = np.argsort(-scores, kind="stable")
        ordered = scores[order]
        ends = np.r_[np.flatnonzero(ordered[1:] != ordered[:-1]), len(order)-1]
        true_positive = np.cumsum(labels[order], dtype=np.int64)[ends]
        false_positive = ends+1-true_positive
        r = true_positive/positives
        p = true_positive/(ends+1)
        result["average_precision"] = float(np.sum(np.diff(np.r_[0., r])*p))
        result["roc_auc"] = float(np.trapezoid(np.r_[0., r], np.r_[0., false_positive/negatives]))
    return result


def distribution(values):
    values = np.asarray(values)
    return dict(n=len(values), mean=float(values.mean()), median=float(np.median(values)),
                p90=float(np.quantile(values, .9)), p99=float(np.quantile(values, .99)),
                maximum=float(values.max()))


@torch.no_grad()
def infer(model, store, batch_size, device):
    n, width = len(store), store.manifest["feature_width"]
    errors = np.empty(n, np.float64)
    baseline = np.empty(n, np.float64)
    latent = np.empty((n, model.to_latent.out_features), np.float32)
    feature_sums = defaultdict(lambda: np.zeros(width, np.float64))
    feature_counts = defaultdict(int)
    model.eval()
    for start in range(0, n, batch_size):
        end = min(start+batch_size, n)
        batch = store.take(slice(start,end))
        x = torch.from_numpy(batch["X"]).to(device)
        mask = torch.from_numpy(batch["mask"]).to(device)
        prediction = model(x, mask)
        errors[start:end] = sequence_loss(x, prediction, mask).cpu().numpy()
        baseline[start:end] = sequence_loss(x, torch.zeros_like(x), mask).cpu().numpy()
        latent[start:end] = model.encode(x, mask).cpu().numpy()
        per_feature = (((x-prediction).square()*mask[..., None]).sum(dim=1)
                       /mask.sum(dim=1).clamp_min(1)[:, None]).cpu().numpy()
        kinds = store.arrays["sample_type"][start:end]
        for kind in np.unique(kinds):
            selected = kinds == kind
            feature_sums[str(kind)] += per_feature[selected].sum(axis=0, dtype=np.float64)
            feature_counts[str(kind)] += int(selected.sum())
    if not np.isfinite(errors).all() or not np.isfinite(latent).all():
        raise ValueError("Nonfinite model output")
    print(f"[inference] {store.split}: {n} sequences", flush=True)
    return dict(mse=errors, zero_baseline=baseline, latent=latent,
                feature_means={k: v/feature_counts[k] for k, v in feature_sums.items()})


def reconstruction_report(store, values, feature_names):
    kinds = store.arrays["sample_type"]
    active = np.asarray(store.arrays["mask"].sum(axis=1), np.int64)
    report = dict(groups={}, primary_attack={}, active_buckets={}, tail={})
    for kind in np.unique(kinds):
        selected = kinds == kind
        stats = distribution(values["mse"][selected])
        zero = float(values["zero_baseline"][selected].mean())
        stats.update(zero_baseline_mse=zero, reduction_vs_zero=1-stats["mean"]/zero if zero else None)
        feature_errors = values["feature_means"][str(kind)]
        stats["top_features"] = [dict(feature=feature_names[i], mse=float(feature_errors[i]),
            loss_share=float(feature_errors[i]/feature_errors.sum()))
            for i in np.argsort(-feature_errors)[:12]]
        report["groups"][str(kind)] = stats
        for lo, hi, name in ((1, 1, "1"), (2, 4, "2-4"), (5, 10, "5-10")):
            subset = selected & (active >= lo) & (active <= hi)
            if subset.any():
                report["active_buckets"][str(kind)+":"+name] = distribution(values["mse"][subset])
    for attack in np.unique(store.arrays["primary_attack"]):
        selected = store.arrays["primary_attack"] == attack
        report["primary_attack"][str(attack)] = distribution(values["mse"][selected])
    for fraction in (.001, .01):
        count = max(1, int(np.ceil(len(store)*fraction)))
        top = np.partition(values["mse"], len(store)-count)[-count:]
        report["tail"][str(fraction)] = dict(n=count, share_of_total_mse=float(top.sum()/values["mse"].sum()))
    top_ids = np.argsort(-values["mse"])[:20]
    report["highest_error_sequences"] = [dict(sample_id=int(store.arrays["sample_id"][i]),
        source_ip=str(store.arrays["left_ip"][i]), label=int(store.arrays["y_sequence"][i]),
        primary_attack=str(store.arrays["primary_attack"][i]), active_buckets=int(active[i]),
        mse=float(values["mse"][i])) for i in top_ids]
    covariance = np.cov(values["latent"], rowvar=False)
    eigenvalues = np.maximum(np.linalg.eigvalsh(np.atleast_2d(covariance)), 0)
    p = eigenvalues/eigenvalues.sum() if eigenvalues.sum() else np.zeros_like(eigenvalues)
    positive = p > 0
    report["latent"] = dict(dimensions=values["latent"].shape[1],
        effective_rank=float(np.exp(-(p[positive]*np.log(p[positive])).sum())) if positive.any() else 0.,
        standard_deviations=np.std(values["latent"], axis=0, dtype=np.float64).tolist())
    return report


def audit_sampling(run, store, summary, config, out):
    ids = np.asarray(store.arrays["sample_id"])
    lookup = np.full(int(ids.max())+1, -1, np.int32)
    lookup[ids] = np.arange(len(ids), dtype=np.int32)
    kind_codes = np.full(len(store), -1, np.int8)
    for k, name in enumerate(KINDS):
        kind_codes[store.arrays["sample_type"] == name] = k
    counts = np.zeros(len(store), np.int64)
    steps = np.zeros((summary["steps"]+1, len(KINDS)), np.int32)
    kind_map = {v: k for k, v in enumerate(KINDS)}
    chunk, records = [], 0
    audit_hash = hashlib.sha256()
    def consume():
        block = np.asarray(chunk, np.int64)
        sample_ids, update, kinds = block.T
        if (sample_ids < 0).any() or (sample_ids >= len(lookup)).any():
            raise ValueError("Unknown sample in audit")
        positions = lookup[sample_ids]
        if (positions < 0).any() or not np.array_equal(kind_codes[positions], kinds):
            raise ValueError("Audit sample is not a matching training example")
        if (update < 1).any() or (update > summary["steps"]).any():
            raise ValueError("Audit update outside completed training budget")
        np.add.at(counts, positions, 1)
        np.add.at(steps, (update, kinds), 1)
        chunk.clear()
    with (run/"sample_draws.jsonl").open("rb") as handle:
        for line in handle:
            audit_hash.update(line)
            row = json.loads(line)
            chunk.append((row["sample_id"], row["step"], kind_map[row["sample_type"]]))
            records += 1
            if len(chunk) == 65536:
                consume()
            if records % 2000000 == 0:
                print(f"[sampling audit] {records}/{summary['sample_draws']} draws", flush=True)
    if chunk:
        consume()
    expected = np.array([config["batch_size"]//2]*2+[0, 0]) if config["real_only"] else np.full(4, config["batch_size"]//4)
    if records != summary["sample_draws"] or not (steps[1:] == expected).all():
        raise ValueError("Audit does not match the balanced per-update sampling contract")
    groups = {}
    for k, name in enumerate(KINDS):
        selected = kind_codes == k
        if not selected.any():
            continue
        observed = counts[selected]
        groups[name] = dict(population=int(selected.sum()), draws=int(observed.sum()),
                           unique_seen=int(np.count_nonzero(observed)), coverage=float(np.mean(observed > 0)),
                           min_draws=int(observed.min()), median_draws=float(np.median(observed)),
                           max_draws=int(observed.max()))
    np.savez_compressed(out/"training_draw_counts.npz", sample_id=ids, draws=counts)
    return dict(records=records, all_updates_balanced=True, all_draws_exported_train_members=True,
                sample_audit_sha256=audit_hash.hexdigest(), groups=groups)


def fit_probe(train_z, train_y, validation_z, device):
    """One fixed convex readout; no validation-driven fitting or hyperparameter search."""
    mean = train_z.mean(axis=0, dtype=np.float64)
    scale = train_z.std(axis=0, dtype=np.float64)
    scale[scale < 1e-8] = 1.
    x = torch.from_numpy(((train_z-mean)/scale).astype(np.float32)).to(device)
    y = torch.from_numpy(train_y.astype(np.float32)).to(device)
    n1, n0 = int(train_y.sum()), int((train_y == 0).sum())
    if not n0 or not n1:
        raise ValueError("Both real training classes needed for a linear probe")
    weights = torch.where(y > 0, .5/n1, .5/n0)
    w = torch.zeros(x.shape[1], device=device, requires_grad=True)
    b = torch.zeros((), device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([w, b], lr=1., max_iter=100, history_size=10,
        tolerance_grad=1e-7, tolerance_change=1e-9, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        objective = (F.binary_cross_entropy_with_logits(x@w+b, y, reduction="none")*weights).sum()+.0005*w.square().sum()
        objective.backward()
        return objective
    optimizer.step(closure)
    objective = float(closure().detach().cpu())
    gradient = float(torch.cat((w.grad.flatten(), b.grad.reshape(1))).abs().max().cpu())
    coefficients, intercept = w.detach().cpu().numpy(), float(b.detach().cpu())
    training_scores = (x@w+b).detach().cpu().numpy()
    validation_scores = (((validation_z-mean)/scale)@coefficients+intercept).astype(np.float64)
    parameters = dict(mean=mean.tolist(), scale=scale.tolist(), weights=coefficients.tolist(), bias=intercept,
        l2=.001, max_iterations=100, fitted_real_train_examples=len(train_y),
        objective=objective, maximum_absolute_gradient=gradient,
        class_weighting="each real training class contributes 50 percent", validation_used_for_fitting=False,
        probabilities_calibrated_to_natural_prevalence=False)
    return parameters, training_scores, validation_scores


def flow_composition(store):
    total = np.empty(len(store), np.int64)
    attack = np.empty(len(store), np.int64)
    offset = 0
    missing_attack_counts = False
    for path in sorted((store.folder/store.split).glob("part_*.npz")):
        with np.load(path, allow_pickle=False) as a:
            n = len(a["sample_id"])
            if not np.array_equal(a["sample_id"], store.arrays["sample_id"][offset:offset+n]):
                raise ValueError("Flow-count shard order differs from model cache")
            total[offset:offset+n] = a["flow_counts"].sum(axis=1)
            if "attack_flow_counts" in a.files:
                attack[offset:offset+n] = a["attack_flow_counts"].sum(axis=1)
            else:
                missing_attack_counts = True
            offset += n
    if missing_attack_counts:
        sidecar = store.folder/f"flow_metadata_{store.split}.npz"
        if sidecar.is_file():
            index = read(store.folder/"flow_metadata.json")
            if not index.get("complete") or digest(sidecar) != index["splits"][store.split]["sha256"]:
                raise ValueError("Flow metadata sidecar checksum differs")
            with np.load(sidecar, allow_pickle=False) as a:
                ids = a["sample_id"]
                positions = np.searchsorted(ids, store.arrays["sample_id"])
                if len(ids) != len(store) or np.any(np.diff(ids) <= 0) or (positions >= len(ids)).any():
                    raise ValueError("Flow metadata sidecar IDs differ")
                if not np.array_equal(ids[positions], store.arrays["sample_id"]) or not np.array_equal(
                        a["total_flows"][positions], total) or not np.array_equal(
                        a["labels"][positions], store.arrays["y_sequence"]):
                    raise ValueError("Flow metadata IDs, total flows or labels differ")
                attack[:] = a["attack_flows"][positions]
            missing_attack_counts = False
    if missing_attack_counts:
        # Legacy scheme2 exports omit attack-flow counts. Recover metadata only
        # from the paired real validation sequences, never from model predictions.
        reference = store.manifest.get("paired_reference_dataset")
        if store.manifest.get("representation") != "scheme2" or reference is None or store.split != "validation":
            raise ValueError("Missing attack-flow counts without a paired scheme2 validation reference")
        ids, ref_totals, ref_attacks, ref_labels = [], [], [], []
        for path in sorted((Path(reference)/"validation").glob("part_*.npz")):
            with np.load(path, allow_pickle=False) as a:
                ids.append(a["sample_id"])
                ref_totals.append(a["flow_counts"].sum(axis=1))
                ref_attacks.append(a["attack_flow_counts"].sum(axis=1))
                ref_labels.append(a["y_sequence"])
        ids = np.concatenate(ids)
        order = np.argsort(ids)
        positions = np.searchsorted(ids[order], store.arrays["sample_id"])
        if len(ids) != len(store) or len(np.unique(ids)) != len(ids) or (positions >= len(ids)).any():
            raise ValueError("Paired reference validation IDs differ")
        positions = order[positions]
        if not np.array_equal(ids[positions], store.arrays["sample_id"]) or not np.array_equal(
                np.concatenate(ref_totals)[positions], total) or not np.array_equal(
                np.concatenate(ref_labels)[positions], store.arrays["y_sequence"]):
            raise ValueError("Paired reference validation IDs, total flows or labels differ")
        attack[:] = np.concatenate(ref_attacks)[positions]
    if offset != len(store) or (total <= 0).any() or (attack < 0).any() or (attack > total).any():
        raise ValueError("Invalid validation flow counts")
    if not np.array_equal(attack > 0, store.arrays["y_sequence"] > 0):
        raise ValueError("Attack counts do not match sequence labels")
    return total, attack


def input_feature_names(manifest, schema):
    if manifest.get("representation") == "scheme2":
        return [f"bucket_embedding_{i}" for i in range(manifest["feature_width"]-1)]+["log_flow_count_scaled"]
    names = schema["feature_names"]
    if len(names) != manifest["feature_width"]:
        raise ValueError("Input feature names do not match the dataset width")
    return names


def probe_report(train, validation, train_values, validation_values, device, out):
    real = train.arrays["right_id"] < 0
    y = np.asarray(train.arrays["y_sequence"][real])
    params, training_scores, scores = fit_probe(train_values["latent"][real], y, validation_values["latent"], device)
    write(out/"linear_probe.json", params)
    thresholds = {"balanced_prior_0.5": 0.,
                  "train_benign_99pct": float(np.quantile(training_scores[y == 0], .99)),
                  "train_benign_99.9pct": float(np.quantile(training_scores[y == 0], .999))}
    validation_y = np.asarray(validation.arrays["y_sequence"])
    total, attack = flow_composition(validation)
    fractions = attack/total
    mixed = (attack > 0) & (attack < total)
    attacks = validation.arrays["primary_attack"]
    report = dict(parameters=params, validation_population="all real validation sequences at natural prevalence",
                  flow_composition_source=("checksummed original-sequence sidecar with exact ID/count/label checks"
                      if (validation.folder/"flow_metadata_validation.npz").is_file() else
                      "paired real validation reference, with exact ID/count/label checks"
                      if validation.manifest.get("representation")=="scheme2" else "exported attack-flow counts"),
                  threshold_selection="fixed 0.5 under balanced priors, or real-training benign score quantiles; no validation threshold tuning",
                  per_attack_grouping="primary_attack; not multilabel attack membership",
                  operating_points={}, mixed_attack_sequences=int(mixed.sum()),
                  pure_attack_sequences=int(np.count_nonzero((attack > 0) & ~mixed)))
    for name, threshold in thresholds.items():
        metrics = binary_metrics(validation_y, scores, threshold)
        predicted = scores >= threshold
        metrics["primary_attack_recall"] = {
            str(a): dict(n=int(selected.sum()), detected=int(predicted[selected].sum()), recall=float(predicted[selected].mean()))
            for a in np.unique(attacks[validation_y > 0])
            for selected in [(attacks == a) & (validation_y > 0)]}
        bands = {}
        for low, high, label in ((0, .01, "(0,1%]"), (.01, .1, "(1%,10%]"), (.1, .5, "(10%,50%]"), (.5, 1., "(50%,100%]")):
            selected = (fractions > low) & (fractions <= high)
            if selected.any():
                bands[label] = dict(n=int(selected.sum()), detected=int(predicted[selected].sum()),
                                    recall=float(predicted[selected].mean()))
        metrics["attack_flow_fraction_recall"] = bands
        metrics["mixed_attack_recall"] = float(predicted[mixed].mean()) if mixed.any() else None
        report["operating_points"][name] = metrics
    np.savez_compressed(out/"validation_scores.npz", sample_id=validation.arrays["sample_id"],
        labels=validation_y, logits=scores, mse=validation_values["mse"], attack_flow_fraction=fractions)
    return report


def evaluate(run, out, device, batch_size):
    run, out = Path(run).resolve(), Path(out).resolve()
    config, summary = read(run/"config.json"), read(run/"summary.json")
    if not summary["complete"]:
        raise ValueError("Completed training run required")
    data = Path(config["dataset"])
    if digest(run/"dataset_manifest.json") != digest(data/"manifest.json"):
        raise ValueError("Dataset manifest changed after training")
    before = {name: digest(run/name) for name in ("best.pt", "last.pt", "history.csv", "config.json", "summary.json", "validation_ids.npy")}
    checkpoint = torch.load(run/"best.pt", map_location="cpu", weights_only=True)
    if checkpoint["epoch"] != summary["best_epoch"]:
        raise ValueError("Best checkpoint and summary disagree")
    transform = config.get("input_transform", "none")
    if checkpoint.get("input_transform", "none") != transform:
        raise ValueError("Checkpoint and config input transforms disagree")
    torch.set_num_threads(2)
    model = TemporalAE(**checkpoint["model_args"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.requires_grad_(False)
    manifest = read(data/"manifest.json")
    names = input_feature_names(manifest, read(data/"schema.json"))
    with ModelInputs(data, "train", transform) as train, ModelInputs(data, "validation", transform) as validation:
        train_values = infer(model, train, batch_size, device)
        validation_values = infer(model, validation, batch_size, device)
        selected_ids = np.load(run/"validation_ids.npy", allow_pickle=False)
        order = np.argsort(validation.arrays["sample_id"])
        positions = np.searchsorted(validation.arrays["sample_id"][order], selected_ids)
        if (positions >= len(validation)).any() or not np.array_equal(validation.arrays["sample_id"][order[positions]], selected_ids):
            raise ValueError("Checkpoint-selection IDs are not validation examples")
        actual = float(validation_values["mse"][order[positions]].mean())
        if not np.isclose(actual, summary["best_validation_mse"], rtol=1e-5, atol=1e-7):
            raise ValueError("Checkpoint-selection validation IDs do not reproduce the saved best score")
        report = dict(complete=False, training_run=str(run), evaluation_directory=str(out),
            best_epoch=checkpoint["epoch"], test_used=False,
            dataset_representation=manifest.get("representation", "legacy"),
            input_transform=transform,
            checkpoint_selection_validation_examples=len(selected_ids),
            reproduced_checkpoint_validation_mse=actual,
            reconstruction_space="per-bucket L2 of all final scaled scheme1 features, including flow count; sequence-weighted MSE over active buckets" if transform=="bucket_l2" else ("unit whole-bucket vectors plus scaled log-flow-count channel; sequence-weighted MSE over active buckets" if manifest.get("representation")=="scheme2" else "scaled/log-transformed model input; sequence-weighted MSE over active buckets"),
            train=reconstruction_report(train, train_values, names),
            validation=reconstruction_report(validation, validation_values, names))
        write(out/"reconstruction.json", report)
        report["sampling"] = audit_sampling(run, train, summary, config, out)
        print("[probe] fitting one fixed linear readout on real train only", flush=True)
        report["linear_probe"] = probe_report(train, validation, train_values, validation_values, device, out)
    if any(digest(run/name) != value for name, value in before.items()):
        raise ValueError("Training artifact changed during evaluation")
    report.update(complete=True, training_artifacts_unchanged=True, source_sha256=before,
        limitations=["validation was used for AE checkpoint selection; these are exploratory validation metrics, not final test results",
                     "a linear probe is a separate diagnostic classifier, not the planned CNN or an AE/CNN hybrid",
                     "no real-only AE control or multiple-seed comparison has been run; augmentation benefit and cross-seed stability are unmeasured"])
    write(out/"report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="Completed AE run; default is latest completed run under outputs/ae_runs")
    parser.add_argument("--out", type=Path, help="Fresh evaluation directory; default is a unique subdirectory of the training run")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("Positive batch size required")
    if args.run is None:
        runs = [p.parent for p in (ROOT/"CIC_new/outputs/ae_runs").glob("*/summary.json") if read(p).get("complete")]
        if not runs:
            parser.error("No completed AE run found")
        args.run = max(runs, key=lambda p: p.name)
    if args.out is None:
        parent = args.run/"evaluations"
        parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        args.out = Path(tempfile.mkdtemp(prefix=stamp+"_validation_diagnostics_", dir=parent))
    else:
        args.out.mkdir(parents=True, exist_ok=False)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    status = dict(status="running", complete=False, started_at=datetime.now().astimezone().isoformat())
    write(args.out/"evaluation_status.json", status)
    print(f"Evaluation directory: {args.out.resolve()}", flush=True)
    started = time.monotonic()
    try:
        report = evaluate(args.run, args.out, device, args.batch_size)
    except BaseException as error:
        status.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        write(args.out/"evaluation_status.json", status)
        raise
    status.update(status="complete", complete=True, seconds=time.monotonic()-started)
    write(args.out/"evaluation_status.json", status)
    print(json.dumps(dict(complete=True, best_epoch=report["best_epoch"],
        linear_probe=report["linear_probe"]["operating_points"]["balanced_prior_0.5"],
        report=str(args.out/"report.json")), indent=2), flush=True)


if __name__ == "__main__":
    main()
