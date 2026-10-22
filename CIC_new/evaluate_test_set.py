"""Evaluate frozen Scheme1 L2 and Scheme2 on paired real held-out test sequences.

Read existing train-fitted probes and freeze thresholds before opening test.
No retraining, probe fitting, synthetic mixtures, or test-based threshold choice.
"""
import argparse
from datetime import datetime
import itertools
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from CIC_new.diagnose_superimposition import load_models, paired_effect, write_csv, percent
from CIC_new.evaluate_autoencoder import read, write, digest, binary_metrics
from CIC_new.modelling.autoencoder import TemporalAE, sequence_loss
from CIC_new.preprocessing.dataset import transform_model_inputs
from CIC_new.run_scheme1_l2_experiment import latest_l2_run
from CIC_new.run_scheme2_experiment import curve_point

NAMES = ("scheme1_l2", "scheme2")
TRAIN_POINTS = ("balanced_prior_0.5", "train_benign_99pct", "train_benign_99.9pct")
META = ("sample_id", "sample_type", "y_sequence", "left_id", "right_id", "right_shift_buckets",
        "left_ip", "right_ip", "primary_attack", "mask", "y_bucket", "flow_counts")


def freeze_references(models):
    references = {}
    for name in NAMES:
        info = models[name]
        params_path = info["report_path"].parent/"linear_probe.json"
        params = read(params_path)
        if params != info["report"]["linear_probe"]["parameters"]:
            raise ValueError("Saved probe differs from the original completed evaluation")
        if params.get("validation_used_for_fitting"):
            raise ValueError("Expected a real-train-only diagnostic readout")
        thresholds = {key: info["report"]["linear_probe"]["operating_points"][key]["threshold_logit"] for key in TRAIN_POINTS}
        # These two thresholds use only the already saved validation predictions.
        for budget, key in ((.01, "validation_fpr_1pct"), (.001, "validation_fpr_0.1pct")):
            values = info["values"]
            thresholds[key] = curve_point(values["labels"], values["logits"], budget)["threshold_logit"]
        data = Path(info["config"]["dataset"])
        manifest = read(data/"manifest.json")
        for filename in ("schema.json", "scaler.json", "embeddings.npz", "frozen_bucket_encoder.pt"):
            if (info["run"]/filename).exists() and digest(info["run"]/filename) != digest(data/filename):
                raise ValueError(f"Input artifact changed: {data/filename}")
        references[name] = dict(run=str(info["run"]), dataset=str(data),
            input_transform=info["config"].get("input_transform", "none"),
            source_report=str(info["report_path"]), source_report_sha256=digest(info["report_path"]),
            probe_path=str(params_path), probe_sha256=digest(params_path),
            dataset_manifest_sha256=digest(data/"manifest.json"),
            source_sha256=info["report"]["source_sha256"],
            thresholds=thresholds, test_examples_expected=manifest["total_samples"]["test"],
            checkpoint_epoch=read(info["run"]/"summary.json")["best_epoch"])
    if references[NAMES[0]]["test_examples_expected"] != references[NAMES[1]]["test_examples_expected"]:
        raise ValueError("Held-out split budgets differ")
    if digest(Path(references[NAMES[0]]["dataset"])/"donor_plan.npz") != digest(Path(references[NAMES[1]]["dataset"])/"donor_plan.npz"):
        raise ValueError("The two training pipelines did not use the same donor plan")
    return references


def test_chunks(folder):
    folder = Path(folder)
    manifest = read(folder/"manifest.json")
    records = {r["path"]: r for r in manifest["shards"]}
    total = 0
    for path in sorted((folder/"test").glob("part_*.npz")):
        relative = path.relative_to(folder).as_posix()
        record = records[relative]
        if digest(path) != record["sha256"]:
            raise ValueError(f"Held-out shard checksum changed: {path}")
        with np.load(path, allow_pickle=False) as archive:
            keys = ("X", *META, *(("attack_flow_counts",) if "attack_flow_counts" in archive.files else ()))
            a = {key: archive[key] for key in keys}
        n = len(a["sample_id"])
        if n != record["count"] or np.any(a["right_id"] >= 0) or np.any(a["right_shift_buckets"]):
            raise ValueError("Invalid held-out count or synthetic/displaced example")
        if not np.isfinite(a["X"]).all() or np.any(a["X"][a["mask"] == 0] != 0):
            raise ValueError("Nonfinite held-out inputs or nonzero padding")
        if not np.array_equal(a["flow_counts"] > 0, a["mask"] > 0):
            raise ValueError("Held-out masks and counts differ")
        total += n
        yield a
    if total != manifest["total_samples"]["test"]:
        raise ValueError("Incomplete held-out shard coverage")


def paired_test_chunks(left, right):
    streams = [iter(test_chunks(p)) for p in (left, right)]
    blocks = [next(s, None) for s in streams]
    offsets = [0, 0]
    while all(block is not None for block in blocks):
        n = min(len(blocks[i]["sample_id"])-offsets[i] for i in (0, 1))
        parts = [{k: v[offsets[i]:offsets[i]+n] for k, v in blocks[i].items()} for i in (0, 1)]
        for key in META:
            if not np.array_equal(parts[0][key], parts[1][key]):
                raise ValueError(f"Held-out pairing differs: {key}")
        if "attack_flow_counts" not in parts[0]:
            raise ValueError("Scheme1 held-out attack counts are required")
        if "attack_flow_counts" in parts[1] and not np.array_equal(parts[0]["attack_flow_counts"], parts[1]["attack_flow_counts"]):
            raise ValueError("Held-out attack counts differ")
        yield parts
        for i in (0, 1):
            offsets[i] += n
            if offsets[i] == len(blocks[i]["sample_id"]):
                blocks[i] = next(streams[i], None)
                offsets[i] = 0
    if any(block is not None for block in blocks):
        raise ValueError("Held-out split lengths differ")


@torch.no_grad()
def predict(model, params, batch, transform, batch_size, device):
    scores, errors = [], []
    mean, scale, weights = (np.asarray(params[key]) for key in ("mean", "scale", "weights"))
    if np.any(scale <= 0):
        raise ValueError("Invalid frozen latent scaler")
    for begin in range(0, len(batch["X"]), batch_size):
        end = begin+batch_size
        mask = np.array(batch["mask"][begin:end], np.float32)
        x = transform_model_inputs(np.array(batch["X"][begin:end]), mask, transform)
        tx, tm = torch.from_numpy(x).to(device), torch.from_numpy(mask).to(device)
        z = model.encode(tx, tm).cpu().numpy()
        scores.append(((z-mean)/scale)@weights+params["bias"])
        errors.append(sequence_loss(tx, model(tx, tm), tm).cpu().numpy())
    scores, errors = np.concatenate(scores), np.concatenate(errors)
    if not np.isfinite(scores).all() or not np.isfinite(errors).all():
        raise ValueError("Nonfinite frozen held-out predictions")
    return scores, errors


def groups(context):
    y = context["labels"] > 0
    fractions = context["attack_flow_fraction"]
    result = [("composition", "mixed", y & (fractions < 1)),
              ("composition", "pure_attack", y & (fractions == 1))]
    for attack in np.unique(context["primary_attack"][y]):
        selected = y & (context["primary_attack"] == attack)
        result.append(("attack", str(attack), selected))
        result.append(("attack_x_composition", str(attack)+":mixed", selected & (fractions < 1)))
    for lo, hi, name in ((0, .01, "(0,1%]"), (.01, .1, "(1%,10%]"), (.1, .5, "(10%,50%]"), (.5, 1., "(50%,100%)")):
        selected = y & (fractions > lo) & (fractions <= hi)
        if hi == 1:
            selected &= fractions < 1
        result.append(("attack_fraction", name, selected))
    for lo, hi, name in ((1, 1, "1"), (2, 4, "2-4"), (5, 10, "5-10")):
        result.append(("active_buckets", name, (context["active_buckets"] >= lo) & (context["active_buckets"] <= hi)))
    for lo, hi, name in ((1, 1, "1"), (2, 10, "2-10"), (11, 100, "11-100"), (101, 1000, "101-1000"), (1001, np.inf, ">1000")):
        result.append(("total_flows", name, (context["total_flows"] >= lo) & (context["total_flows"] <= hi)))
    return [(d, label, selected) for d, label, selected in result if selected.any()]


def summarize(context, scores, references):
    labels = context["labels"] > 0
    metrics, slices, paired = {}, [], {}
    for name in NAMES:
        metrics[name] = {}
        for point, threshold in references[name]["thresholds"].items():
            overall = binary_metrics(labels, scores[name], threshold)
            metrics[name][point] = overall
            pred = scores[name] >= threshold
            for dimension, group, selected in groups(context):
                pos, neg = selected & labels, selected & ~labels
                slices.append(dict(model=name, threshold=point, dimension=dimension, group=group,
                    positives=int(pos.sum()), negatives=int(neg.sum()),
                    positive_hosts=len(np.unique(context["left_ip"][pos])),
                    detected=int(pred[pos].sum()), recall=float(pred[pos].mean()) if pos.any() else None,
                    false_positives=int(pred[neg].sum()), fpr=float(pred[neg].mean()) if neg.any() else None))
    for point in references[NAMES[0]]["thresholds"]:
        left = scores[NAMES[0]] >= references[NAMES[0]]["thresholds"][point]
        right = scores[NAMES[1]] >= references[NAMES[1]]["thresholds"][point]
        paired[point] = {group: paired_effect(left, right, selected, context["left_ip"])
            for group, selected in (("all_attacks", labels),
                ("mixed", labels & (context["attack_flow_fraction"] < 1)),
                ("pure_attack", labels & (context["attack_flow_fraction"] == 1)))}
    return metrics, slices, paired


def markdown(out, result, models):
    lines = ["# Frozen Scheme1 L2 versus Scheme2: real held-out test", "",
        f"Population: {result['test_examples']} original real test sequences; {result['test_attacks']} positive and {result['test_benign']} benign.",
        "No synthetic mixtures, fitting, checkpoint selection or test-based threshold choice. Saved train-fitted linear readouts remain frozen.",
        "The readout is not the planned CNN. This compares two complete representation pipelines from one seed.", "",
        "| Model | Validation ROC-AUC | Test ROC-AUC | Validation AP | Test AP |", "|---|---:|---:|---:|---:|"]
    for name in NAMES:
        val = models[name]["report"]["linear_probe"]["operating_points"]["balanced_prior_0.5"]
        test = result["metrics"][name]["balanced_prior_0.5"]
        lines.append(f"| {name} | {val['roc_auc']:.5f} | {test['roc_auc']:.5f} | {percent(val['average_precision'])} | {percent(test['average_precision'])} |")
    lines += ["", "## Fixed thresholds", "",
        "Thresholds were serialized before test shards were opened. Actual test FPR is shown; no test ROC point is selected to force equal FPR.", "",
        "| Threshold source | Model | Test FPR | Precision | Recall | TP | FP | Mixed recall |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for point in result["references"][NAMES[0]]["thresholds"]:
        for name in NAMES:
            m = result["metrics"][name][point]
            mixed = result["paired"][point]["mixed"]
            count = mixed.get("both_detected", 0)+mixed.get("only_scheme1_l2" if name == NAMES[0] else "only_scheme2", 0)
            recall = count/mixed["n"] if mixed["n"] else None
            lines.append(f"| {point} | {name} | {percent(m['fpr'])} | {percent(m['precision'])} | {percent(m['recall'])} | {m['tp']} | {m['fp']} | {percent(recall)} |")
    for point in ("train_benign_99pct", "train_benign_99.9pct"):
        lines += ["", f"## Primary attack types: {point}", "", "| Attack | n | Hosts | L2 detected | L2 recall | S2 detected | S2 recall |", "|---|---:|---:|---:|---:|---:|---:|"]
        for attack in sorted({r["group"] for r in result["slices"] if r["dimension"] == "attack"}):
            a = {r["model"]: r for r in result["slices"] if r["dimension"] == "attack" and r["group"] == attack and r["threshold"] == point}
            left, right = a[NAMES[0]], a[NAMES[1]]
            lines.append(f"| {attack} | {left['positives']} | {left['positive_hosts']} | {left['detected']} | {percent(left['recall'])} | {right['detected']} | {percent(right['recall'])} |")
    lines += ["", "## Scope", "",
        "- Test means the original held-out dataset/test shards at natural prevalence, not the previous train-derived stress test.",
        "- Existing validation-selected AE checkpoints and train-fitted probes are used unchanged. No parameter fitting or test threshold calibration occurred.",
        "- Validation-FPR threshold rows apply a threshold selected from the old saved validation predictions; test FPR may differ.",
        "- Train/validation/test use the original time-block policy; this is not an unseen-host or unseen-campaign guarantee.",
        "- Primary attack grouping is not multilabel membership; small group counts and flow-fraction denominators must be inspected.",
        "- Paired source-host bootstrap intervals condition on these fixed models/thresholds and do not estimate training-seed uncertainty.",
        "- Model/probe/representation hashes are checked unchanged. Scores, IDs, fractions and all fixed-threshold group counts are saved separately.",
        "- Future tuning informed by these scores is test-informed development and needs a fresh untouched holdout for another final claim."]
    (out/"TEST_COMPARISON.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def evaluate(models, out, device="cpu", batch_size=1024):
    torch.set_num_threads(2)
    references = freeze_references(models)
    # This file is written before test_chunks opens any held-out feature shard.
    write(out/"frozen_evaluation_plan.json", dict(complete=True, parameter_fitting=False,
        test_threshold_selection=False, references=references))
    networks, params = {}, {}
    for name in NAMES:
        saved = torch.load(models[name]["run"]/"best.pt", map_location="cpu", weights_only=True)
        if saved["epoch"] != references[name]["checkpoint_epoch"] or saved.get("input_transform", "none") != references[name]["input_transform"]:
            raise ValueError("Checkpoint epoch or input transform differs")
        model = TemporalAE(**saved["model_args"]).to(device)
        model.load_state_dict(saved["model_state"])
        model.eval().requires_grad_(False)
        networks[name] = model
        params[name] = read(references[name]["probe_path"])
    scores, errors = {n: [] for n in NAMES}, {n: [] for n in NAMES}
    context = {k: [] for k in ("sample_id", "labels", "primary_attack", "left_ip", "total_flows", "attack_flows", "active_buckets")}
    count = 0
    for a, b in paired_test_chunks(*(references[n]["dataset"] for n in NAMES)):
        for name, batch in zip(NAMES, (a, b)):
            s, e = predict(networks[name], params[name], batch, references[name]["input_transform"], batch_size, device)
            scores[name].append(s); errors[name].append(e)
        total, attack = a["flow_counts"].sum(1), a["attack_flow_counts"].sum(1)
        if (total <= 0).any() or (attack < 0).any() or (attack > total).any() or not np.array_equal(attack > 0, a["y_sequence"] > 0):
            raise ValueError("Held-out attack-flow counts are invalid")
        for key, values in (("sample_id", a["sample_id"]), ("labels", a["y_sequence"]),
            ("primary_attack", a["primary_attack"]), ("left_ip", a["left_ip"]),
            ("total_flows", total), ("attack_flows", attack), ("active_buckets", a["mask"].sum(1).astype(np.int64))):
            context[key].append(values)
        count += len(total)
        if count % 20000 < len(total):
            print(f"[paired real test] {count:,}/{references[NAMES[0]]['test_examples_expected']:,}", flush=True)
    context = {k: np.concatenate(v) for k, v in context.items()}
    if count != references[NAMES[0]]["test_examples_expected"] or len(np.unique(context["sample_id"])) != count:
        raise ValueError("Held-out coverage or ID uniqueness failed")
    context["attack_flow_fraction"] = context["attack_flows"]/context["total_flows"]
    scores, errors = ({n: np.concatenate(v) for n, v in mapping.items()} for mapping in (scores, errors))
    metrics, slices, paired = summarize(context, scores, references)
    for name in NAMES:
        ref = references[name]
        if any(digest(Path(ref["run"])/f) != sha for f, sha in ref["source_sha256"].items()) or digest(Path(ref["probe_path"])) != ref["probe_sha256"]:
            raise ValueError("Model/readout artifacts changed during test evaluation")
        if digest(Path(ref["dataset"])/"manifest.json") != ref["dataset_manifest_sha256"]:
            raise ValueError("Dataset manifest changed during test evaluation")
        np.savez_compressed(out/(name+"_test_scores.npz"), **context, logits=scores[name], reconstruction_mse=errors[name])
    write_csv(out/"test_slices.csv", slices)
    result = dict(complete=True, evaluation_population="real held-out test", test_used=True,
        parameter_fitting=False, test_threshold_selection=False, test_examples=count,
        test_attacks=int(context["labels"].sum()), test_benign=int((context["labels"] == 0).sum()),
        mixed_attack_sequences=int(((context["attack_flow_fraction"] > 0) & (context["attack_flow_fraction"] < 1)).sum()),
        references=references, metrics=metrics, slices=slices, paired=paired,
        model_and_probe_hashes_unchanged=True, paired_shard_hashes_and_metadata_checked=True)
    write(out/"report.json", result)
    markdown(out, result, models)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l2-run", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("Positive inference batch required")
    parent = ROOT/"CIC_new/outputs/test_evaluations"
    parent.mkdir(parents=True, exist_ok=True)
    if args.out is None:
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        out = Path(tempfile.mkdtemp(prefix=stamp+"_scheme1_l2_vs_scheme2_", dir=parent))
    else:
        out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    write(out/"status.json", dict(status="running", complete=False))
    print(f"Real held-out test evaluation directory: {out}", flush=True)
    try:
        models = load_models(args.l2_run or latest_l2_run())
        device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
        result = evaluate(models, out, device, args.batch_size)
    except BaseException as error:
        write(out/"status.json", dict(status="failed", complete=False, error=str(error)))
        raise
    write(out/"status.json", dict(status="complete", complete=True, test_used=True,
        parameter_fitting=False, test_threshold_selection=False))
    print(f"COMPLETE: {out/'TEST_COMPARISON.md'}; {result['test_examples']:,} paired real test examples", flush=True)


if __name__ == "__main__":
    main()
