"""Paired validation behavior profiles for selecting a superimposition pipeline.

No fitting, threshold tuning, synthetic validation mixtures or test access.
Reads saved predictions; optionally probes temporal order on a fixed small sample.
"""
import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from CIC_new.evaluate_autoencoder import read, write, digest, binary_metrics
from CIC_new.modelling.autoencoder import TemporalAE
from CIC_new.preprocessing.dataset import ModelInputs
from CIC_new.run_scheme1_l2_experiment import latest_l2_run
from CIC_new.run_scheme2_experiment import curve_point, CONTROLS

MODELS = ("scheme1", "scheme1_l2", "scheme2")
POINTS = ("validation_fpr_1pct", "validation_fpr_0.1pct", "train_benign_99pct", "train_benign_99.9pct")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def paired_effect(left, right, selected, hosts, repetitions=1000, seed=4266):
    """Paired source-host bootstrap with fixed thresholds; not a multi-seed CI."""
    selected = np.asarray(selected, bool)
    if not selected.any():
        return dict(n=0)
    host_names, codes = np.unique(hosts[selected], return_inverse=True)
    sizes = np.bincount(codes).astype(np.float64)
    effects = np.bincount(codes, weights=right[selected].astype(float)-left[selected].astype(float))
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions)
    for i in range(repetitions):
        drawn = rng.integers(len(sizes), size=len(sizes))
        estimates[i] = effects[drawn].sum()/sizes[drawn].sum()
    a, b = left[selected], right[selected]
    return dict(n=int(selected.sum()), source_hosts=len(host_names),
        both_detected=int((a & b).sum()), only_scheme1_l2=int((a & ~b).sum()),
        only_scheme2=int((~a & b).sum()), neither_detected=int((~a & ~b).sum()),
        delta_scheme2_minus_l2=float(b.mean()-a.mean()),
        host_bootstrap_low=float(np.quantile(estimates, .025)),
        host_bootstrap_high=float(np.quantile(estimates, .975)))


def order_indices(mask, mode, seed=4266):
    """Permute content only among active positions; retain mask and padding."""
    rng = np.random.default_rng(seed)
    indices = np.broadcast_to(np.arange(mask.shape[1]), mask.shape).copy()
    for i, row in enumerate(mask):
        active = np.flatnonzero(row > 0)
        indices[i, active] = rng.permutation(active) if mode == "shuffle" else active[::-1]
    return indices


def load_models(l2_run):
    comparisons = [p for p in Path(l2_run).glob("evaluations/*/three_way_comparison.json") if read(p).get("complete")]
    if not comparisons:
        raise FileNotFoundError("A completed three-way evaluation is required")
    out = max(comparisons, key=lambda p: p.parent.name).parent
    first, second = read(out/"comparison.json"), read(out/"vs_scheme2/comparison.json")
    paths = dict(scheme1=Path(first["baseline_report"]), scheme1_l2=out/"report.json",
                 scheme2=Path(second["baseline_report"]))
    models = {}
    for name, path in paths.items():
        report = read(path)
        run = Path(report["training_run"])
        config = read(run/"config.json")
        if not report.get("complete") or report.get("test_used") or not report.get("training_artifacts_unchanged"):
            raise ValueError("Completed frozen validation evaluation required")
        for filename, expected in report["source_sha256"].items():
            if digest(run/filename) != expected:
                raise ValueError(f"Training artifact changed: {run/filename}")
        if digest(run/"dataset_manifest.json") != digest(Path(config["dataset"])/"manifest.json"):
            raise ValueError("Input dataset manifest changed")
        with np.load(path.parent/"validation_scores.npz", allow_pickle=False) as a:
            values = {key: a[key] for key in a.files}
        if len(np.unique(values["sample_id"])) != len(values["sample_id"]) or not np.isfinite(values["logits"]).all():
            raise ValueError("Invalid saved prediction IDs or scores")
        models[name] = dict(run=run, report_path=path, report=report, config=config, values=values,
            score_sha256=digest(path.parent/"validation_scores.npz"), report_sha256=digest(path))
    if models["scheme1_l2"]["config"].get("input_transform") != "bucket_l2":
        raise ValueError("Expected the Scheme1 L2 candidate")
    for name in MODELS:
        for key in ("sample_id", "labels", "attack_flow_fraction"):
            if not np.array_equal(models[name]["values"][key], models["scheme1_l2"]["values"][key]):
                raise ValueError(f"Predictions are not paired: {name}/{key}")
        if any(models[name]["config"][k] != models["scheme1_l2"]["config"][k] for k in CONTROLS):
            raise ValueError("Training controls differ")
    return models


def validation_context(models):
    data = Path(models["scheme1"]["config"]["dataset"])
    manifest, schema, scaler = (read(data/name) for name in ("manifest.json", "schema.json", "scaler.json"))
    if any(digest(data/name) != manifest[name.replace(".json", "_sha256")] for name in ("schema.json", "scaler.json")):
        raise ValueError("Scheme1 feature transforms changed")
    records = {r["path"]: r for r in manifest["shards"]}
    parts = {k: [] for k in ("sample_id", "flow_counts", "attack_flow_counts", "mask", "primary_attack", "left_ip")}
    for path in sorted((data/"validation").glob("part_*.npz")):
        if digest(path) != records[path.relative_to(data).as_posix()]["sha256"]:
            raise ValueError("Validation shard changed")
        with np.load(path, allow_pickle=False) as a:
            for key in parts:
                parts[key].append(a[key])
    context = {k: np.concatenate(v) for k, v in parts.items()}
    values = models["scheme1"]["values"]
    if not np.array_equal(context["sample_id"], values["sample_id"]):
        raise ValueError("Validation metadata order differs")
    counts = context["flow_counts"]
    total = counts.sum(1)
    attack = context["attack_flow_counts"].sum(1)
    if (total <= 0).any() or not np.array_equal(attack > 0, values["labels"] > 0):
        raise ValueError("Invalid flow counts or labels")
    if not np.allclose(attack/total, values["attack_flow_fraction"], rtol=0, atol=1e-15):
        raise ValueError("Saved attack proportions differ from original counts")
    context.update(total_flows=total, attack_flows=attack,
        active_buckets=(context["mask"] > 0).sum(1),
        attack_buckets=(context["attack_flow_counts"] > 0).sum(1))
    names = schema["feature_names"]
    descriptors = ("protocol_tcp_ratio", "protocol_udp_ratio", "icmp_flow_ratio", "dns_flow_ratio")
    columns = [names.index(name) for name in descriptors]
    fractions = np.empty((len(total), len(columns)), np.float64)
    with ModelInputs(data, "validation") as store:
        for begin in range(0, len(total), 4096):
            end = min(begin+4096, len(total))
            x = np.array(store.arrays["X"][begin:end, :, columns])
            unscaled = x*np.asarray(scaler["scale"])[columns]+np.asarray(scaler["mean"])[columns]
            fractions[begin:end] = (unscaled*counts[begin:end, :, None]).sum(1)/total[begin:end, None]
    protocol = np.column_stack((fractions[:, :3], 1-fractions[:, :3].sum(1)))
    winner = protocol.argmax(1)
    context["protocol"] = np.where(protocol.max(1) >= .8,
        np.array(["TCP", "UDP", "ICMP", "other"])[winner], "mixed_protocol")
    context["dns_fraction"] = np.clip(fractions[:, 3], 0, 1)
    context["source_dataset"] = str(data.resolve())
    return context


def strata(context, values):
    y = values["labels"] > 0
    fraction = values["attack_flow_fraction"]
    result = [("composition", "benign", ~y), ("composition", "mixed", y & (fraction < 1)),
              ("composition", "pure_attack", y & (fraction == 1))]
    for attack in np.unique(context["primary_attack"][y]):
        selected = y & (context["primary_attack"] == attack)
        result.append(("attack", str(attack), selected))
        result.extend(("attack_x_composition", str(attack)+":"+name, selected & condition)
                      for name, condition in (("mixed", fraction < 1), ("pure", fraction == 1)))
        for name, condition in (("<=10%", fraction <= .1), ("10-50%", (fraction > .1) & (fraction <= .5)),
                                (">50%-mixed", (fraction > .5) & (fraction < 1))):
            result.append(("attack_x_fraction", str(attack)+":"+name, selected & condition))
    for lo, hi, name in ((0, .01, "(0,1%]"), (.01, .05, "(1%,5%]"), (.05, .1, "(5%,10%]"),
                         (.1, .25, "(10%,25%]"), (.25, .5, "(25%,50%]"), (.5, 1., "(50%,100%)")):
        selected = y & (fraction > lo) & (fraction <= hi)
        if hi == 1:
            selected &= fraction < 1
        result.append(("attack_fraction", name, selected))
    for lo, hi, name in ((1, 1, "1"), (2, 4, "2-4"), (5, 10, "5-10")):
        result.append(("active_buckets", name, (context["active_buckets"] >= lo) & (context["active_buckets"] <= hi)))
    for lo, hi, name in ((1, 1, "1"), (2, 10, "2-10"), (11, 100, "11-100"), (101, 1000, "101-1000"),
                         (1001, 10000, "1001-10000"), (10001, np.inf, ">10000")):
        result.append(("total_flows", name, (context["total_flows"] >= lo) & (context["total_flows"] <= hi)))
    for protocol in np.unique(context["protocol"]):
        result.append(("protocol", str(protocol), context["protocol"] == protocol))
    result.extend(("dns", name, condition) for name, condition in (
        ("none", context["dns_fraction"] < 1e-6),
        ("some", (context["dns_fraction"] >= 1e-6) & (context["dns_fraction"] < .8)),
        ("dominant", context["dns_fraction"] >= .8)))
    for lo, hi, name in ((1, 1, "1"), (2, 4, "2-4"), (5, 10, "5-10")):
        result.append(("attack_buckets", name, y & (context["attack_buckets"] >= lo) & (context["attack_buckets"] <= hi)))
    return [(dimension, name, selected) for dimension, name, selected in result if selected.any()]


def operating_points(models):
    points = {}
    for name, model in models.items():
        v = model["values"]
        points[name] = {
            "validation_fpr_1pct": curve_point(v["labels"], v["logits"], .01),
            "validation_fpr_0.1pct": curve_point(v["labels"], v["logits"], .001),
            **{key: model["report"]["linear_probe"]["operating_points"][key] for key in POINTS[2:]},
        }
        for key in POINTS:
            p = points[name][key]
            verified = binary_metrics(v["labels"], v["logits"], p["threshold_logit"])
            if any(verified[k] != p[k] for k in ("tp", "fp", "fn", "tn")):
                raise ValueError("Operating-point counts do not reproduce")
    return points


def summarize(models, context, points, out):
    y = models["scheme1"]["values"]["labels"] > 0
    fraction = models["scheme1"]["values"]["attack_flow_fraction"]
    groups = strata(context, models["scheme1"]["values"])
    rows, paired_rows, host_rows = [], [], []
    effects = {}
    for point in POINTS:
        predictions = {name: m["values"]["logits"] >= points[name][point]["threshold_logit"] for name, m in models.items()}
        for dimension, label, selected in groups:
            pos, neg = selected & y, selected & ~y
            for name in MODELS:
                pred = predictions[name]
                rows.append(dict(point=point, dimension=dimension, group=label, model=name,
                    n=int(selected.sum()), positives=int(pos.sum()), negatives=int(neg.sum()),
                    positive_hosts=len(np.unique(context["left_ip"][pos])),
                    detected=int(pred[pos].sum()), recall=float(pred[pos].mean()) if pos.any() else None,
                    false_positives=int(pred[neg].sum()), fpr=float(pred[neg].mean()) if neg.any() else None,
                    median_logit=float(np.median(models[name]["values"]["logits"][selected]))))
            if pos.any():
                a, b = predictions["scheme1_l2"][pos], predictions["scheme2"][pos]
                paired_rows.append(dict(point=point, dimension=dimension, group=label, n=int(pos.sum()),
                    both=int((a & b).sum()), only_l2=int((a & ~b).sum()), only_scheme2=int((~a & b).sum()),
                    neither=int((~a & ~b).sum())))
        effects[point] = {name: paired_effect(predictions["scheme1_l2"], predictions["scheme2"], selected, context["left_ip"])
            for name, selected in (("all_attacks", y), ("mixed", y & (fraction < 1)),
                                   ("pure", y & (fraction == 1)), ("attack_fraction_le10pct", y & (fraction <= .1)))}
        for host in np.unique(context["left_ip"][y]):
            selected = y & (context["left_ip"] == host)
            row = dict(point=point, source_ip=str(host), n=int(selected.sum()),
                       attacks=";".join(np.unique(context["primary_attack"][selected])))
            for name in MODELS:
                row[name+"_recall"] = float(predictions[name][selected].mean())
            host_rows.append(row)
    write_csv(out/"slices.csv", rows)
    write_csv(out/"paired_errors.csv", paired_rows)
    write_csv(out/"positive_hosts.csv", host_rows)
    write(out/"paired_host_bootstrap.json", effects)
    overall = {}
    supported = [attack for attack in np.unique(context["primary_attack"][y])
                 if np.count_nonzero(y & (context["primary_attack"] == attack)) >= 20]
    for name in MODELS:
        overall[name] = dict(roc_auc=points[name][POINTS[0]]["roc_auc"],
            average_precision=points[name][POINTS[0]]["average_precision"], operating_points=points[name],
            supported_attack_types=supported,
            macro_recall_supported={point: float(np.mean([
                r["recall"] for r in rows if r["point"] == point and r["model"] == name
                and r["dimension"] == "attack" and r["group"] in supported])) if supported else None for point in POINTS},
            macro_host_recall={point: float(np.mean([r[name+"_recall"] for r in host_rows if r["point"] == point])) for point in POINTS})
    # Every positive and every false alarm in the primary equal-FPR comparison.
    main = POINTS[0]
    preds = {name: m["values"]["logits"] >= points[name][main]["threshold_logit"] for name, m in models.items()}
    include = y | np.logical_or.reduce([p & ~y for p in preds.values()])
    percentiles = {name: np.searchsorted(np.sort(m["values"]["logits"][~y]), m["values"]["logits"], side="right")/np.count_nonzero(~y)
                   for name, m in models.items()}
    cases = []
    for i in np.flatnonzero(include):
        row = dict(sample_id=int(context["sample_id"][i]), source_ip=str(context["left_ip"][i]),
            label=int(y[i]), primary_attack=str(context["primary_attack"][i]),
            attack_flow_fraction=float(fraction[i]), total_flows=int(context["total_flows"][i]),
            active_buckets=int(context["active_buckets"][i]), attack_buckets=int(context["attack_buckets"][i]),
            protocol=str(context["protocol"][i]), dns_fraction=float(context["dns_fraction"][i]))
        for name in MODELS:
            row[name+"_logit"] = float(models[name]["values"]["logits"][i])
            row[name+"_benign_percentile"] = float(percentiles[name][i])
            for point in POINTS:
                row[name+"_"+point] = int(models[name]["values"]["logits"][i] >= points[name][point]["threshold_logit"])
        cases.append(row)
    write_csv(out/"cases.csv", cases)
    return overall, rows, paired_rows, effects


@torch.no_grad()
def temporal_probe(models, context, points, out, benign_limit=4096):
    """Frozen-model sensitivity, not an assertion that changed flows are realistic."""
    y = models["scheme1"]["values"]["labels"] > 0
    rng = np.random.default_rng(4266)
    negative = np.flatnonzero(~y)
    ids = np.sort(np.r_[np.flatnonzero(y), rng.choice(negative, min(benign_limit, len(negative)), replace=False)])
    mask = context["mask"][ids]
    many = mask.sum(1) >= 2
    permutations = {mode: order_indices(mask, mode) for mode in ("shuffle", "reverse")}
    rows = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name, info in models.items():
        config = info["config"]
        saved = torch.load(info["run"]/"best.pt", map_location="cpu", weights_only=True)
        model = TemporalAE(**saved["model_args"]).to(device)
        model.load_state_dict(saved["model_state"])
        model.eval().requires_grad_(False)
        params = read(info["report_path"].parent/"linear_probe.json")
        mean, scale, weights = (np.asarray(params[key]) for key in ("mean", "scale", "weights"))
        transform = config.get("input_transform", "none")
        scores = {}
        with ModelInputs(config["dataset"], "validation", transform) as store:
            if not np.array_equal(store.arrays["sample_id"][ids], context["sample_id"][ids]):
                raise ValueError("Temporal diagnostic samples are not paired")
            for mode in ("original", "shuffle", "reverse"):
                chunks = []
                for begin in range(0, len(ids), 1024):
                    end = min(begin+1024, len(ids))
                    batch = store.take(ids[begin:end])
                    x = batch["X"]
                    if mode != "original":
                        x = np.take_along_axis(x, permutations[mode][begin:end, :, None], axis=1)
                    latent = model.encode(torch.from_numpy(x).to(device), torch.from_numpy(batch["mask"]).to(device)).cpu().numpy()
                    chunks.append(((latent-mean)/scale)@weights+params["bias"])
                scores[mode] = np.concatenate(chunks)
        reference = info["values"]["logits"][ids]
        if not np.allclose(scores["original"], reference, rtol=2e-4, atol=1e-3):
            raise ValueError(f"Diagnostic scores do not reproduce saved predictions: {name}, max={np.max(np.abs(scores['original']-reference))}")
        for mode in ("shuffle", "reverse"):
            for group, selected in (("all_attacks_2plus_buckets", y[ids] & many),
                                    ("mixed_attacks_2plus_buckets", y[ids] & many & (info["values"]["attack_flow_fraction"][ids] < 1)),
                                    ("benign_2plus_buckets", ~y[ids] & many)):
                if not selected.any():
                    continue
                for point in POINTS:
                    threshold = points[name][point]["threshold_logit"]
                    old, new = scores["original"][selected] >= threshold, scores[mode][selected] >= threshold
                    rows.append(dict(model=name, perturbation=mode, group=group, point=point, n=int(selected.sum()),
                        original_positive_rate=float(old.mean()), perturbed_positive_rate=float(new.mean()),
                        positive_to_negative=int((old & ~new).sum()), negative_to_positive=int((~old & new).sum()),
                        decision_flip_rate=float((old != new).mean()),
                        median_absolute_logit_change=float(np.median(np.abs(scores[mode][selected]-scores["original"][selected])))))
        print(f"[temporal sensitivity] {name}: {len(ids)} fixed samples; no fitting", flush=True)
    write_csv(out/"temporal_order.csv", rows)
    return dict(sampled_sequences=len(ids), positive_sequences=int(y[ids].sum()),
        benign_sequences=int((~y[ids]).sum()), threshold_refit=False,
        device=str(device), inference_batch_size=1024,
        note="descriptive sensitivity on selected validation samples; not a natural-prevalence benchmark or a causal feature attribution",
        rows=rows)


def percent(value):
    return "n/a" if value is None else f"{value:.2%}"


def markdown(out, overall, rows, effects, temporal):
    lines = ["# Superimposition choice: paired validation diagnostics", "",
        "Frozen AE + previously fitted real-train-only linear readout. No fitting or test access.",
        "The primary choice compares Scheme1 L2 (statistical composition) with Scheme2 (unit latent addition).",
        "These are complete pipelines; this diagnostic does not isolate the augmentation formula causally.", "",
        "| Model | ROC-AUC | AP | Recall @1% FPR | Recall @0.1% FPR | Macro attack recall @1% (n>=20) | Macro host recall @1% |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name in MODELS:
        m = overall[name]
        p = m["operating_points"]
        lines.append(f"| {name} | {m['roc_auc']:.5f} | {percent(m['average_precision'])} | {percent(p[POINTS[0]]['recall'])} | "
                     f"{percent(p[POINTS[1]]['recall'])} | {percent(m['macro_recall_supported'][POINTS[0]])} | {percent(m['macro_host_recall'][POINTS[0]])} |")
    for point in POINTS[:2]:
        lines += ["", f"## Attack type: {point}", "", "| Primary attack | n | Hosts | Scheme1 | Scheme1 L2 | Scheme2 |", "|---|---:|---:|---:|---:|---:|"]
        attacks = sorted({r["group"] for r in rows if r["dimension"] == "attack"})
        for attack in attacks:
            a = {r["model"]: r for r in rows if r["dimension"] == "attack" and r["group"] == attack and r["point"] == point}
            base = a["scheme1"]
            lines.append(f"| {attack} | {base['positives']} | {base['positive_hosts']} | "+" | ".join(percent(a[n]["recall"]) for n in MODELS)+" |")
    for dimension in ("composition", "attack_fraction", "active_buckets", "total_flows", "protocol", "dns", "attack_buckets"):
        lines += ["", f"## {dimension}: validation_fpr_1pct", "",
            "| Group | Attacks | Benign | L2 recall | S2 recall | L2 FPR | S2 FPR |", "|---|---:|---:|---:|---:|---:|---:|"]
        labels = list(dict.fromkeys(r["group"] for r in rows if r["dimension"] == dimension))
        for label in labels:
            a = {r["model"]: r for r in rows if r["dimension"] == dimension and r["group"] == label and r["point"] == POINTS[0]}
            x, z = a["scheme1_l2"], a["scheme2"]
            lines.append(f"| {label} | {x['positives']} | {x['negatives']} | {percent(x['recall'])} | {percent(z['recall'])} | {percent(x['fpr'])} | {percent(z['fpr'])} |")
    lines += ["", "## Paired positive cases and source-host bootstrap", "",
        "Delta = Scheme2 recall - Scheme1 L2 recall. Thresholds fixed during bootstrap; no training-seed uncertainty included.", "",
        "| Point | Group | n | Hosts | Both | L2 only | S2 only | Neither | Delta | Host-bootstrap 95% interval |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for point in POINTS[:2]:
        for group, r in effects[point].items():
            if not r["n"]:
                continue
            lines.append(f"| {point} | {group} | {r['n']} | {r['source_hosts']} | {r['both_detected']} | {r['only_scheme1_l2']} | {r['only_scheme2']} | {r['neither_detected']} | "
                         f"{percent(r['delta_scheme2_minus_l2'])} | [{percent(r['host_bootstrap_low'])}, {percent(r['host_bootstrap_high'])}] |")
    if temporal:
        lines += ["", "## Temporal-order sensitivity (2+ active buckets only)", "",
            "Saved thresholds retained. Selected sample; benign rows are false-positive rates. This is an order perturbation, not circular shifting or synthetic superposition.", "",
            "| Model | Perturbation | Group | n | Original positive rate | Changed positive rate | Decision flips |",
            "|---|---|---|---:|---:|---:|---:|"]
        for r in temporal["rows"]:
            if r["point"] == POINTS[0]:
                lines.append(f"| {r['model']} | {r['perturbation']} | {r['group']} | {r['n']} | {percent(r['original_positive_rate'])} | {percent(r['perturbed_positive_rate'])} | {percent(r['decision_flip_rate'])} |")
    lines += ["", "## Reading the evidence", "",
        "- Fixed-FPR points are exploratory validation curve points; train-calibrated thresholds are retained in report.json and slices.csv.",
        "- Macro attack recall averages types with >=20 positive examples; inspect small groups rather than treating them as reliable rankings.",
        "- Attack grouping uses primary_attack, not multilabel membership. Sequence flow fraction does not measure packet/byte severity or per-bucket dilution.",
        "- Volume/protocol/length slices are associations; class and attack-type composition differ between slices.",
        "- Host bootstrap accounts for sampled source-host clustering conditionally on the frozen models and thresholds; it is not a guarantee about new attack families or training seeds.",
        "- Temporal perturbations may be out of distribution. Decision changes establish sensitivity, not useful temporal understanding.",
        "- cases.csv contains every positive and every false alarm at the 1% point across all three models, avoiding selected-example-only claims.",
        "- Per-attack x composition/fraction rows, all four operating points and source-host profiles are available in the CSV files.",
        "- No real-only control, controlled dilution study or multi-seed replication was added; additive embedding equivalence is not assumed or established."]
    (out/"DECISION_DIAGNOSTICS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l2-run", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--skip-temporal", action="store_true")
    parser.add_argument("--benign-limit", type=int, default=4096)
    args = parser.parse_args()
    if args.benign_limit < 1:
        parser.error("Positive benign limit required")
    if args.out is None:
        parent = ROOT/"CIC_new/outputs/diagnostics"
        parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        out = Path(tempfile.mkdtemp(prefix=stamp+"_superimpose_choice_", dir=parent))
    else:
        out = args.out.resolve()
        out.mkdir(parents=True, exist_ok=False)
    write(out/"status.json", dict(status="running", complete=False))
    print(f"Diagnostic directory: {out}", flush=True)
    try:
        torch.set_num_threads(2)
        models = load_models(args.l2_run or latest_l2_run())
        context = validation_context(models)
        points = operating_points(models)
        overall, rows, paired, effects = summarize(models, context, points, out)
        print("[paired profiles] all validation scores and source counts verified", flush=True)
        temporal = None if args.skip_temporal else temporal_probe(models, context, points, out, args.benign_limit)
        sources = {}
        for name, m in models.items():
            if digest(m["report_path"]) != m["report_sha256"] or digest(m["report_path"].parent/"validation_scores.npz") != m["score_sha256"]:
                raise ValueError("Source evaluation changed during diagnosis")
            if any(digest(m["run"]/filename) != expected for filename, expected in m["report"]["source_sha256"].items()):
                raise ValueError("Source training artifacts changed during diagnosis")
            sources[name] = dict(run=str(m["run"]), report=str(m["report_path"]),
                report_sha256=m["report_sha256"], score_sha256=m["score_sha256"], training_artifacts_unchanged=True)
        result = dict(complete=True, test_used=False, fitting_performed=False,
            validation_examples=len(context["sample_id"]), models=overall, paired_host_bootstrap=effects,
            temporal_probe=temporal, sources=sources,
            interpretation="descriptive comparison of frozen representation pipelines, not a causal isolation of superimposition")
        write(out/"report.json", result)
        markdown(out, overall, rows, effects, temporal)
    except BaseException as error:
        write(out/"status.json", dict(status="failed", complete=False, error=str(error)))
        raise
    write(out/"status.json", dict(status="complete", complete=True, test_used=False, fitting_performed=False))
    print(f"COMPLETE: {out/'DECISION_DIAGNOSTICS.md'}", flush=True)


if __name__ == "__main__":
    main()
