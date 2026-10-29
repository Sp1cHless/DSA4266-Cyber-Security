"""Frozen LSTM context interventions; oracle stress diagnostics, not model validation."""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path[0] = str(Path(__file__).resolve().parents[1])

import numpy as np
import torch

from train.common.io import ROOT, fresh, read_json, sha256, unique_name, write_json, environment
from train.common.networks import pool, rotate_sequence, sequence_input
from train.common.torch_runner import load, setup
from train.run_circular_shift import write_csv

TARGETS = ("Infilteration", "Brute_Force_-Web")
STUDY = ROOT / "runs/20261024_223102_671768_circular_multiseed_start5_ef34ba"
TEMPORAL = ROOT / "runs/20261025_100701_778931_lstm_temporal_seed4266_d278c9"
BASELINE = ROOT / "lstm/runs/20261023_023651_471618_lstm_seed4266_b121bc"
VARIANTS = ("original", "oracle_mean_attack", "oracle_pool_attack", "oracle_remove_background", "last_active_final")
LIMITATIONS = [
    "Exploratory frozen-model interventions on previously inspected real validation data, not held-out test performance.",
    "No weights are fitted and no thresholds are selected. Every threshold comes from the original full-validation run.",
    "Oracle variants use validation bucket labels to locate attack-bearing buckets and cannot be deployed.",
    "An attack-bearing bucket can contain benign flows or other attack types; these interventions do not separate them.",
    "Oracle mean/pool interventions reuse unchanged recurrent states; background can remain in those states.",
    "Oracle removal zeros only pure-benign active buckets, preserves elapsed-time positions, and recomputes recurrence.",
    "Interventions change the distribution seen by a frozen classifier; gains do not establish gains after retraining.",
    "All target positives are included. A reproducible benign subsample only checks label-free last-active sensitivity; its FPR is descriptive.",
    "Oracle variants are not evaluated as detectors on benign samples. No oracle FPR or deployment recall is claimed.",
    "Cross-window cohorts and encoders differ. Compare changes within each run; seed repeats are not independent observations.",
    "Runtime differences from historical checkpoints are recorded, and original scores are replay-checked.",
    "Historical score replay allows at most 0.002 absolute probability error; per-row saved counts and replay threshold flips are disclosed.",
    "All intervention differences use the current-runtime unchanged forward pass as their paired control, not historical saved scores.",
    "Existing training data were reused by the saved fits; this diagnostic does not open training or test shards.",
]


@torch.no_grad()
def intervene(model, x, mask, attack_mask):
    """Separate pooling from recurrence while retaining the saved classification head."""
    x, mask = rotate_sequence(x, mask, model.circular_shift_start)
    attack_mask = torch.roll(attack_mask, -model.circular_shift_start, 1)
    # Benign controls have no oracle variant: keep their original mask as a safe placeholder.
    positive = attack_mask.any(1)
    selected = torch.where(positive[:, None], attack_mask, mask).to(mask.dtype)
    states, (last, _) = model.lstm(sequence_input(x, mask))
    pooled = pool(states, mask)
    oracle_pooled = pool(states, selected)
    width = states.shape[-1]
    positions = torch.arange(mask.shape[1], device=mask.device)[None, :]
    final_active = torch.where(mask.bool(), positions, -1).max(1).values
    last_active = states[torch.arange(len(x), device=x.device), final_active]
    reduced_states, (reduced_last, _) = model.lstm(sequence_input(x, selected))
    representations = {
        "original": torch.cat((pooled, last[-1]), 1),
        "oracle_mean_attack": torch.cat((oracle_pooled[:, :width], pooled[:, width:], last[-1]), 1),
        "oracle_pool_attack": torch.cat((oracle_pooled, last[-1]), 1),
        "oracle_remove_background": torch.cat((pool(reduced_states, selected), reduced_last[-1]), 1),
        "last_active_final": torch.cat((pooled, last_active), 1),
    }
    return {name: model.head(value).squeeze(-1).cpu().numpy() for name, value in representations.items()}


def sigmoid(logits):
    return 1 / (1 + np.exp(-np.clip(logits.astype(np.float64), -700, 700)))


def select_data(config, predictions, benign_limit):
    dataset = Path(config["dataset"])
    if not (dataset / "manifest.json").is_file():
        # Team checkpoints may contain an absolute path from another computer.
        dataset = ROOT.parent / "datasets/dataset"
    manifest = read_json(dataset / "manifest.json")
    if sha256(dataset / "manifest.json") != config["dataset_manifest_sha256"]:
        raise ValueError("Dataset identity changed")
    target = (predictions["y"] == 1) & np.isin(predictions["primary_attack"], TARGETS)
    negative = np.flatnonzero(predictions["y"] == 0)
    chosen = np.random.default_rng(4266).choice(negative, min(benign_limit, len(negative)), replace=False)
    keep = target.copy()
    keep[chosen] = True
    selected_ids = predictions["sample_id"][keep]
    id_set = set(map(int, selected_ids))
    chunks = {key: [] for key in ("X", "mask", "y_bucket", "flow_counts", "sample_id")}
    for record in manifest["shards"]:
        if Path(record["path"]).parts[0] != "validation":
            continue
        file = dataset / record["path"]
        if sha256(file) != record["sha256"]:
            raise ValueError("Validation shard hash mismatch: " + str(file))
        with np.load(file, allow_pickle=False) as shard:
            take = np.fromiter((int(v) in id_set for v in shard["sample_id"]), bool, len(shard["sample_id"]))
            for key in chunks:
                chunks[key].append(shard[key][take])
    arrays = {key: np.concatenate(values) for key, values in chunks.items()}
    index = {int(value): i for i, value in enumerate(arrays["sample_id"])}
    order = [index[int(value)] for value in selected_ids]
    arrays = {key: value[order] for key, value in arrays.items()}
    np.testing.assert_array_equal(arrays["sample_id"], selected_ids)
    np.testing.assert_array_equal(arrays["y_bucket"].max(1), predictions["y"][keep])
    arrays.update({key: value[keep] for key, value in predictions.items() if key != "sample_id"})
    return arrays


def summarize_scores(label, config, arrays, logits, thresholds):
    scores = {name: sigmoid(value) for name, value in logits.items()}
    np.testing.assert_allclose(scores["original"], arrays["score"], atol=0.002, rtol=0)
    attack = arrays["y_bucket"].astype(bool)
    background = (arrays["mask"].astype(bool) & ~attack).sum(1)
    rows = []
    for category in TARGETS + ("Benign",):
        group = arrays["y"] == 0 if category == "Benign" else (arrays["primary_attack"] == category) & (arrays["y"] == 1)
        slices = ("all",) if category == "Benign" else ("all", "has_pure_benign_bucket", "no_pure_benign_bucket")
        for subgroup in slices:
            keep = group.copy()
            if subgroup != "all":
                keep &= background > 0 if subgroup == "has_pure_benign_bucket" else background == 0
            n = int(keep.sum())
            if not n:
                continue
            for variant in VARIANTS[1:]:
                if category == "Benign" and variant.startswith("oracle"):
                    continue
                for operating, threshold in thresholds.items():
                    original = scores["original"][keep] >= threshold
                    changed = scores[variant][keep] >= threshold
                    saved = arrays["score"][keep] >= threshold
                    rows.append({"run": label, "seed": config["seed"], "start": config.get("input_transform", {}).get("start_bucket", 0),
                        "attack": category, "subgroup": subgroup, "variant": variant, "operating_point": operating,
                        "n": n, "hosts": int(len(np.unique(arrays["source_host"][keep]))),
                        "original_detected": int(original.sum()), "changed_detected": int(changed.sum()),
                        "historical_saved_detected": int(saved.sum()), "replay_threshold_flips": int((saved != original).sum()),
                        "gained": int((~original & changed).sum()), "lost": int((original & ~changed).sum()),
                        "original_rate": float(original.mean()), "changed_rate": float(changed.mean()),
                        "delta_pp": float((changed.mean() - original.mean()) * 100),
                        "mean_logit_delta": float((logits[variant][keep] - logits["original"][keep]).mean()),
                        "median_logit_delta": float(np.median(logits[variant][keep] - logits["original"][keep])),
                        "fraction_logit_increased": float((logits[variant][keep] > logits["original"][keep] + 1e-5).mean()),
                        "threshold": threshold})
    return rows


def run_one(label, folder, out, device, benign_limit, cached):
    config, summary = read_json(folder / "config.json"), read_json(folder / "summary.json")
    if summary["model"] != "lstm" or not summary.get("complete") or summary.get("test_used"):
        raise ValueError("A completed validation-only LSTM is required")
    if sha256(folder / "model.pt") != summary["model_sha256"] or sha256(folder / "thresholds.json") != summary["thresholds_sha256"]:
        raise ValueError("Checkpoint or threshold changed")
    with np.load(folder / "predictions.npz", allow_pickle=False) as saved:
        predictions = {key: saved[key] for key in saved.files}
    identity = config["dataset_manifest_sha256"]
    if identity not in cached:
        cached[identity] = select_data(config, predictions, benign_limit)
    arrays = dict(cached[identity])
    ids_to_position = {int(value): i for i, value in enumerate(predictions["sample_id"])}
    selection = [ids_to_position[int(value)] for value in arrays["sample_id"]]
    for key in ("y", "primary_attack", "source_host"):
        np.testing.assert_array_equal(arrays[key], predictions[key][selection])
    arrays["score"] = predictions["score"][selection]
    setup(config["seed"], 4, device)
    model = load("lstm", config, folder / "model.pt", device)
    model.eval()
    parts = {name: [] for name in VARIANTS}
    for start in range(0, len(arrays["sample_id"]), 512):
        end = start + 512
        values = intervene(model, torch.tensor(arrays["X"][start:end], device=device),
            torch.tensor(arrays["mask"][start:end], dtype=torch.float32, device=device),
            torch.tensor(arrays["y_bucket"][start:end], dtype=torch.float32, device=device))
        with torch.no_grad():
            direct = model(torch.tensor(arrays["X"][start:end], device=device),
                torch.tensor(arrays["mask"][start:end], dtype=torch.float32, device=device)).cpu().numpy()
        np.testing.assert_allclose(values["original"], direct, atol=2e-6, rtol=0)
        for name in VARIANTS:
            parts[name].append(values[name])
    logits = {name: np.concatenate(values) for name, values in parts.items()}
    no_background = ((arrays["mask"] > 0) & (arrays["y_bucket"] == 0)).sum(1) == 0
    for name in VARIANTS[1:4]:
        np.testing.assert_allclose(logits[name][no_background], logits["original"][no_background], atol=2e-6, rtol=0)
    thresholds = read_json(folder / "thresholds.json")["values"]
    rows = summarize_scores(label, config, arrays, logits, thresholds)
    np.savez_compressed(out / (label + ".npz"), sample_id=arrays["sample_id"], y=arrays["y"],
        attack=arrays["primary_attack"], host=arrays["source_host"], active_buckets=arrays["mask"].sum(1),
        attack_buckets=arrays["y_bucket"].sum(1), original_saved_score=arrays["score"], **logits)
    error = float(np.max(np.abs(sigmoid(logits["original"]) - arrays["score"])))
    print(f"{label}: direct-forward identity passed; {len(arrays['sample_id'])} samples; saved-score max error={error:.8f}", flush=True)
    return rows, {"run": label, "folder": str(folder), "checkpoint_sha256": summary["model_sha256"],
        "dataset_sha256": identity, "replay_max_score_error": float(np.max(np.abs(sigmoid(logits["original"]) - arrays["score"]))),
        "historical_environment": config["environment"], "thresholds": thresholds}


def circular_summary(study):
    plan = read_json(study / "plan.json")
    fits = {}
    for record in plan["runs"]:
        if record["model"] != "lstm":
            continue
        folder = Path(record["path"])
        config = read_json(folder / "config.json")
        with np.load(folder / "predictions.npz", allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        fits[(config["seed"], record["start_bucket"])] = (arrays, read_json(folder / "thresholds.json")["values"])
    rows = []
    for seed in plan["seeds"]:
        (left, lt), (right, rt) = fits[(seed, 0)], fits[(seed, plan["start_bucket"])]
        for key in ("sample_id", "y", "primary_attack", "source_host"):
            np.testing.assert_array_equal(left[key], right[key])
        for attack in TARGETS:
            keep = (left["y"] == 1) & (left["primary_attack"] == attack)
            for operating in lt:
                a, b = left["score"][keep] >= lt[operating], right["score"][keep] >= rt[operating]
                rows.append({"seed": seed, "attack": attack, "operating_point": operating,
                    "n": int(keep.sum()), "original_detected": int(a.sum()), "shifted_detected": int(b.sum()),
                    "gained": int((~a & b).sum()), "lost": int((a & ~b).sum()),
                    "original_recall": float(a.mean()), "shifted_recall": float(b.mean()), "delta_pp": float((b.mean()-a.mean())*100)})
    return rows


def report(out, rows, circular):
    lines = ["# Frozen LSTM context diagnostic", "", "Oracle stress diagnostic; not deployable model validation. No training or test scoring.", "",
        "## Existing five-seed circular retraining", "", "Each retrained model uses its own saved global validation threshold.", "",
        "| Attack | FPR budget | Original mean recall | Shift mean recall | Delta (pp) |", "|---|---|---:|---:|---:|"]
    for attack in TARGETS:
        for operating in ("validation_fpr_1pct", "validation_fpr_0.1pct"):
            subset = [r for r in circular if r["attack"] == attack and r["operating_point"] == operating]
            lines.append(f"| {attack} | {operating} | {np.mean([r['original_recall'] for r in subset]):.2%} | {np.mean([r['shifted_recall'] for r in subset]):.2%} | {np.mean([r['delta_pp'] for r in subset]):+.2f} |")
    lines += ["", "## Frozen context interventions: 1% original FPR threshold", "",
        "All target samples, including those with no removable background. Gained/lost are fixed-threshold crossings.", "",
        "| Run | Attack | Variant | n | Original detected | Changed detected | Gained / lost | Median logit change |",
        "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        if (row["subgroup"] == "all" and row["attack"] != "Benign" and row["operating_point"] == "validation_fpr_1pct"
                and not row["run"].startswith("paired_")):
            lines.append(f"| {row['run']} | {row['attack']} | {row['variant']} | {row['n']} | {row['original_detected']} | {row['changed_detected']} | {row['gained']} / {row['lost']} | {row['median_logit_delta']:+.3f} |")
    lines += ["", "See interventions.csv for both operating points, context-eligible subsets and benign last-active controls.",
        "Per-sample NPZ files retain paired logits, labels, occupancy and original saved scores.", "", "## Boundaries", ""] + ["- " + item for item in LIMITATIONS]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--temporal", type=Path, default=TEMPORAL)
    parser.add_argument("--benign-samples", type=int, default=4096)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    if args.benign_samples < 1:
        raise ValueError("benign-samples must be positive")
    out = fresh(args.out or ROOT / "runs" / unique_name("lstm_context_interventions"))
    write_json(out / "status.json", {"status": "running", "test_used": False})
    jobs = [("historical100s_10s", BASELINE)]
    for record in read_json(args.study / "plan.json")["runs"]:
        if record["model"] == "lstm":
            folder = Path(record["path"])
            seed = read_json(folder / "config.json")["seed"]
            jobs.append((f"paired_seed{seed}_start{record['start_bucket']}", folder))
    for folder in sorted(args.temporal.glob("sequence*/lstm_attempt_*/summary.json")):
        if read_json(folder).get("complete"):
            jobs.append((folder.parent.parent.name, folder.parent))
    rows, provenance, cached = [], [], {}
    for label, folder in jobs:
        try:
            result, source = run_one(label, folder, out, args.device, args.benign_samples, cached)
        except Exception as error:
            write_json(out / "status.json", {"status": "failed", "run": label, "error": repr(error), "test_used": False})
            raise
        rows.extend(result)
        provenance.append(source)
    circular = circular_summary(args.study)
    write_csv(out / "interventions.csv", rows)
    write_csv(out / "circular_target_results.csv", circular)
    write_json(out / "provenance.json", {"runs": provenance, "environment": environment(), "limitations": LIMITATIONS,
        "script_sha256": sha256(Path(__file__)), "test_used": False, "training_used": False})
    report(out, rows, circular)
    write_json(out / "status.json", {"status": "complete", "test_used": False, "training_used": False})
    print(str(out), flush=True)
    return out


if __name__ == "__main__":
    main()
