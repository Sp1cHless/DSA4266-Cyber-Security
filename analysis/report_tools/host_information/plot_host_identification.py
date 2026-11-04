"""Clarify S1 labels using the completed host experiment summary tables."""

import csv
import hashlib
import json
from pathlib import Path
import sys


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "CIC_new").is_dir())
LOCAL_DEPS = ROOT / "_local_archive/report_plot_dependencies_20261029"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RUN = ROOT / "train/runs/20261030_000611_945296_lstm_host_adversarial_seed4266_b1a7f9"
SOURCES = [RUN / "probe_comparison.csv", RUN / "overall_comparison.csv"]
STEM = "fig_host_identification_explained"
OUTPUTS = [HERE / "figures" / f"{STEM}.{ext}" for ext in ("png", "svg")]
PROVENANCE = HERE / "results" / f"{STEM}_provenance.json"


def main():
    for path in [*OUTPUTS, PROVENANCE]:
        if path.exists():
            raise FileExistsError(f"Preserve existing output: {path}")
    source_hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES}
    with SOURCES[0].open(encoding="utf-8-sig", newline="") as stream:
        probes = {(r["variant"], r["task"]): r for r in csv.DictReader(stream)}
    with SOURCES[1].open(encoding="utf-8-sig", newline="") as stream:
        detection = {r["variant"]: r for r in csv.DictReader(stream)}
    tasks = ["attack_type:Bot", "attack_type:Brute_Force_-Web", "attack_type:Infilteration"]
    variants = ["lambda0", "lambda0.01", "lambda0.1"]
    baseline = [probes["lambda0", task] for task in tasks]
    assert [int(r["hosts"]) for r in baseline] == [8, 6, 22]
    assert [int(r["evaluation_n"]) for r in baseline] == [231, 51, 396]
    assert all(probes[v, "attack"]["hosts"] == "43" for v in variants)
    assert all(probes[v, "attack"]["evaluation_n"] == "640" for v in variants)
    assert all(probes[v, "attack"]["detector_training_data_reused"] == "True" for v in variants)
    aggregate = [100 * float(probes[v, "attack"]["balanced_accuracy"]) for v in variants]
    aps = [100 * float(detection[v]["ap"]) for v in variants]
    old = json.loads((HERE / "results/figure_provenance.json").read_text(encoding="utf-8"))
    assert np.allclose(aggregate, old["aggregate_attack_probe_balanced_accuracy_percent"], atol=1e-12, rtol=0)
    assert np.allclose(aps, old["validation_ap_percent"], atol=1e-12, rtol=0)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.labelsize": 11.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none", "axes.axisbelow": True,
    })
    blue, teal, orange, gray = "#4279A8", "#4E9390", "#D68B46", "#CBD6E0"
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 5.6),
                             gridspec_kw={"width_ratios": [1.65, 1.05, 1.05]})
    x = np.arange(3)
    within = {}
    ax = axes[0]
    for label, offset, color, values in [
        ("Chance (1/K)", -.25, gray, [100 * float(r["uniform_chance"]) for r in baseline]),
        ("Baseline (λ = 0)", 0, blue, [100 * float(r["balanced_accuracy"]) for r in baseline]),
        ("Strength λ = 0.1", .25, orange,
         [100 * float(probes["lambda0.1", t]["balanced_accuracy"]) for t in tasks]),
    ]:
        within[label] = values
        ax.bar(x + offset, values, width=.23, color=color, label=label)
        for pos, value in zip(x + offset, values):
            lift = 6 if label == "Strength λ = 0.1" and pos < .5 else 1.5
            ax.text(pos, value + lift, f"{value:.1f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x, ["Bot\nK = 8; n = 231", "Web brute force\nK = 6; n = 51",
                      "Infilteration\nK = 22; n = 396"])
    ax.tick_params(axis="x", labelsize=10)
    ax.set_ylim(0, 90)
    ax.set_ylabel("Source-host identification\nBalanced accuracy (%)")
    ax.set_title("(a) Host identification\nwithin attack types", loc="left", pad=12)
    ax.legend(loc="upper left", bbox_to_anchor=(-.05, -.23), frameon=False,
              fontsize=10, ncol=2, handlelength=1.1, columnspacing=.8)

    ax = axes[1]
    ax.bar(x, aggregate, width=.52, color=[blue, teal, orange])
    for pos, value in zip(x, aggregate):
        ax.text(pos, value + 1.3, f"{value:.2f}", ha="center", va="bottom", fontsize=11)
    chance = 100 * float(probes["lambda0", "attack"]["uniform_chance"])
    ax.axhline(chance, color="#777777", linestyle="--", linewidth=1.2)
    ax.text(1, chance + 1.3, f"Chance: {chance:.2f}%", ha="center", va="bottom",
            fontsize=10, color="#666666")
    ax.set_ylim(0, 60)
    ax.set_xticks(x, ["0\nBaseline", "0.01", "0.1"])
    ax.set_xlabel("Adversarial strength λ")
    ax.set_ylabel("Source-host identification\nBalanced accuracy (%)")
    ax.set_title("(b) Host identification\nafter adversarial training", loc="left", pad=12)

    ax = axes[2]
    ax.plot(x, aps, color="#818D98", linewidth=1.6, zorder=2)
    ax.scatter(x, aps, color=[blue, teal, orange], s=58, zorder=3)
    for pos, value in zip(x, aps):
        ax.text(pos, value + .19, f"{value:.2f}", ha="center", va="bottom", fontsize=11)
    ax.set_xticks(x, ["0\nBaseline", "0.01", "0.1"])
    ax.set_xlim(-.45, 2.45)
    ax.set_ylim(58.7, 63)
    ax.set_xlabel("Adversarial strength λ")
    ax.set_ylabel("Attack detection\nValidation AP (%)")
    ax.set_title("(c) Attack detection\nafter adversarial training", loc="left", pad=12)

    for ax in axes:
        ax.grid(axis="y", alpha=.16)
    fig.subplots_adjust(left=.075, right=.985, top=.74, bottom=.29, wspace=.57)
    fig.text(.075, .947, "Source-host identification and attack detection",
             fontsize=14, weight="bold")
    fig.text(.075, .89, "Host probes predict source identity from frozen LSTM sequence representations.",
             fontsize=11, color="#475569")
    fig.text(.985, .13, "K = eligible source hosts; n = host-probe evaluation sequences",
             ha="right", fontsize=10, color="#475569")
    fig.text(.075, .067, "(a–b) Host probes reuse detector-training records in separate purged time partitions; (b) K = 43, n = 640.",
             fontsize=9.6, color="#475569")
    fig.text(.075, .025, "(c) Attack detection uses real validation sequences. All comparisons use detector seed 4266.",
             fontsize=9.6, color="#475569")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    for label in fig.findobj(matplotlib.text.Text):
        if label.get_visible() and label.get_text():
            box = label.get_window_extent(renderer)
            assert box.x0 >= bounds.x0 - 1 and box.x1 <= bounds.x1 + 1, label.get_text()
            assert box.y0 >= bounds.y0 - 1 and box.y1 <= bounds.y1 + 1, label.get_text()
    for path in OUTPUTS:
        fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)

    audit = {
        "scope": "S1 labels and layout revised from saved summary tables; no fitting or rescoring",
        "seed": 4266,
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest}
                    for p, digest in source_hashes.items()],
        "within_type_percent": within,
        "pooled_host_balanced_accuracy_percent": aggregate,
        "detection_validation_ap_percent": aps,
        "metrics_match_previous_figure": True,
        "web_label": "Brute_Force_-Web only, displayed as Web brute force",
        "outputs": [{"path": p.relative_to(ROOT).as_posix(),
                     "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in OUTPUTS],
    }
    PROVENANCE.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in source_hashes.items())
    print("Created PNG and SVG; all plotted metrics match the completed experiment.")
    print("\n".join(str(p) for p in OUTPUTS))


if __name__ == "__main__":
    main()
