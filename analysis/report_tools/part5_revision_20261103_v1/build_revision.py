"""Build an explanatory Part 5 revision from preserved completed evidence.

Run from any directory with the bundled Python runtime. This script does not
train, refit, rescore, or change any completed experiment or source document.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CIC_new").is_dir())
PLOT_DEPS = ROOT / "_local_archive/report_plot_dependencies_20261029"
if PLOT_DEPS.exists():
    sys.path.insert(0, str(PLOT_DEPS))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
    Table, TableStyle,
)

WEB = ROOT / "train/runs/20261025_013446_294043_web_temporal_support_diagnostic_7c620b"
HOST = ROOT / "train/runs/20261030_000611_945296_lstm_host_adversarial_seed4266_b1a7f9"
AUDIT = ROOT / "train/runs/20261024_122800_754357_pure_infilteration_raw_flow_audit_6146f6/audit.json"
SOURCES = {
    "latest_report": ROOT / "docs/newest.pdf",
    "baseline": ROOT / "docs/research/03_model_selection/results/five_seed_baseline.csv",
    "rotation": ROOT / "docs/research/03_model_selection/results/paired_shift_summary.csv",
    "type_inventory": ROOT / "docs/research/01_data_exploration/results/attack_type_inventory.csv",
    "web_profiles": WEB / "cohort_profiles.csv",
    "web_group_means": WEB / "web_recall_by_occupancy.csv",
    "web_seed_recalls": WEB / "web_recall_per_seed.csv",
    "web_past_context": WEB / "past_context_comparison.csv",
    "infilteration_audit": AUDIT,
    "host_probes": HOST / "probe_comparison.csv",
    "host_detection": HOST / "overall_comparison.csv",
    "temporal": ROOT / "docs/research/06_web_and_temporal_context/temporal/results/temporal_results.csv",
    "attack_visualization_png": ROOT / "docs/research/04_attack_profiles/figures/fig_attack_distributions.png",
    "attack_visualization_svg": ROOT / "docs/research/04_attack_profiles/figures/fig_attack_distributions.svg",
    "attack_visualization_protocol": ROOT / "docs/research/04_attack_profiles/results/figure_provenance.json",
    "attack_native_neighbors": ROOT / "docs/research/04_attack_profiles/results/native_neighbor_summary.csv",
}
BLUE = "#2F6B9A"
ORANGE = "#B96625"
INK = "#172B3A"
GRAY = "#697786"
LIGHT = "#E8EDF1"
FIG = HERE / "figures"
RESULTS = HERE / "results"
GENERATED: list[Path] = []
LAYOUT: dict[str, dict] = {}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(name: str) -> list[dict]:
    with SOURCES[name].open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_text(name: str, content: str) -> None:
    path = HERE / name
    path.write_text(content.strip() + "\n", encoding="utf-8")
    GENERATED.append(path)


def save_figure(fig, stem: str) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    overflow = []
    for item in fig.findobj(matplotlib.text.Text):
        if item.get_visible() and item.get_text():
            bounds = item.get_window_extent(renderer)
            if (bounds.x0 < -1 or bounds.y0 < -1 or
                    bounds.x1 > fig.bbox.x1 + 1 or bounds.y1 > fig.bbox.y1 + 1):
                overflow.append(item.get_text())
    if overflow:
        raise AssertionError(f"Figure text outside canvas: {stem}: {overflow}")
    LAYOUT[stem] = {"text_within_canvas": True, "axes": len(fig.axes)}
    for ext in ("png", "svg"):
        path = FIG / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, facecolor="white")
        GENERATED.append(path)
    plt.close(fig)


def plot_web_sparsity(profile: dict) -> None:
    total = int(profile["n"])
    counts = [int(profile["single_attack_bucket_n"]), int(profile["single_bucket_n"])]
    fig, ax = plt.subplots(figsize=(8.0, 2.85))
    fig.subplots_adjust(left=.30, right=.96, top=.72, bottom=.22)
    y = np.array([1, 0])
    ax.barh(y, [100, 100], height=.39, color=LIGHT)
    ax.barh(y, 100 * np.array(counts) / total, height=.39, color=BLUE)
    for pos, n in zip(y, counts):
        ax.text(2, pos, f"{n}/{total} sequences ({100*n/total:.1f}%)",
                va="center", fontsize=11, color="white", weight="bold")
    ax.set_yticks(y, ["Attack evidence in\nexactly one bucket", "All recorded activity in\nexactly one bucket"])
    ax.set_xlim(0, 100)
    ax.set_ylim(-.46, 1.48)
    ax.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100"])
    ax.set_xlabel("Share of the same 181 Web brute-force validation sequences (%)")
    ax.spines[["left", "top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=12)
    fig.text(.035, .92, "Attack evidence is usually confined to one bucket", weight="bold", fontsize=13, color=INK)
    fig.text(.035, .81, "The lower group is a subset of the upper group; the two rows are not added.", fontsize=10, color=GRAY)
    save_figure(fig, "fig_5_1_web_evidence_sparsity")


def plot_web_advantage(seed_rows: list[dict], means: dict) -> list[dict]:
    groups = [("one", "One active bucket", 96), ("two", "Two active buckets", 19),
              ("three_or_more", "Three or more active buckets", 66)]
    lookup = {(r["model"], int(r["seed"]), r["group"]): float(r["recall"]) for r in seed_rows}
    seeds = list(range(4266, 4271))
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    fig.subplots_adjust(left=.35, right=.97, top=.67, bottom=.24)
    ax.axvspan(-50, 0, color=BLUE, alpha=.045)
    ax.axvspan(0, 25, color=ORANGE, alpha=.055)
    ax.axvline(0, color=GRAY, linewidth=1.1)
    summary = []
    for y, (group, label, n) in zip([2, 1, 0], groups):
        delta = np.array([100*(lookup["lstm", seed, group] - lookup["xgboost", seed, group]) for seed in seeds])
        mean = float(delta.mean())
        color = BLUE if mean < 0 else ORANGE
        ax.hlines(y, 0, mean, color=color, linewidth=3, alpha=.55)
        ax.scatter(delta, y + np.linspace(-.13, .13, 5), s=32,
                   color=color, alpha=.75, edgecolors="white", linewidths=.5, zorder=3)
        ax.scatter([mean], [y], marker="D", s=90, color=color,
                   edgecolors="white", linewidths=.8, zorder=4)
        ax.text(mean, y+.27, f"Mean {mean:+.1f} points", ha="center", fontsize=11, color=color, weight="bold")
        assert abs(float(np.mean([lookup['lstm', s, group] for s in seeds])) - means['lstm',group]) < 1e-12
        assert abs(float(np.mean([lookup['xgboost', s, group] for s in seeds])) - means['xgboost',group]) < 1e-12
        expected_positive = {"one": 0, "two": 4, "three_or_more": 5}
        assert int(np.sum(delta > 0)) == expected_positive[group]
        summary.append({"group":group, "n":n, "lstm_minus_xgboost_mean_pp":mean,
                        "paired_seed_sd_pp":float(delta.std(ddof=1)),
                        "paired_seed_deltas_pp":delta.tolist(),
                        "lstm_mean_recall_percent":100*means['lstm',group],
                        "xgboost_mean_recall_percent":100*means['xgboost',group]})
    ax.set_yticks([2, 1, 0], [f"{label}\n(n = {n})\nXGBoost {100*means['xgboost',group]:.1f}% | LSTM {100*means['lstm',group]:.1f}%"
                             for group,label,n in groups], fontsize=9.5)
    ax.tick_params(axis="y", length=0, pad=12)
    ax.set_xlim(-50, 25)
    ax.set_ylim(-.53, 2.62)
    ax.set_xticks([-40, -20, 0, 20])
    ax.set_xlabel("Recall difference: LSTM minus XGBoost (percentage points)", labelpad=10)
    ax.spines[["left", "top", "right"]].set_visible(False)
    ax.grid(axis="x", color=LIGHT, linewidth=.7)
    fig.text(.035, .93, "The model advantage changes with observed activity", fontsize=13, weight="bold", color=INK)
    fig.text(.035, .84, "Left of zero: XGBoost detects more. Right of zero: LSTM detects more.", fontsize=10.5, color=INK)
    fig.text(.035, .76, "Small circles: five paired seeds. Diamond: their mean. Recall values are group means.", fontsize=9.8, color=GRAY)
    fig.text(.035, .06, "Each model uses its own global 1% FPR-budget threshold on the same validation cohort.", fontsize=9.5, color=GRAY)
    save_figure(fig, "fig_5_2_web_model_advantage")
    return summary


def box(ax, x, y, width, height, text, color=INK, fill="#F3F6F8", fontsize=10.5):
    ax.add_patch(FancyBboxPatch((x,y), width, height, boxstyle="round,pad=0.012,rounding_size=0.015",
                               edgecolor=color, facecolor=fill, linewidth=1.2))
    ax.text(x+width/2, y+height/2, text, ha="center", va="center", fontsize=fontsize, color=color, linespacing=1.4)


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12,
                                color=GRAY, linewidth=1.2, shrinkA=3, shrinkB=3))


def plot_collision(audit: dict) -> dict:
    pair = next(r for r in audit["collisions"] if r["outcome"] == "FN" and r["stage"] == "scaled_pre_encoder")
    assert pair["unscaled_numeric_exact"] and pair["original_category_counts_exact"]
    assert pair["unscaled_aggregate_and_histograms_exact"]
    fig, ax = plt.subplots(figsize=(8, 4.25))
    fig.subplots_adjust(left=.025, right=.975, top=.78, bottom=.04)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.axis("off")
    box(ax, .03, .78, .40, .17, f"Attack-labelled infiltration sequence\nSample {pair['sample_id']}", ORANGE)
    box(ax, .57, .78, .40, .17, f"Benign-labelled sequence\nSample {pair['example_benign_id']}", BLUE)
    box(ax, .20, .51, .60, .17, "Identical measured inputs before encoding\nNumeric aggregates and categorical counts match")
    arrow(ax, (.23,.78), (.37,.68))
    arrow(ax, (.77,.78), (.63,.68))
    box(ax, .20, .27, .60, .13, "Same embedding, intensity and mask\nSame deterministic detector score")
    arrow(ax, (.5,.51), (.5,.40))
    box(ax, .035, .025, .43, .14, "Flag both\nDetect the attack; raise a false alarm", ORANGE, fontsize=10)
    box(ax, .535, .025, .43, .14, "Reject both\nAvoid that false alarm; miss the attack", BLUE, fontsize=10)
    arrow(ax, (.43,.27), (.25,.165))
    arrow(ax, (.57,.27), (.75,.165))
    fig.text(.035,.94,"Identical inputs force a detection / false-alarm tradeoff", fontsize=13, weight="bold", color=INK)
    fig.text(.035,.85,"One audited pair illustrates the limit; the imported flow labels differ.", fontsize=10.5, color=GRAY)
    save_figure(fig,"fig_5_3_infiltration_input_collision")
    return pair


def plot_host_baseline(probes: dict) -> None:
    tasks = [("attack_type:Bot","Bot"), ("attack_type:Brute_Force_-Web","Web brute force"),
             ("attack_type:Infilteration","Infilteration")]
    fig, ax = plt.subplots(figsize=(8,4.0))
    fig.subplots_adjust(left=.36,right=.97,top=.69,bottom=.23)
    for y,(task,label) in zip([2,1,0],tasks):
        r = probes["lambda0", task]
        learned = 100*float(r["balanced_accuracy"])
        guess = 100*float(r["uniform_chance"])
        ax.hlines(y, guess, learned, color="#B9C5CF", linewidth=2)
        ax.scatter([guess],[y],marker="|",s=150,color=GRAY,zorder=3)
        ax.scatter([learned],[y],s=65,color=BLUE,zorder=4)
        ax.text(learned+2,y+.03,f"Probe {learned:.1f}%",va="bottom",color=BLUE,fontsize=10.5,weight="bold")
        ax.text(guess,y-.29,f"Uniform guess {guess:.1f}%",ha="left",fontsize=9.5,color=GRAY)
    labels = [f"{label}\n{probes['lambda0',task]['hosts']} candidate hosts; {probes['lambda0',task]['evaluation_n']} records" for task,label in tasks]
    ax.set_yticks([2,1,0],labels)
    ax.tick_params(axis="y",length=0,pad=12)
    ax.set_xlim(0,100)
    ax.set_ylim(-.6,2.52)
    ax.set_xlabel("Mean identification recall across candidate source hosts (%)")
    ax.spines[["left","right","top"]].set_visible(False)
    ax.grid(axis="x",color=LIGHT)
    fig.text(.035,.94,"Source identity remains recoverable within an attack type",fontsize=13,weight="bold",color=INK)
    fig.text(.035,.84,"Task: identify which eligible source host produced a sequence, not whether it is malicious.",fontsize=10,color=INK)
    fig.text(.035,.76,"Uniform guess = selecting each candidate host with equal probability (1 / number of hosts).",fontsize=9.7,color=GRAY)
    fig.text(.035,.055,"Purged probe partitions reuse detector-training records; baseline LSTM seed 4266.",fontsize=9.5,color=GRAY)
    save_figure(fig,"fig_c_1_within_type_host_identification")


def plot_adversarial(probes: dict, detection: dict, metric: str) -> None:
    variants = ["lambda0","lambda0.01","lambda0.1"]
    is_host = metric == "host"
    if is_host:
        values = [100*float(probes[v,"attack"]["balanced_accuracy"]) for v in variants]
        title = "Stronger suppression reduces pooled host identification"
        subtitle = "Separate host probe: the same 43 candidate attack hosts and 640 evaluation records."
        ylabel = "Host-balanced\naccuracy (%)"
        note = "Probe partitions reuse detector-training records. Uniform guessing among 43 hosts = 2.33%."
        stem = "fig_c_2_adversarial_host_identification"
        delta = values[-1]-values[0]
    else:
        values = [100*float(detection[v]["ap"]) for v in variants]
        title = "Stronger suppression also reduces attack-detection AP"
        subtitle = "Attack detection on the same real validation cohort; detector seed 4266."
        ylabel = "Detection AP (%)"
        note = "Single-seed diagnostic; the vertical scale is restricted to show the observed AP differences."
        stem = "fig_c_3_adversarial_detection_ap"
        delta = values[-1]-values[0]
    fig,ax=plt.subplots(figsize=(8,3.4))
    fig.subplots_adjust(left=.16,right=.96,top=.69,bottom=.25)
    ax.plot([0,1,2],values,color="#A1AFBA",linewidth=1.5)
    ax.scatter([0,1,2],values,color=[BLUE,GRAY,ORANGE],s=75,zorder=3)
    for x,value in enumerate(values):
        ax.annotate(f"{value:.2f}%",(x,value),xytext=(0,10),textcoords="offset points",ha="center",fontsize=11,weight="bold",color=INK)
    ax.set_xticks([0,1,2],["Baseline\nNo suppression","Weak suppression\nWeight 0.01","Stronger suppression\nWeight 0.1"])
    ax.set_xlim(-.35,2.35)
    ax.set_ylim((0,60) if is_host else (58.5,63.2))
    ax.set_ylabel(ylabel)
    ax.grid(axis="y",color=LIGHT)
    fig.text(.035,.94,title,fontsize=13,weight="bold",color=INK)
    fig.text(.035,.83,subtitle,fontsize=10,color=GRAY)
    fig.text(.035,.75,f"Baseline to stronger suppression: {delta:+.2f} percentage points.",fontsize=10,color=ORANGE,weight="bold")
    fig.text(.035,.055,note,fontsize=9.2,color=GRAY)
    save_figure(fig,stem)


def figure(stem: str, caption: str) -> str:
    return f"![{caption.split('. ', 1)[0]}](figures/{stem}.png)\n\n{caption}"


def build_main(pair: dict) -> str:
    baseline = rows("baseline")
    table = ["| Model | AP (%) | Recall at 1% FPR budget (%) | Recall at 0.1% FPR budget (%) |",
             "|---|---:|---:|---:|"]
    for r in baseline:
        cells = [r["model"]] + [f"{float(r[a]):.2f} +/- {float(r[b]):.2f}" for a,b in [
            ("ap_mean_percent","ap_sd_percentage_points"),
            ("recall_1pct_mean_percent","recall_1pct_sd_percentage_points"),
            ("recall_0_1pct_mean_percent","recall_0_1pct_sd_percentage_points")]]
        table.append("| " + " | ".join(cells) + " |")
    return f"""
# 5 Phase I: Baseline Detection and Focused Error Analysis

Phase I compares detectors using the shared bucket representations and then examines two limits hidden by overall performance: sparse Web evidence and ambiguous infiltration inputs. These analyses explain the working baseline's detection behaviour before Phase II examines attacks excluded from training.

Each detector selects its checkpoint by validation average precision (AP), which summarizes precision-recall ranking across thresholds. For each false-positive rate (FPR) budget, the threshold maximizes validation recall subject to FPR <= 1% or <= 0.1%. One global threshold is used across all attack types for each model fit and budget. Recall is TP/(TP+FN), precision is TP/(TP+FP), and FPR is FP/(FP+TN), where positive sequences contain at least one attack-labelled flow. Thresholds are frozen before test evaluation, where actual FPR may differ from the validation budget. The comparisons below use the inspected real validation cohort; repeated seeds measure training variability on the same samples, not independent evaluation populations.

## 5.1 Detector Comparison and Baseline Selection

XGBoost, CNN, LSTM and CNN-LSTM use the same frozen bucket embeddings, intensity channel and activity masks, with equal total objective weights for the four real/superimposed benign/malicious training groups. Five seeds (4266-4270) are evaluated on 251,146 real validation sequences, including 917 positives (Table M1).

Table M1. Detector validation results, mean +/- sample SD across five seeds on the fixed real cohort. Subtype evaluation uses each fit's global threshold, without subtype-specific calibration.

{chr(10).join(table)}

The neural AP means are close. CNN has the highest mean AP and recall at the stricter budget, while LSTM has the highest mean recall at 1% and the lowest seed variation in recall at 0.1%. We retain the original-order LSTM as a working baseline, without claiming a consistently superior architecture.

To examine dependence on the sequence start position, all embedding channels, intensity values and masks were rotated together by five buckets: [1,...,10] becomes [6,...,10,1,...,5]. All four families were retrained with matching seeds, selecting checkpoints and thresholds separately. Overall AP changed little, but LSTM Web recall at the 1% budget fell in all five pairs, by 5.75 percentage points on average. Circular rotation preserves most local adjacency, so this measures adaptation to a rotated grid rather than isolating the value of temporal order.

<!-- pagebreak -->

Overall performance does not describe how individual attack types are represented. Figure D1 therefore examines the baseline LSTM's learned sequence representations, highlighting infiltration, Web brute force and DDoS on the same projection. Web and infiltration are distributed across broader regions, while DDoS is concentrated in compact groups. This contrast motivates examining the evidence available for individual attack types rather than interpreting aggregate AP as uniform detection ability.

{figure('fig_d1_attack_distributions', 'Figure D1. Attack-type visualization of baseline LSTM sequence representations (historical original-order seed 4266). All three panels use the same 4,917 real validation samples (917 attack-bearing and 4,000 sampled benign), the same two PCA components and camera, with sphereization enabled. Red highlights (a) Infilteration, (b) Brute_Force_-Web and (c) the three DDoS types; grey/black points are all remaining samples, including benign traffic and other attacks. Panel n denotes the highlighted sample count. This attack-enriched projection describes learned representations and does not measure detection performance.')}

A separate calculation in unit-normalized native 64D space supports the local-overlap distinction: the mean benign share among ten nearest neighbours is 20.72% for Web and 24.81% for infiltration, compared with 0% for each DDoS type in the same sampled cohort. These statistics are computed before projection; the picture alone does not establish class separability or explain the cause of errors.

We focus next on Web brute force and infiltration. Their full-source coverage is broader than that of most other types (561 and 609 source hosts, respectively; Section 3.1), permitting closer examination across sources. Broader coverage does not establish host independence: separate probes still recover source identity from frozen LSTM representations within attack types. Those probes reuse detector-training records in separate purged time partitions; their protocol and results are in Appendix C.1. Here, Web denotes Brute_Force_-Web, not the complete Web family used in Phase II; infiltration corresponds to the dataset label Infilteration.

<!-- pagebreak -->

## 5.2 Sparse Web Evidence and Temporal Context

We first ask how much cross-bucket evidence is available to the detector. Of 181 Web brute-force validation sequences, 171 contain attack-labelled flows in only one bucket; 96 contain only one active bucket in total (Figure 5.1). An attack-bearing bucket contains an attack-labelled flow, whereas an active bucket contains any flow. A sequence can therefore have one attack-bearing bucket alongside benign activity in other buckets.

{figure('fig_5_1_web_evidence_sparsity', 'Figure 5.1. Two measures of sparse Web evidence on the same 181 real validation sequences. Blue shows sequences satisfying each condition; the light remainder does not. The 96 sequences with only one active bucket are a subset of the 171 with only one attack-bearing bucket. These are overlapping descriptions, not two disjoint groups.')}

Does model performance differ with the amount of observed activity? Figure 5.2 compares LSTM and XGBoost on the same occupancy groups using five original-order seeds. In the single-active-bucket group, XGBoost has higher mean recall (64.79% versus 38.75%). LSTM has higher mean recall in the two-bucket group (23.16% versus 12.63%) and the group with three or more active buckets (34.55% versus 24.24%). XGBoost leads in all five single-bucket seed pairs; LSTM leads in four of five two-bucket pairs and all five pairs with three or more buckets. Absolute recall remains limited.

{figure('fig_5_2_web_model_advantage', 'Figure 5.2. Recall difference by observed activity: LSTM minus XGBoost, in percentage points. Negative values favour XGBoost; positive values favour LSTM. Each small circle is a matched seed (4266-4270), and each diamond is the five-seed mean. Each fit uses its own global 1% validation FPR-budget threshold. Group sizes are fixed across seeds; circles show training variability, not confidence intervals over new samples.')}

These observational groups suggest complementary detection behaviour: some isolated Web events favour the tree model, while the LSTM performs better in the observed multi-bucket groups. They do not establish that adding context causes the difference. Both models receive time positions and masks, and sparse benign traffic also exists.

Longer history only partly changes the available evidence. A raw-flow diagnostic extends the past window from 100 to 400 seconds at fixed 10-second resolution for 143 eligible Web anchors, keeping their endpoints unchanged. Median active buckets rise from one to three, but 47/143 anchors remain single-bucket. This measures available traffic, not newly evaluated recall. Separate detector fits at other temporal geometries rebuild the cohorts and are reported as supplementary comparisons in Appendix C.3.

<!-- pagebreak -->

## 5.3 Infiltration Input Ambiguity

We next ask whether difficult infiltration examples remain distinguishable in the measured inputs. Among 138 pure infiltration validation sequences, whose flows are all attack-labelled, the historical LSTM detects 104 and misses 34 at its frozen 1% FPR-budget threshold. Reconstructing the raw bucket statistics and searching all 250,229 benign validation sequences finds exact benign counterparts for ten misses. Equality already holds in unscaled numerical aggregates and categorical counts, before neural encoding.

{figure('fig_5_3_infiltration_input_collision', f'Figure 5.3. An audited input collision between missed infiltration sample {pair["sample_id"]} and benign sample {pair["example_benign_id"]}. Both have the same measured bucket inputs and therefore receive the same deterministic score. Any common threshold flags both or rejects both. This schematic illustrates verified input equality, not equality of the underlying raw flows or proof that either imported label is wrong.')}

These pairs impose a direct detection/false-alarm tradeoff on a classifier using the current inputs. Changing the downstream architecture cannot distinguish the paired labels without additional information. However, ten of the 104 detected pure infiltration sequences also have exact benign counterparts. Collisions therefore do not explain every miss, and detected collisions carry a corresponding false-alarm cost.

Other cases leave room for learning improvements: at least one alternative detector recovers 17 of the 34 LSTM misses. The evidence identifies both an information limit in the aggregated inputs and architecture-dependent recovery elsewhere. This is an exploratory audit of inspected validation records, with no model or threshold refitted; it is not an independent test result.

Together, the Web and infiltration analyses qualify the aggregate baseline: observed activity can be too sparse to provide much cross-bucket evidence, and different labels can share identical measured inputs. Phase II addresses a separate boundary by excluding selected attack types from training and assessing transfer under the protocol in Section 6.1.
"""


def build_supplementary() -> str:
    temporal = rows("temporal")
    table = ["| Window / bucket (s) | Real validation n | AP (%) | Web n | Web recall 1% / 0.1% (%) | Infiltration n | Infiltration recall 1% / 0.1% (%) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in temporal:
        table.append("| " + " | ".join([
            f"{r['sequence_seconds']} / {r['bucket_seconds']}" + (" [historical]" if r["case"] == "existing_baseline" else ""),
            f"{int(r['validation_n']):,}",f"{100*float(r['ap']):.2f}",r['web_n'],
            f"{100*float(r['web_recall_at_1pct']):.2f} / {100*float(r['web_recall_at_0.1pct']):.2f}",r['infilteration_n'],
            f"{100*float(r['infilteration_recall_at_1pct']):.2f} / {100*float(r['infilteration_recall_at_0.1pct']):.2f}"]) + " |")
    return f"""
# Appendix C Supplementary Phase I Diagnostics

## C.1 Source-Host Information in the Baseline

A host probe is a separate classifier trained to predict a source host from a frozen LSTM sequence representation. Its balanced accuracy is the mean identification recall across eligible hosts. It answers a different question from attack detection: whether source identity remains recoverable after explicit source identifiers have been excluded.

{figure('fig_c_1_within_type_host_identification', 'Figure C.1. Baseline host identification within three attack types. Each circle is the probe result; the short grey marker is uniform guessing among that row\'s eligible hosts. The guessing probabilities are 1/8 = 12.5% for Bot, 1/6 = 16.7% for Web brute force, and 1/22 = 4.5% for Infilteration. Candidate hosts and evaluation records differ between rows, so probe accuracy should not be used to rank attack-detection difficulty.')}

The probes use separate purged time partitions of reused detector-training records, with seed 4266. Eligible hosts are subsets with sufficient fitting and evaluation support; these counts are not the full-source counts in Section 3.1. Recoverable identity does not establish the causal contribution of host information to attack detection, or performance on unseen hosts.

<!-- pagebreak -->

## C.2 Partial Suppression and Detection Tradeoffs

An auxiliary adversarial objective discourages source-host identification during detector fitting, with weights 0, 0.01 and 0.1. Independent probes then evaluate each frozen representation. These weights are training settings, not FPR thresholds.

{figure('fig_c_2_adversarial_host_identification', 'Figure C.2. Pooled source-host identification after adversarial training, using the same 43 eligible attack hosts and 640 probe evaluation records. Host-balanced accuracy falls from 46.31% to 36.56% at weight 0.1. Uniform guessing among these 43 candidate hosts would yield 2.33% (100/43); this is an identification reference, not a false-positive rate. The probes reuse detector-training records in separate purged time partitions.')}

{figure('fig_c_3_adversarial_detection_ap', 'Figure C.3. Attack-detection validation AP for the same fitted variants. At weight 0.1, AP falls from 61.53% to 59.89%. This figure uses real validation sequences, whereas Figure C.2 uses reused detector-training records for host probing. The vertical axis spans 58.5%-63.2% to show the observed differences.')}

Suppression is partial and varies by type: Web host identification rises from 70.83% to 76.39% at weight 0.1. The single-seed results show a pooled suppression/detection tradeoff, not removal of host information or proof that host identity causes the detector's successes.

<!-- pagebreak -->

## C.3 Rebuilt Temporal Geometries

The common-anchor diagnostic in Section 5.2 holds eligible endpoints fixed and measures available traffic. The detector comparisons below answer a different question: how separately rebuilt pipelines behave at different window and bucket widths. Each new geometry refits preprocessing and its encoder and uses seed 4266, selecting its own checkpoint and global thresholds.

Table C.1. Historical temporal-scale comparisons. Recall pairs use global 1% / 0.1% validation FPR budgets. The historical 100/10 reference uses a different runtime from the four new fits and is not the five-seed mean in Table M1. Cohorts and subtype support change, so these are descriptive pipeline comparisons rather than a matched effect of longer history.

{chr(10).join(table)}

For example, 400/40 reaches 58.12% Web recall and 73.33% infiltration recall at the 1% budget, but infiltration recall is only 8.89% at 0.1%. Its validation cohort has 117 Web and 45 infiltration sequences, compared with 181 and 366 in the historical reference. These results do not establish an optimal duration or replace the working baseline.

"""


def build_integration() -> str:
    return """
# Integration Instructions for the Part 5 Revision

This package is an insertion draft based on docs/newest.pdf (28 pages, reviewed on 3 November 2026). The source PDF and completed experiments are unchanged. All replacement text below is in English. Figure D1 is retained; Figure 5.1-5.3 and C.1-C.3 are provisional labels. Apply the final document's numbering consistently. Appendix C is proposed because the other editing session reserved Appendix B for the split EDA figures.

## Replacement boundary

Replace everything from "5 Establishing the Detection Baseline" to immediately before "6. Phase II - Generalisation to Unseen Attack Types" with PART5_REPLACEMENT.md. Keep Part 6 and its current tables. Insert SUPPLEMENTARY_DIAGNOSTICS.md after the existing appendices, adjusting the appendix letter if needed.

## Structure and figure migration

| Current item | Revised treatment |
|---|---|
| 5.1 and Table M1 | Retain the comparison, compress rotation and state the qualified baseline choice. |
| 5.2 / Figure D1 | Retain D1 unchanged in the main text, at the end of 5.1 after model comparison and rotation. Explain the contrasting attack-type distributions, distinguish native 64D neighbour statistics from the projection, and use the result to introduce the focused Web/infiltration analyses. Keep the following host-information paragraph brief. |
| Figure S1 | Remove from the main text. Figures C.1, C.2 and C.3 independently explain baseline host recoverability, pooled suppression and detection AP. |
| 5.3 / Figure W1(a) | New 5.2 and Figure 5.1; explain overlapping sparsity measures and their shared denominator. |
| Figure W1(b) | New Figure 5.2; plot paired LSTM-minus-XGBoost recall differences and retain absolute recalls in direct labels. Five small circles show individual seeds rather than visually ambiguous error bars. |
| Figure T1(a) | Retain the Web matched-anchor finding in prose; remove the composite chart. |
| Figure T1(b-c) | Appendix C.3, Table C.1; keep changed cohort sizes and the historical runtime warning adjacent to results. |
| Current 5.4 | New 5.3; add Figure 5.3 to explain a verified input collision and the resulting decision tradeoff. |
| Current 5.5 / Table V1 / Figure I2 | Remove from the revised report narrative. Completed experiments remain preserved in their original research directories. |

The six newly drawn explanatory figures are separate PNG and editable SVG files. The original D1 PNG/SVG is retained byte-for-byte, with its three panels comparing highlights on the same projection. Its red points represent the selected attack type(s); grey/black points include both benign samples and other attacks. Captions specify the task, population and denominator. Figure C.1 uses "uniform guessing" and lists candidate-host counts; the old 2.33% figure is explained in C.2 as 100/43, not an attack-detection FPR. The Web labels refer only to Brute_Force_-Web. Preserve the imported label Infilteration when referring to recorded data.

## Part 3 cross-reference from the other session

In the revised EDA temporal-scale paragraph, replace:

> Detector-level temporal-scale comparisons are reported in Section 5.3.

with:

> A matched-anchor analysis of available Web context is reported in Section 5.2; separately rebuilt detector pipelines at other temporal scales are reported in Appendix C.3.

This does not alter the agreed Part 3 findings or the proposed EDA figure sequence.

## Section 4.3 final paragraph

Replace the paragraph beginning "This bucket-level visualization examines..." with:

> This projection describes individual bucket embeddings rather than detection performance. Figure D1 in Section 5.1 instead examines sequence representations learned by the baseline LSTM. The focused analyses in Sections 5.2 and 5.3 then examine observed activity and ambiguity in aggregated inputs.

This updates the reference to the retained D1 and removes the obsolete reference to Projector views in Section 6.4.

## Part 6 opening

Replace only the introductory paragraph immediately under the Part 6 heading with:

> Phase I established the working LSTM baseline and examined detection limits among attack types represented in training. Phase II asks whether detection transfers to types excluded from training. We remove selected primary attack labels from the training fold, rebuild and retrain the pipeline, and evaluate the excluded types on the unchanged test split. These types remain available in validation for checkpoint and threshold selection, so this measures transfer after training exclusions under that selection policy, rather than fully unseen-attack or zero-day generalization.

The protocol details and numerical results in Sections 6.1-6.4 are outside this Part 5 edit. Review categorical causal wording there separately, particularly claims that subtype resemblance or traffic volume alone explains transfer.

## Section 7.1 second paragraph

Keep the first paragraph about nonlinearity and physical flow merging. Replace the second paragraph with:

> The EDA in Section 3.1 shows that natural benign-attack mixtures are unevenly distributed across attack types. Several flooding and brute-force types occur only in attack-only windows at the examined scales, whereas infiltration frequently shares windows with benign traffic. Superimposition introduces additional background combinations, but their realism and benefit require separate assessment. Reusing recorded training donors does not supply independent attack events, and the present detector comparison does not isolate the effect of augmentation.

This removes the old Section 6.3 mixing-diagnostic reference and does not imply that the deleted interventions demonstrated an augmentation benefit.

## Section 7.2 complete replacement

### 7.2 Temporal Context and Sparse Evidence

> The close neural AP means in Section 5.1 support retaining LSTM as a working baseline without establishing a consistently superior architecture. Section 5.2 shows that the relative advantage depends on observed activity: XGBoost performs better for single-active-bucket Web sequences, while LSTM performs better in the observed multi-bucket groups. This qualifies the value of a temporal model when much of the attack evidence is confined to one bucket. The comparison is observational and does not show that adding context causes better detection. Both models receive time-position information, and sparse activity also occurs in benign traffic.
>
> Longer history adds recorded activity for some matched anchors, but a sparse subgroup remains. Separately rebuilt temporal geometries change cohort composition and cannot isolate the effect of duration (Appendix C.3). Similarly, circular rotation preserves most local adjacency and is not a test that removes temporal information. A method combining local and contextual evidence is a reasonable next hypothesis, but the present results do not evaluate that combination.
>
> The host probes in Appendix C.1 also show that excluding explicit identifiers does not eliminate recoverable source information. Partial adversarial suppression changes both host identification and detection AP (Appendix C.2), but does not establish how much attack detection causally depends on host information.

## Section 7.3 complete replacement

### 7.3 Information Limits and Learning Opportunities

> The exact benign counterparts in Section 5.3 identify a concrete limit of the current representation. Some differently labelled sequences are identical in their measured inputs before encoding, so a deterministic detector must assign each pair the same score. A downstream architecture change alone cannot distinguish those pairs; additional discriminative information would be needed.
>
> This is a limit on the available inputs, not a complete explanation of infiltration errors or evidence that the imported labels are incorrect. Collisions occur among both missed and detected pure infiltration sequences, while alternative detectors recover some LSTM misses. Better use of the existing evidence remains possible for other cases. Both analyses use inspected validation data and should be treated as exploratory findings requiring independent confirmation.

## Section 7.4 complete replacement

### 7.4 Scope of Transfer

> Section 6.3 shows that detection after training exclusions depends on which related patterns remain available. Pooled recall changes little when selected subtypes are removed but their siblings remain, whereas removing the flooding or Web families produces larger losses. These results are consistent with transfer supported by related training coverage. Differences in subtype support, source recurrence and traffic volume prevent attributing the outcomes to any one mechanism alone.
>
> Withheld types remain in validation for checkpoint and threshold selection. The experiment therefore measures transfer after training exclusions under that selection policy; it does not establish fully unseen-attack or zero-day generalization. The seed spread measures retraining variability on fixed evaluation samples rather than uncertainty over new hosts or campaigns.

This removes the stale "separate flooding diagnostic" claim, which is not presented in the current Part 6, and keeps the discussion tied to the retained evidence.

## Section 7.5 complete replacement

### 7.5 Limitations

> The available data associate many attack types with very few source hosts. Although Web brute force and infiltration have broader full-source coverage, most other types involve at most ten sources. Hosts and campaigns can recur across the blocked split, so high recall does not establish host-independent attack recognition. The host probes reuse detector-training records and demonstrate recoverable identity rather than performance on new hosts. Family exclusions also remove attack patterns, so they do not isolate host identity as a cause of transfer losses.
>
> Further limits include uncertainty in imported labels, dependence between synthetic examples that reuse donors, the realism of embedding-space superimposition, single-seed supplementary interventions, and repeated analysis of the same validation cohort. Full-source host counts and probe-eligible host counts describe different populations. The baseline evaluates held-out time blocks in the same environment, not a chronological future deployment or a new-host population. Historical test inspection also means that an independent final assessment requires a fresh holdout. More diverse attack sources and evaluation on unseen hosts and campaigns are needed to strengthen the conclusions.

## Part 8 complete replacement

### 8 Conclusion

> We developed a source-level intrusion classification pipeline using frozen bucket embeddings, a separate traffic-intensity channel and training-only vector superimposition. The neural detectors achieve similar validation AP, with the original-order LSTM retained as the working baseline. Focused error analysis qualifies the aggregate comparison: isolated Web evidence favours XGBoost in the single-active-bucket group, whereas LSTM is stronger in the observed multi-bucket groups; some infiltration sequences are indistinguishable from benign counterparts in the measured inputs before encoding.
>
> Detection after training exclusions remains strong when related subtypes are retained and weakens after broader family removal. These results support conditional transfer within the evaluated environment, while recoverable host information and validation exposure limit claims of general attack recognition. Future work should prioritize independent evaluation on more diverse, unseen hosts and campaigns, additional information for ambiguous inputs, and methods that combine local and contextual evidence.

## Final insertion check

Search the online document for S1, W1, T1, D1, V1, I2, "Section 6.1", "Section 6.2", "Section 6.3", "Section 6.4", "Section 5.2" and "Section 5.3". Update only references to relocated material; retain genuine references to the current holdout protocol and results. Remove claims about context deletion, attention weights, feature extensions, oracle flow removal and bucket localization if their supporting experiments are excluded. The baseline table, held-out results and completed research records must not be overwritten.

## Reproduction

Run build_revision.py using the bundled Python runtime. Matplotlib is loaded from the existing local plotting dependencies if available. It reads completed CSV/JSON evidence, exports explanatory PNG/SVG figures, writes insertion text and a PDF preview, and verifies hashes before and after. The generated results contain plotted values, source hashes and figure-layout checks; no model is fitted or scored.
"""


def inline(text: str) -> str:
    text = html.escape(text)
    return re.sub(r"\*\*(.+?)\*\*",r"<b>\1</b>",text)


def render_pdf(paths: list[Path]) -> Path:
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportBody",fontName="Times-Roman",fontSize=10.7,leading=14.2,spaceAfter=7))
    styles.add(ParagraphStyle(name="ReportCaption",fontName="Times-Roman",fontSize=9.3,leading=11.7,spaceAfter=10,textColor=colors.HexColor("#384653")))
    styles.add(ParagraphStyle(name="ReportTitle",fontName="Times-Bold",fontSize=15,leading=18,spaceAfter=12,keepWithNext=True))
    styles.add(ParagraphStyle(name="ReportSection",fontName="Times-Bold",fontSize=12.4,leading=15,spaceBefore=7,spaceAfter=8,keepWithNext=True))
    styles.add(ParagraphStyle(name="Cell",fontName="Times-Roman",fontSize=8.1,leading=10))
    story = []
    width = 504
    for doc_index,path in enumerate(paths):
        if doc_index:
            story.append(PageBreak())
        blocks = re.split(r"\n\s*\n",path.read_text(encoding="utf-8").strip())
        skip = False
        for index,block in enumerate(blocks):
            if skip:
                skip=False
                continue
            if block == "<!-- pagebreak -->":
                story.append(PageBreak())
            elif block.startswith("!["):
                image_path=HERE/re.search(r"\]\((.+?)\)",block).group(1)
                iw,ih=ImageReader(str(image_path)).getSize()
                im=Image(str(image_path),width=width,height=width*ih/iw)
                cap=Paragraph(inline(blocks[index+1]),styles["ReportCaption"])
                story.append(KeepTogether([im,Spacer(1,5),cap]))
                skip=True
            elif block.startswith("# "):
                story.append(Paragraph(inline(block[2:]),styles["ReportTitle"]))
            elif block.startswith("## "):
                story.append(Paragraph(inline(block[3:]),styles["ReportSection"]))
            elif block.startswith("|"):
                lines=block.splitlines()
                data=[[Paragraph(inline(c.strip()),styles["Cell"]) for c in l.strip().strip("|").split("|")]
                      for i,l in enumerate(lines) if i != 1]
                widths=[110,94,150,150] if len(data[0])==4 else [65,62,43,40,110,60,124]
                assert sum(widths) == width
                t=Table(data,colWidths=widths,repeatRows=1,hAlign="LEFT")
                t.setStyle(TableStyle([
                    ("VALIGN",(0,0),(-1,-1),"TOP"), ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#EAF0F4")),
                    ("LINEBELOW",(0,0),(-1,0),.6,colors.HexColor("#62798A")),
                    ("LINEBELOW",(0,-1),(-1,-1),.5,colors.HexColor("#62798A")),
                    ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
                    ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
                ]))
                story.extend([t,Spacer(1,10)])
            else:
                sty=styles["ReportCaption"] if block.startswith(("Figure ","Table ")) else styles["ReportBody"]
                story.append(Paragraph(inline(block),sty))
    def page_footer(canvas,doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#CBD5DD"))
        canvas.line(54,40,558,40)
        canvas.setFillColor(colors.HexColor("#617383"))
        canvas.setFont("Helvetica",8)
        canvas.drawString(54,28,"Part 5 insertion draft | 3 November 2026 | Existing evidence only")
        canvas.drawRightString(558,28,str(doc.page))
        canvas.restoreState()
    path=ROOT/"output/pdf/part5_revision_20261103_v1.pdf"
    path.parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(path),pagesize=(612,792),rightMargin=54,leftMargin=54,
                          topMargin=46,bottomMargin=53,title="Part 5 Revision: Baseline and Focused Error Analysis",
                          author="DSA4266 Project Team")
    doc.build(story,onFirstPage=page_footer,onLaterPages=page_footer)
    GENERATED.append(path)
    return path


def main():
    hashes = {name:digest(path) for name,path in SOURCES.items()}
    FIG.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10.5,"axes.labelsize":10.2,
                         "axes.spines.top":False,"axes.spines.right":False,"svg.fonttype":"none",
                         "axes.axisbelow":True})
    profile=next(r for r in rows("web_profiles") if r["group"]=="Web")
    assert int(profile['n']) == 181 and int(profile['single_bucket_n']) == 96 and int(profile['single_attack_bucket_n']) == 171
    means={(r['model'],r['group']):float(r['mean_recall']) for r in rows('web_group_means')}
    audit=json.loads(AUDIT.read_text(encoding="utf-8"))
    assert audit['pure_fn_n']==34 and audit['pure_tp_n']==104
    assert audit['all_four_missed_among_pure_lstm_fn']==17
    assert all(r['with_exact_benign_counterpart']==10 for r in audit['collision_summary'])
    assert not audit['training_rows_used'] and not audit['test_rows_read'] and not audit['model_or_threshold_refitted']
    probes={(r['variant'],r['task']):r for r in rows('host_probes')}
    detection={r['variant']:r for r in rows('host_detection')}
    for v in ['lambda0','lambda0.01','lambda0.1']:
        assert probes[v,'attack']['hosts']=='43' and probes[v,'attack']['evaluation_n']=='640'
        assert probes[v,'attack']['detector_training_data_reused']=='True'
    geometry=json.loads(SOURCES['attack_visualization_protocol'].read_text(encoding='utf-8'))
    assert geometry['cohort']['total']==4917 and geometry['cohort']['attack']==917
    assert geometry['settings']['sphereize'] is True
    neighbors={r['primary_attack']:r for r in rows('attack_native_neighbors')}
    column='mean_benign_fraction_among_10_unit_64d_neighbors'
    assert round(100*float(neighbors['Brute_Force_-Web'][column]),2)==20.72
    assert round(100*float(neighbors['Infilteration'][column]),2)==24.81
    assert all(float(r[column])==0 for name,r in neighbors.items() if name.startswith(('DDOS_','DDoS_')))
    for ext in ('png','svg'):
        original=SOURCES[f'attack_visualization_{ext}']
        retained=FIG/f'fig_d1_attack_distributions.{ext}'
        retained.write_bytes(original.read_bytes())
        assert digest(retained)==hashes[f'attack_visualization_{ext}']
        GENERATED.append(retained)
    LAYOUT['fig_d1_attack_distributions']={'retained_original':True,'panels':3,'source_bytes_unchanged':True}
    plot_web_sparsity(profile)
    differences=plot_web_advantage(rows('web_seed_recalls'),means)
    pair=plot_collision(audit)
    plot_host_baseline(probes)
    plot_adversarial(probes,detection,'host')
    plot_adversarial(probes,detection,'ap')
    write_text('PART5_REPLACEMENT.md',build_main(pair))
    write_text('SUPPLEMENTARY_DIAGNOSTICS.md',build_supplementary())
    write_text('INTEGRATION_UPDATES.md',build_integration())
    pdf_path=render_pdf([HERE/'PART5_REPLACEMENT.md',HERE/'SUPPLEMENTARY_DIAGNOSTICS.md'])
    for name,path in SOURCES.items():
        assert digest(path)==hashes[name],f"Source changed: {path}"
    summary={
        'operation':'Editorial rewrite and replotting of completed evidence; no training, refitting or rescoring',
        'source_pdf_unchanged':True,
        'source_records_unchanged':True,
        'scope':{'main':'Inspected real validation: baseline comparison and post-selection diagnostics',
                 'host':'Purged probe partitions of reused detector-training records; detection AP uses real validation',
                 'temporal':'Common-anchor availability differs from rebuilt-cohort detector fits',
                 'seeds':'Training variability on fixed samples, not independent sampling uncertainty'},
        'sources':[{"name":name,"path":path.relative_to(ROOT).as_posix(),"sha256":hashes[name]} for name,path in SOURCES.items()],
        'web_paired_differences':differences,
        'collision_example':pair,
        'figure_layout_checks':LAYOUT,
        'outputs':[{"path":p.relative_to(ROOT).as_posix(),"sha256":digest(p)} for p in GENERATED],
    }
    (RESULTS/'provenance_and_checks.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    with (RESULTS/'web_paired_differences.csv').open('w',encoding='utf-8',newline='') as f:
        fieldnames=[k for k in differences[0] if k!='paired_seed_deltas_pp']
        writer=csv.DictWriter(f,fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({k:r[k] for k in fieldnames} for r in differences)
    print(json.dumps({'bundle':str(HERE),'figures':len(LAYOUT),'unchanged_sources':len(SOURCES),
                      'pdf':str(pdf_path),'web_differences':differences},indent=2))


if __name__=='__main__':
    main()
