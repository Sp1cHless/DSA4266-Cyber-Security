"""Create report figures and an offline interactive viewer from Part 6 exports."""
import argparse
import csv
import json
from pathlib import Path
import sys
import urllib.request

import numpy as np
from sklearn.decomposition import PCA

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.common.io import ROOT, read_json, write_json

sys.path.append(str(ROOT / "cache/diagnostic_plot_dependencies"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch


WEB = ["Brute_Force_-Web", "SQL_Injection"]
XSS = "Brute_Force_-XSS"
FLOODS = ["DoS_attacks-GoldenEye", "DoS_attacks-SlowHTTPTest", "DDOS_attack-HOIC",
          "DDoS_attacks-LOIC-HTTP", "DDOS_attack-LOIC-UDP"]
SHORT = ["GoldenEye", "SlowHTTPTest", "HOIC", "LOIC-HTTP", "LOIC-UDP"]
COLORS = ["#0072B2", "#009E73", "#CC79A7", "#D55E00", "#E69F00"]


def load(out, arm):
    folder = out / f"{arm}_projector"
    with (folder / "metadata.tsv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    a = np.load(folder / "representations.npz", allow_pickle=False)
    n = len(rows)
    result = {"rows": rows, "coords": a["coordinates"].copy(), "vectors": a["vectors"].copy(),
              "manifest": read_json(folder / "manifest.json")}
    result.update({k: np.array([r[k] for r in rows]) for k in ("split", "label", "primary_attack", "outcome_1pct")})
    result["flows"] = np.array([int(r["flow_count"]) for r in rows])
    result["train"] = result["split"] == "train"
    result["test"] = ~result["train"]
    result["benign"] = result["label"] == "Benign"
    checkpoint = torch.load(out / f"{arm}_lstm/model.pt", map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"]
    weight = state["head.3.weight"].numpy().ravel().astype(np.float64)
    bias = float(state["head.3.bias"].numpy()[0])
    vectors = result["vectors"].astype(np.float64)
    unit = weight / np.linalg.norm(weight)
    residual = vectors - (vectors @ unit)[:, None] * unit
    pca = PCA(n_components=2, svd_solver="full").fit(residual[result["train"]])
    t = result["manifest"]["thresholds"]["validation_fpr_1pct"]
    if not 0 < t < 1:
        raise ValueError("A finite logit threshold is needed")
    margin = vectors @ weight + bias - np.log(t / (1-t))
    predicted = margin >= 0
    expected = np.isin(result["outcome_1pct"], ["TP", "FP"])
    if not np.array_equal(predicted, expected):
        raise ValueError("Decision-aligned coordinate does not recover frozen predictions")
    result["decision"] = np.column_stack((margin, pca.transform(residual)))
    result["decision_orthogonal_variance"] = pca.explained_variance_ratio_.tolist()
    a.close()
    if n != len(result["vectors"]):
        raise ValueError("Metadata length mismatch")
    return result


def target(d, family):
    return d["test"] & (d["primary_attack"] == XSS if family == "web" else np.isin(d["primary_attack"], FLOODS))


def background(ax, d, xy):
    for keep, color, size, alpha in ((d["train"] & d["benign"], "#969CA4", 3, .13),
                                      (d["train"] & ~d["benign"], "#7199AD", 4, .14)):
        ax.scatter(xy[keep, 0], xy[keep, 1], c=color, s=size, alpha=alpha, rasterized=True, linewidths=0)


def plot_figure(out, arms, space):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 10.7))
    for ax, arm, letter in ((axes[0, 0], "A", "a"), (axes[0, 1], "B2", "b")):
        d = arms[arm]
        xy = d[space]
        background(ax, d, xy)
        siblings = d["train"] & np.isin(d["primary_attack"], WEB)
        if siblings.any():
            ax.scatter(xy[siblings, 0], xy[siblings, 1], s=9, c="#009E73", marker="D", alpha=.35,
                       rasterized=True, linewidths=0)
        keep = target(d, "web")
        for outcome, marker in (("TP", "o"), ("FN", "X")):
            selected = keep & (d["outcome_1pct"] == outcome)
            ax.scatter(xy[selected, 0], xy[selected, 1], s=42 if marker == "X" else 28,
                       c="#8B3591", marker=marker, alpha=.85, edgecolors="white", linewidths=.35, zorder=5)
        tp = int((keep & (d["outcome_1pct"] == "TP")).sum())
        count = int(keep.sum())
        setting = "Web siblings retained" if arm == "A" else "Entire web family removed"
        ax.set_title(f"({letter}) Set {arm}: {setting}\nXSS detected: {tp}/{count} ({tp/count:.1%}), seed 4266", loc="left", fontsize=11)
    d = arms["B1"]
    xy = d[space]
    for ax in axes[1]:
        background(ax, d, xy)
    for name, color, short in zip(FLOODS, COLORS, SHORT):
        for outcome, marker in (("TP", "o"), ("FN", "X")):
            selected = d["test"] & (d["primary_attack"] == name) & (d["outcome_1pct"] == outcome)
            axes[1, 0].scatter(xy[selected, 0], xy[selected, 1], s=37, c=color, marker=marker,
                               edgecolors="white", linewidths=.3, alpha=.85, zorder=4)
    flooding = target(d, "flood")
    logflows = np.log10(1 + d["flows"])
    norm = matplotlib.colors.Normalize(vmin=logflows[flooding].min(), vmax=logflows[flooding].max())
    for outcome, marker in (("TP", "o"), ("FN", "X")):
        selected = flooding & (d["outcome_1pct"] == outcome)
        sc = axes[1, 1].scatter(xy[selected, 0], xy[selected, 1], s=37, c=logflows[selected], cmap="viridis", norm=norm,
                               marker=marker, edgecolors="white", linewidths=.3, alpha=.9, zorder=4)
    tp = int((flooding & (d["outcome_1pct"] == "TP")).sum())
    axes[1, 0].set_title(f"(c) Set B-1: Flooding subtypes\nFlooding detected: {tp}/{flooding.sum()} ({tp/flooding.sum():.1%}), seed 4266", loc="left", fontsize=11)
    axes[1, 1].set_title("(d) Set B-1: Test-sequence traffic volume\nSame points, coordinates and outcome shapes as (c)", loc="left", fontsize=11)
    fig.subplots_adjust(left=.075, right=.95, top=.84, bottom=.17, hspace=.44, wspace=.24)
    # A separate horizontal colorbar keeps (c) and (d) at exactly the same axes dimensions.
    cbax = fig.add_axes([.61, .082, .29, .016])
    fig.colorbar(sc, cax=cbax, orientation="horizontal", label="log10(1 + test sequence flow count)")
    for i, (ax, arm) in enumerate(zip(axes.ravel(), ["A", "B2", "B1", "B1"])):
        ax.grid(alpha=.14, linewidth=.6)
        ax.set_axisbelow(True)
        if space == "coords":
            ev = arms[arm]["manifest"]["explained_variance_ratio"]
            ax.set_xlabel(f"Train-reference PC1 ({ev[0]:.1%})")
            ax.set_ylabel(f"Train-reference PC2 ({ev[1]:.1%})")
        else:
            ax.axvline(0, c="#4B4F54", linewidth=1, ls="--", zorder=2)
            ax.set_xlabel("Attack logit - frozen threshold logit")
            ax.set_ylabel("Decision-orthogonal PC1")
        ax.ticklabel_format(axis="both", style="sci", scilimits=(-3, 4))
    axes[1, 1].set_xlim(axes[1, 0].get_xlim())
    axes[1, 1].set_ylim(axes[1, 0].get_ylim())
    refs = [Line2D([], [], color="#969CA4", marker="o", ls="", markersize=5, label="Real train benign (sample)"),
            Line2D([], [], color="#7199AD", marker="o", ls="", markersize=5, label="Real train retained attacks"),
            Line2D([], [], color="#009E73", marker="D", ls="", markersize=5, label="Real train web siblings"),
            Line2D([], [], color="#8B3591", marker="o", ls="", markersize=5, label="Test XSS")]
    fig.legend(handles=refs, loc="upper center", bbox_to_anchor=(.5, .95), ncol=4, frameon=False, fontsize=9)
    outcomes = [Line2D([], [], color="#333333", marker="o", ls="", label="TP: detected"),
                Line2D([], [], color="#333333", marker="X", ls="", label="FN: missed")]
    fig.legend(handles=outcomes, loc="upper center", bbox_to_anchor=(.5, .918), ncol=2, frameon=False, fontsize=9)
    subtype_handles = [Line2D([], [], color=c, marker="o", ls="", label=s) for c, s in zip(COLORS, SHORT)]
    fig.legend(handles=subtype_handles, loc="lower left", bbox_to_anchor=(.075, .069), ncol=3, frameon=False, fontsize=9)
    title = "LSTM representation geometry after training exclusions"
    if space == "decision":
        title += "\nDecision-aligned projection"
    fig.suptitle(title, y=.987, fontsize=15, fontweight="bold")
    fig.text(.5, .016, "Single-seed exploratory test visualization; real training records reused as reference.\n"
             "Each model has its own training-fitted projection; cross-model coordinates are not aligned. Tables retain the report's three-seed estimates.",
             ha="center", fontsize=9, color="#4B4F54")
    stem = "fig_part6_pca" if space == "coords" else "fig_part6_decision_aligned"
    fig.savefig(out / f"{stem}.png", dpi=190, facecolor="white")
    fig.savefig(out / f"{stem}.svg", facecolor="white")
    plt.close(fig)


def web_view(out, arms):
    payload = {}
    for name, d in arms.items():
        selected = d["train"] | target(d, "web" if name != "B1" else "flood")
        ids = np.flatnonzero(selected)
        payload[name] = {"pca": d["coords"][ids].round(6).tolist(), "decision": d["decision"][ids].round(6).tolist(),
                         "rows": [d["rows"][i] for i in ids], "manifest": {"explained_variance_ratio": d["manifest"]["explained_variance_ratio"]}}
    script = out / "plotly-3.1.0.min.js"
    if not script.exists():
        urllib.request.urlretrieve("https://cdn.plot.ly/plotly-3.1.0.min.js", script)
    template = Path(__file__).with_name("part6_viewer_template.html").read_text(encoding="utf-8")
    html = template.replace("__PART6_DATA__", json.dumps(payload, separators=(",", ":")))
    (out / "part6_vector_space.html").write_text(html, encoding="utf-8")


def report_notes(out, arms):
    lines = ["# Part 6: Fixed-seed LSTM representation visualization", "",
             "## Outputs", "",
             "- `fig_part6_pca.png` and `.svg`: four-panel report figure, fitted on real training references only.",
             "- `fig_part6_decision_aligned.png` and `.svg`: complementary projection with an exact threshold direction.",
             "- `part6_vector_space.html`: offline 2D/3D interactive viewer; keep `plotly-3.1.0.min.js` beside it.",
             "- `{A,B1,B2}_projector/vectors.tsv` and `metadata.tsv`: native 64-dimensional vectors for TensorFlow Embedding Projector.",
             "- `{A,B1,B2}_projector/pca_coordinates.tsv`: precomputed training-fitted PCA coordinates. Load these instead of raw vectors to reproduce the report projection in Projector, without sphereizing or fitting another PCA.",
             "- `{A,B1,B2}_lstm/`: new baseline checkpoints and validation-selection records. Existing completed runs were not changed.", "",
             "## Measured results from the new local fits", "",
             "All runs use the original baseline architecture and training settings, seed 4266. These single-seed results are distinct from the report's three-seed means. Test was scored once per fit; no test-based threshold or checkpoint selection was performed.", "",
             "| Arm | Held-out subtype | Test n | TP | Recall at frozen validation 1% FPR threshold | Median test flows |", "|---|---|---:|---:|---:|---:|"]
    for arm in ("A", "B2", "B1"):
        for row in arms[arm]["manifest"]["per_subtype"]:
            if row["primary_attack"] in ([XSS] if arm != "B1" else FLOODS):
                lines.append(f"| {arm} | {row['primary_attack']} | {row['n']} | {row['tp']} | {row['recall']:.2%} | {row['median_test_flows']:g} |")
    lines += ["", "## Suggested insertion into Section 6.4", "",
              "Place the PCA figure after the first two findings, shortening their repetition of Section 6.3 to make room for the caption. Keep the existing three-seed numerical conclusions. The decision-aligned figure is a supplementary alternative, rather than a second required main-text figure.", "",
              "**Suggested caption:** LSTM sequence representations after training exclusions (seed 4266). (a-b) The same 62 XSS test sequences with web siblings retained or removed. (c-d) The same 202 flooding test sequences, coloured by subtype or flow count. Circles denote detections and crosses misses at frozen validation thresholds. PCA is fitted separately to real training references; cross-model coordinates are not aligned.", "",
              "## Scope and interpretation", "",
              "The figures reuse real training records as geometric references: all retained real train attacks and 4,000 real train benign sequences sampled with seed 4266. No synthetic training rows are displayed. The reference is attack-enriched, and its density and 20-neighbor fractions do not estimate population prevalence. Native-dimensional neighbor statistics in the manifests use this same sampled reference, not the full training population.", "",
              "Validation retains excluded attack types for checkpoint and threshold selection. Historical and current test inspection make this exploratory visualization, not independent zero-day validation. Source hosts are checked for matching evaluation populations but are not published in the interactive viewer. Related subtype proximity may also reflect host-associated behavior; geometry does not isolate the causal mechanism of transfer.", "",
              "Each arm refits the LSTM and uses its supplied, separately fitted bucket encoder. PCA is fitted on training references only, without standardization or sphereizing. Different model axes cannot be interpreted as literal sample movement. Panels (c) and (d) share identical coordinates and axis limits. The decision-aligned supplement projects exactly along the frozen final linear classifier and along training-fitted orthogonal PCs; its vertical threshold is valid, unlike a boundary inferred from ordinary PCA.", "",
              "## Storage and reproducibility", "",
              "All disposable mmap caches were created under the task-specific C: cache root and removed after each arm. Source datasets, previous results, and all completed model checkpoints were retained. This output is compact; no extra raw-data copy was made on D:.", "",
              "```powershell",
              '.\\.venv-cnn\\Scripts\\python.exe train/run_part6_visualizations.py --out <fresh-output-directory> --cache-root <fresh-C-drive-cache-directory>',
              f'.\\.venv-cnn\\Scripts\\python.exe train/plot_part6_visualizations.py --out "{out}"',
              "```", ""]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    arms = {k: load(out, k) for k in ("A", "B1", "B2")}
    # Check the exact plotted XSS IDs, in addition to the full-cohort hashes.
    matched = [[r["sample_id"] for i, r in enumerate(arms[k]["rows"]) if target(arms[k], "web")[i]] for k in ("A", "B2")]
    if matched[0] != matched[1] or len(matched[0]) != 62:
        raise ValueError("Expected the same 62 XSS test sequences in A and B2")
    if target(arms["B1"], "flood").sum() != 202:
        raise ValueError("Expected 202 flooding test sequences")
    for space in ("coords", "decision"):
        plot_figure(out, arms, space)
    web_view(out, arms)
    report_notes(out, arms)
    write_json(out / "figure_checks.json", {"matched_xss_n": 62, "flooding_n": 202,
                                            "decision_projection_recovers_all_saved_predictions": True,
                                            "projection_fit_excludes_test": True,
                                            "B1_panels_share_exact_coordinates": True})
    print("Created PCA and decision-aligned PNG/SVG figures, offline web viewer, and figure checks.")


if __name__ == "__main__":
    main()
