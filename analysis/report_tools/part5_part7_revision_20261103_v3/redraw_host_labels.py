"""Use explicit labels for completed diagnostics; retain every plotted value."""

from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CIC_new").is_dir())
DEPS = ROOT / "_local_archive/report_plot_dependencies_20261029"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def redraw_host_labels():
    data = json.loads((HERE / "results/plotted_values.json").read_text(encoding="utf-8"))
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":11, "axes.labelsize":11,
                         "axes.spines.top":False, "axes.spines.right":False,
                         "axes.axisbelow":True, "svg.fonttype":"none"})
    paths = []

    def save(fig, stem):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for t in fig.findobj(matplotlib.text.Text):
            if t.get_visible() and t.get_text():
                b = t.get_window_extent(renderer)
                assert b.x0 >= -1 and b.y0 >= -1 and b.x1 <= fig.bbox.x1+1 and b.y1 <= fig.bbox.y1+1, t.get_text()
        for ext in ("png", "svg"):
            p = HERE / "figures" / f"{stem}.{ext}"
            fig.savefig(p, dpi=300, facecolor="white")
            paths.append(p)
        plt.close(fig)

    stem = "fig_h1_within_type_host_identification"
    values = data[stem]["values"]
    rows = values["baseline_rows"]
    fig, ax = plt.subplots(figsize=(8, 3.7))
    fig.subplots_adjust(left=.35, right=.96, top=.86, bottom=.19)
    y = np.array([2,1,0])
    for offset, numbers, color, label in [(.15, values["uniform_guess_percent"], "#CBD3DA", "Random guessing"),
                                         (-.15, values["probe_percent"], "#3478A6", "Source classifier")]:
        ax.barh(y+offset, numbers, height=.27, color=color, label=label)
        for pos, value in zip(y+offset, numbers):
            ax.text(value+1.5, pos, f"{value:.1f}", va="center", fontsize=10)
    names = ["Bot", "Web brute force", "Infilteration"]
    ax.set_yticks(y, [f"{name}\n{r['hosts']} hosts; {r['evaluation_n']} sequences" for name,r in zip(names,rows)])
    ax.set_xlim(0,100)
    ax.set_xlabel("Average correct-identification rate across hosts (%)")
    ax.tick_params(axis="y", length=0, pad=10)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", alpha=.18)
    ax.legend(loc="lower left", bbox_to_anchor=(0,1.01), ncol=2, frameon=False, fontsize=10)
    save(fig, stem)

    for stem, is_host in [("fig_h2_adversarial_host_identification", True), ("fig_h3_adversarial_detection_ap", False)]:
        ys = data[stem]["values"]["percent"]
        fig, ax = plt.subplots(figsize=(8,3.1))
        fig.subplots_adjust(left=.17, right=.96, top=.93, bottom=.23)
        ax.plot(range(3), ys, color="#3478A6", marker="o", markersize=6, linewidth=1.6)
        for x,value in enumerate(ys):
            ax.annotate(f"{value:.2f}", (x,value), xytext=(0,9), textcoords="offset points", ha="center", fontsize=11)
        ax.set_xticks(range(3), ["0 (baseline)", "0.01", "0.1"])
        ax.set_xlabel("Penalty weight")
        ax.set_ylabel("Host identification\nAverage recall (%)" if is_host else "Validation AP (%)")
        ax.set_xlim(-.4,2.4)
        ax.set_ylim((0,60) if is_host else (58.5,63.2))
        ax.set_yticks([0,20,40,60] if is_host else [59,60,61,62,63])
        ax.grid(axis="y",alpha=.18)
        save(fig,stem)
    stem = "fig_i1_exact_benign_counterparts"
    values = data[stem]["values"]
    assert values["n"] == [34,104] and values["counts"] == [10,10]
    rates = 100*np.array(values["counts"])/values["n"]
    fig, ax = plt.subplots(figsize=(8,2.6))
    fig.subplots_adjust(left=.31,right=.96,bottom=.24,top=.94)
    ax.barh([1,0],rates,color="#3478A6",height=.43)
    for pos,count,n,value in zip([1,0],values["counts"],values["n"],rates):
        ax.text(value+1,pos,f"{count}/{n} ({value:.1f}%)",va="center",fontsize=11)
    ax.set_yticks([1,0],["Missed: 34 sequences","Detected: 104 sequences"])
    ax.set_xlim(0,50)
    ax.set_ylim(-.6,1.6)
    ax.set_xlabel("Sequences matching a benign sequence's inputs (%)")
    ax.tick_params(axis="y",length=0,pad=10)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x",alpha=.18)
    save(fig,stem)

    stem = "fig_i2_pure_infiltration_sparsity"
    values = data[stem]["values"]
    assert values["counts"] == {"pure_FN":[30,29],"pure_TP":[48,46]}
    fig, ax = plt.subplots(figsize=(8,2.6))
    fig.subplots_adjust(left=.28,right=.96,bottom=.23,top=.84)
    y = np.array([1,0])
    for group,offset,color,label in [("pure_FN",.16,"#C77D35","Missed: 34 sequences"),
                                     ("pure_TP",-.16,"#3478A6","Detected: 104 sequences")]:
        n = values["n"][group]
        counts = values["counts"][group]
        rates = 100*np.array(counts)/n
        ax.barh(y+offset,rates,height=.28,color=color,label=label)
        for pos,count,value in zip(y+offset,counts,rates):
            ax.text(value+1,pos,f"{count}/{n}",va="center",fontsize=10)
    ax.set_yticks(y,["One active bucket","At most two flows"])
    ax.set_xlim(0,108)
    ax.set_xticks([0,25,50,75,100])
    ax.set_xlabel("Share of missed or detected sequences (%)")
    ax.tick_params(axis="y",length=0,pad=10)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x",alpha=.18)
    ax.legend(loc="lower left",bbox_to_anchor=(0,1.01),ncol=2,frameon=False,fontsize=10)
    save(fig,stem)
    return paths


if __name__ == "__main__":
    print("\n".join(str(p) for p in redraw_host_labels()))
