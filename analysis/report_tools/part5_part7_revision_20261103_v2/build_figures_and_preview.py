"""Replot completed studies and preview the scoped Part 5 / Part 7 insertion.

No model fitting, threshold selection, prediction generation or source edits.
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
from matplotlib.patches import Patch
import numpy as np
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer

WEB = ROOT / "train/runs/20261025_013446_294043_web_temporal_support_diagnostic_7c620b"
HOST = ROOT / "train/runs/20261030_000611_945296_lstm_host_adversarial_seed4266_b1a7f9"
AUDIT = ROOT / "train/runs/20261024_122800_754357_pure_infilteration_raw_flow_audit_6146f6"
PROFILES = ROOT / "docs/research/04_attack_profiles"
SOURCES = {
    "original_report": ROOT / "docs/newest.pdf",
    "d1_png": PROFILES / "figures/fig_attack_distributions.png",
    "d1_svg": PROFILES / "figures/fig_attack_distributions.svg",
    "d1_protocol": PROFILES / "results/figure_provenance.json",
    "native_neighbors": PROFILES / "results/native_neighbor_summary.csv",
    "web_profiles": WEB / "cohort_profiles.csv",
    "web_group_means": WEB / "web_recall_by_occupancy.csv",
    "web_seed_recalls": WEB / "web_recall_per_seed.csv",
    "past_context": WEB / "past_context_comparison.csv",
    "anchor_activity": ROOT / "docs/research/06_web_and_temporal_context/temporal/results/common_anchor_activity.csv",
    "temporal": ROOT / "docs/research/06_web_and_temporal_context/temporal/results/temporal_results.csv",
    "host_probes": HOST / "probe_comparison.csv",
    "host_detection": HOST / "overall_comparison.csv",
    "input_audit": AUDIT / "audit.json",
    "input_profiles": AUDIT / "raw_flow_profiles.csv",
}
BLUE, ORANGE, GRAY = "#3478A6", "#C77D35", "#CBD3DA"
FIG = HERE / "figures"
RESULTS = HERE / "results"
OUTPUTS: list[Path] = []
PLOT_DATA: dict = {}
LAYOUT: dict = {}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(name: str) -> list[dict]:
    with SOURCES[name].open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def canvas(height=3.4, left=.19, bottom=.20, top=.91):
    fig, ax = plt.subplots(figsize=(8, height))
    fig.subplots_adjust(left=left, right=.96, bottom=bottom, top=top)
    return fig, ax


def finish(fig, stem: str, sources: list[str], values: dict):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bad = []
    for label in fig.findobj(matplotlib.text.Text):
        if label.get_visible() and label.get_text():
            bbox = label.get_window_extent(renderer)
            if bbox.x0 < -1 or bbox.y0 < -1 or bbox.x1 > fig.bbox.x1 + 1 or bbox.y1 > fig.bbox.y1 + 1:
                bad.append(label.get_text())
    assert not bad, (stem, bad)
    assert len(fig.axes) == 1
    LAYOUT[stem] = {"axes": 1, "text_within_canvas": True, "explanatory_text_boxes": 0}
    PLOT_DATA[stem] = {"sources": sources, "values": values}
    for ext in ("png", "svg"):
        path = FIG / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, facecolor="white")
        OUTPUTS.append(path)
    plt.close(fig)


def plot_sparsity():
    row = next(r for r in records("web_profiles") if r["group"] == "Web")
    n = int(row["n"])
    counts = [int(row["single_attack_bucket_n"]), int(row["single_bucket_n"])]
    assert (n, counts) == (181, [171, 96])
    fig, ax = canvas(2.8, left=.34, bottom=.23, top=.94)
    y = np.array([1, 0])
    ax.barh(y, [100, 100], color="#E9EEF2", height=.45)
    ax.barh(y, 100*np.array(counts)/n, color=BLUE, height=.45)
    for pos, count in zip(y, counts):
        ax.text(2, pos, f"{count}/{n} ({100*count/n:.1f}%)", va="center", color="white", fontsize=11, weight="bold")
    ax.set_yticks(y, ["One attack-bearing bucket", "One active traffic bucket"])
    ax.tick_params(axis="y", length=0, pad=12)
    ax.set_xlim(0, 100)
    ax.set_ylim(-.65, 1.65)
    ax.set_xlabel("Share of Web validation sequences (%)")
    ax.spines["left"].set_visible(False)
    finish(fig, "fig_w1_web_sparsity", ["web_profiles"], {"n": n, "counts": counts})


def plot_model_difference():
    seed_rows = records("web_seed_recalls")
    means = {(r["model"], r["group"]):float(r["mean_recall"]) for r in records("web_group_means")}
    lookup = {(r["model"], int(r["seed"]), r["group"]):float(r["recall"]) for r in seed_rows}
    groups = [("one", "1 (n = 96)"), ("two", "2 (n = 19)"), ("three_or_more", "3+ (n = 66)")]
    fig, ax = canvas(3.5, left=.23, bottom=.23, top=.88)
    ax.axvline(0, color="#637482", linewidth=1)
    values = {}
    for y, (group, label) in zip([2, 1, 0], groups):
        d = np.array([100*(lookup["lstm", s, group] - lookup["xgboost", s, group]) for s in range(4266, 4271)])
        mean = float(d.mean())
        color = BLUE if mean < 0 else ORANGE
        ax.barh(y, mean, height=.40, color=color, alpha=.52)
        ax.scatter(d, y+np.linspace(-.14, .14, 5), s=35, color=color, edgecolors="white", linewidths=.5, zorder=3)
        ax.text(mean + (-1.2 if mean < 0 else 1.2), y+.29, f"{mean:+.1f}",
                ha="right" if mean < 0 else "left", fontsize=11, color=color, weight="bold")
        assert np.isclose(mean, 100*(means["lstm", group]-means["xgboost", group]))
        assert sum(d > 0) == {"one":0, "two":4, "three_or_more":5}[group]
        values[group] = {"paired_seed_differences_pp": d.tolist(), "mean_pp": mean,
                         "lstm_recall_percent": 100*means["lstm",group],
                         "xgboost_recall_percent": 100*means["xgboost",group]}
    ax.set_yticks([2,1,0], [label for _,label in groups])
    ax.set_ylabel("Active traffic buckets")
    ax.set_xlabel("Recall difference: LSTM - XGBoost (percentage points)")
    ax.set_xlim(-50, 28)
    ax.set_ylim(-.6, 2.6)
    ax.grid(axis="x", alpha=.18)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=9)
    finish(fig, "fig_w2_web_recall_difference", ["web_seed_recalls", "web_group_means"], values)


def plot_anchor_activity():
    rows = records("anchor_activity")
    fig, ax = canvas(3.4, left=.14, bottom=.21, top=.93)
    values = {}
    for name, label, color, marker in [("Brute_Force_-Web", "Web brute force", BLUE, "o"),
                                       ("Infilteration", "Infiltration", ORANGE, "s")]:
        selected = sorted([r for r in rows if r["attack_type"]==name], key=lambda r:int(r["window_seconds"]))
        n = int(selected[0]["anchors"])
        xs = [int(r["window_seconds"]) for r in selected]
        ys = [float(r["median_active_buckets"]) for r in selected]
        ax.plot(xs, ys, marker=marker, linewidth=1.8, color=color, label=f"{label} (n = {n})")
        for x,y in zip(xs,ys):
            ax.annotate(f"{y:g}", (x,y), xytext=(0,7), textcoords="offset points", ha="center", color=color, fontsize=10)
        values[name] = {"n":n, "window_seconds":xs, "median_active_buckets":ys}
    assert values["Brute_Force_-Web"]["median_active_buckets"] == [1,2,3]
    assert values["Infilteration"]["median_active_buckets"] == [3,5,8]
    ax.set_xticks([100,200,400])
    ax.set_xlim(75,425)
    ax.set_ylim(0,10)
    ax.set_xlabel("Past-window duration (seconds)")
    ax.set_ylabel("Median active buckets")
    ax.grid(axis="y", alpha=.18)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    finish(fig, "fig_t1_matched_anchor_activity", ["anchor_activity"], values)


def plot_temporal(target: str):
    rows = records("temporal")
    prefix = "web" if target=="web" else "infilteration"
    fig, ax = canvas(4.0, left=.29, bottom=.16, top=.88)
    y = np.arange(len(rows))[::-1]
    loose = [100*float(r[f"{prefix}_recall_at_1pct"]) for r in rows]
    strict = [100*float(r[f"{prefix}_recall_at_0.1pct"]) for r in rows]
    for offset, values, color, label in [(.16,loose,BLUE,"1% FPR budget"),(-.16,strict,ORANGE,"0.1% FPR budget")]:
        bars = ax.barh(y+offset, values, height=.28, color=color, label=label)
        bars[0].set_hatch("///")
        bars[0].set_edgecolor("#555555")
        bars[0].set_linewidth(.6)
        for pos,value in zip(y+offset,values):
            ax.text(value+1,pos,f"{value:.1f}",va="center",fontsize=9.7,color="#263B4A")
    labels = [f"{r['sequence_seconds']}/{r['bucket_seconds']}" + (" [historical]" if i==0 else "")
              + f"\nn = {r[prefix+'_n']}" for i,r in enumerate(rows)]
    ax.set_yticks(y,labels)
    ax.tick_params(axis="y",length=0,pad=10)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0,100)
    ax.set_xlabel("Validation recall (%)")
    ax.set_ylabel("Window / bucket (seconds)",labelpad=10)
    ax.grid(axis="x",alpha=.18)
    ax.legend(handles=[Patch(facecolor=BLUE,label="1% FPR budget"),
                       Patch(facecolor=ORANGE,label="0.1% FPR budget")],
              loc="lower left",bbox_to_anchor=(0,1.01),ncol=2,frameon=False,fontsize=10)
    stem = "fig_t2_web_temporal_recall" if target=="web" else "fig_t3_infiltration_temporal_recall"
    finish(fig,stem,["temporal"],{"rows":rows,"recall_at_1pct_percent":loose,"recall_at_0_1pct_percent":strict})


def plot_host_baseline(probes):
    tasks = [("attack_type:Bot","Bot"),("attack_type:Brute_Force_-Web","Web brute force"),("attack_type:Infilteration","Infilteration")]
    selected = [probes["lambda0",t] for t,_ in tasks]
    fig, ax = canvas(3.7,left=.30,bottom=.19,top=.86)
    y = np.array([2,1,0])
    guess = [100*float(r["uniform_chance"]) for r in selected]
    measured = [100*float(r["balanced_accuracy"]) for r in selected]
    for offset, values, color, label in [(.15,guess,GRAY,"Uniform guessing"),(-.15,measured,BLUE,"Host probe")]:
        ax.barh(y+offset,values,height=.27,color=color,label=label)
        for pos,value in zip(y+offset,values):
            ax.text(value+1.5,pos,f"{value:.1f}",va="center",fontsize=10)
    ax.set_yticks(y,[f"{label}\nK = {r['hosts']}; n = {r['evaluation_n']}" for (_,label),r in zip(tasks,selected)])
    ax.set_xlim(0,100)
    ax.set_xlabel("Host-balanced accuracy (%)")
    ax.tick_params(axis="y",length=0,pad=10)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x",alpha=.18)
    ax.legend(loc="lower left",bbox_to_anchor=(0,1.01),ncol=2,frameon=False,fontsize=10)
    finish(fig,"fig_h1_within_type_host_identification",["host_probes"],{"baseline_rows":selected,"uniform_guess_percent":guess,"probe_percent":measured})


def plot_adversarial(probes,detection,is_host):
    variants = ["lambda0","lambda0.01","lambda0.1"]
    ys = [100*float(probes[v,"attack"]["balanced_accuracy"]) for v in variants] if is_host else [100*float(detection[v]["ap"]) for v in variants]
    fig,ax = canvas(3.1,left=.17,bottom=.23,top=.93)
    ax.plot(range(3),ys,color=BLUE,marker="o",markersize=6,linewidth=1.6)
    for x,value in enumerate(ys):
        ax.annotate(f"{value:.2f}",(x,value),xytext=(0,9),textcoords="offset points",ha="center",fontsize=11)
    ax.set_xticks(range(3),["0 (baseline)","0.01","0.1"])
    ax.set_xlabel("Adversarial objective weight")
    ax.set_ylabel("Host-balanced accuracy (%)" if is_host else "Validation AP (%)")
    ax.set_xlim(-.4,2.4)
    ax.set_ylim((0,60) if is_host else (58.5,63.2))
    ax.set_yticks([0,20,40,60] if is_host else [59,60,61,62,63])
    ax.grid(axis="y",alpha=.18)
    stem="fig_h2_adversarial_host_identification" if is_host else "fig_h3_adversarial_detection_ap"
    finish(fig,stem,["host_probes" if is_host else "host_detection"],{"variants":variants,"percent":ys})


def plot_collisions(audit):
    selected = [next(r for r in audit["collision_summary"] if r["stage"]=="scaled_pre_encoder" and r["group"]==group)
                for group in ["pure_FN","pure_TP"]]
    ns = [int(r["n"]) for r in selected]
    counts = [int(r["with_exact_benign_counterpart"]) for r in selected]
    assert ns==[34,104] and counts==[10,10]
    assert all(r["counterpart_search_benign_n"]==250229 for r in selected)
    fig,ax = canvas(2.6,left=.28,bottom=.24,top=.94)
    values=100*np.array(counts)/ns
    ax.barh([1,0],values,color=BLUE,height=.43)
    for y,n,count,value in zip([1,0],ns,counts,values):
        ax.text(value+1,y,f"{count}/{n} ({value:.1f}%)",va="center",fontsize=11)
    ax.set_yticks([1,0],["Missed (n = 34)","Detected (n = 104)"])
    ax.set_xlim(0,50)
    ax.set_ylim(-.6,1.6)
    ax.set_xlabel("Sequences with an exact benign input counterpart (%)")
    ax.tick_params(axis="y",length=0,pad=10)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x",alpha=.18)
    finish(fig,"fig_i1_exact_benign_counterparts",["input_audit"],{"n":ns,"counts":counts,"search_benign_n":250229})


def plot_input_sparsity():
    profiles=records("input_profiles")
    grouped={g:[r for r in profiles if r["group"]==g] for g in ["pure_FN","pure_TP"]}
    assert [len(grouped[g]) for g in grouped]==[34,104]
    counts={g:[sum(int(r["active_buckets"])==1 for r in group),sum(int(r["flows"])<=2 for r in group)]
            for g,group in grouped.items()}
    assert counts=={"pure_FN":[30,29],"pure_TP":[48,46]}
    fig,ax=canvas(2.6,left=.28,bottom=.23,top=.84)
    y=np.array([1,0])
    for group,offset,color,label in [("pure_FN",.16,ORANGE,"Missed (n = 34)"),("pure_TP",-.16,BLUE,"Detected (n = 104)")]:
        n=len(grouped[group])
        vals=100*np.array(counts[group])/n
        ax.barh(y+offset,vals,height=.28,color=color,label=label)
        for pos,count,value in zip(y+offset,counts[group],vals):
            ax.text(value+1,pos,f"{count}/{n}",va="center",fontsize=10)
    ax.set_yticks(y,["One active bucket","At most two flows"])
    ax.set_xlim(0,108)
    ax.set_xticks([0,25,50,75,100])
    ax.set_xlabel("Share within each infiltration outcome group (%)")
    ax.tick_params(axis="y",length=0,pad=10)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x",alpha=.18)
    ax.legend(loc="lower left",bbox_to_anchor=(0,1.01),ncol=2,frameon=False,fontsize=10)
    finish(fig,"fig_i2_pure_infiltration_sparsity",["input_profiles"],{"n":{"pure_FN":34,"pure_TP":104},"counts":counts})


def render_pdf():
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodyReport",fontName="Times-Roman",fontSize=10.8,leading=14.2,spaceAfter=8))
    styles.add(ParagraphStyle(name="CaptionReport",fontName="Times-Roman",fontSize=9.4,leading=12,spaceAfter=11,textColor=colors.HexColor("#3E5060")))
    styles.add(ParagraphStyle(name="SectionReport",fontName="Times-Bold",fontSize=13.2,leading=17,spaceBefore=8,spaceAfter=9,keepWithNext=True))
    styles.add(ParagraphStyle(name="SubsectionReport",fontName="Times-Bold",fontSize=11.6,leading=15,spaceBefore=9,spaceAfter=8,keepWithNext=True))
    story=[]
    for doc_index,name in enumerate(["PART5_AFTER_5_1.md","SECTION7_FROM_PART5.md"]):
        if doc_index:
            story.append(PageBreak())
        blocks=re.split(r"\n\s*\n",(HERE/name).read_text(encoding="utf-8").strip())
        skip=False
        for i,block in enumerate(blocks):
            if skip:
                skip=False
                continue
            if block.startswith("!["):
                path=HERE/re.search(r"\]\((.*?)\)",block).group(1)
                iw,ih=ImageReader(str(path)).getSize()
                image=Image(str(path),width=504,height=504*ih/iw)
                cap=Paragraph(html.escape(blocks[i+1]),styles["CaptionReport"])
                story.append(KeepTogether([image,Spacer(1,5),cap]))
                skip=True
            elif block.startswith("### "):
                story.append(Paragraph(html.escape(block[4:]),styles["SubsectionReport"]))
            elif block.startswith("## "):
                if block.startswith("## 7.2"):
                    story.append(PageBreak())
                story.append(Paragraph(html.escape(block[3:]),styles["SectionReport"]))
            else:
                story.append(Paragraph(html.escape(block),styles["BodyReport"]))
    def footer(canvas,doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#CBD5DD"))
        canvas.line(54,40,558,40)
        canvas.setFillColor(colors.HexColor("#617383"))
        canvas.setFont("Helvetica",8)
        canvas.drawString(54,28,"Scoped insertion: Part 5 after 5.1 and Part 7 diagnostics | 3 November 2026")
        canvas.drawRightString(558,28,str(doc.page))
        canvas.restoreState()
    path=ROOT/"output/pdf/part5_part7_revision_20261103_v2.pdf"
    path.parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(path),pagesize=(612,792),leftMargin=54,rightMargin=54,topMargin=46,bottomMargin=53,
                          title="Part 5 and Part 7: Scoped Revision",author="DSA4266 Project Team")
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    OUTPUTS.append(path)
    return path


def main():
    hashes={name:sha(path) for name,path in SOURCES.items()}
    FIG.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.labelsize":11,
                         "axes.spines.top":False,"axes.spines.right":False,"axes.axisbelow":True,"svg.fonttype":"none"})
    for ext in ("png","svg"):
        dest=FIG/f"fig_d1_attack_distributions.{ext}"
        dest.write_bytes(SOURCES[f"d1_{ext}"].read_bytes())
        assert sha(dest)==hashes[f"d1_{ext}"]
        OUTPUTS.append(dest)
    LAYOUT["fig_d1_attack_distributions"]={"original_retained_byte_for_byte":True,"panels":3}
    plot_sparsity()
    plot_model_difference()
    plot_anchor_activity()
    plot_temporal("web")
    plot_temporal("infilteration")
    probes={(r["variant"],r["task"]):r for r in records("host_probes")}
    detection={r["variant"]:r for r in records("host_detection")}
    assert all(probes[v,"attack"]["hosts"]=="43" and probes[v,"attack"]["evaluation_n"]=="640"
               and probes[v,"attack"]["detector_training_data_reused"]=="True" for v in ["lambda0","lambda0.01","lambda0.1"])
    plot_host_baseline(probes)
    plot_adversarial(probes,detection,True)
    plot_adversarial(probes,detection,False)
    audit=json.loads(SOURCES["input_audit"].read_text(encoding="utf-8"))
    assert not audit["model_or_threshold_refitted"] and not audit["test_rows_read"] and not audit["training_rows_used"]
    assert audit["pure_fn_n"]==34 and audit["pure_tp_n"]==104 and audit["all_four_missed_among_pure_lstm_fn"]==17
    assert all(r["unscaled_aggregate_and_histograms_exact"] for r in audit["collisions"] if r["stage"]=="scaled_pre_encoder")
    plot_collisions(audit)
    plot_input_sparsity()
    native={r["primary_attack"]:r for r in records("native_neighbors")}
    key="mean_benign_fraction_among_10_unit_64d_neighbors"
    assert round(100*float(native["Brute_Force_-Web"][key]),2)==20.72
    assert round(100*float(native["Infilteration"][key]),2)==24.81
    past={r["group"]:r for r in records("past_context")}
    assert past["Web"]["still_single_400s_n"]=="47" and past["Web"]["common_eligible_n"]=="143"
    assert past["Web_XGB_ge4_LSTM_le1"]["still_single_400s_n"]=="14" and past["Web_XGB_ge4_LSTM_le1"]["common_eligible_n"]=="17"
    captions={}
    for name in ["PART5_AFTER_5_1.md","SECTION7_FROM_PART5.md"]:
        text=(HERE/name).read_text(encoding="utf-8")
        assert "## 5.1 " not in text and "circular" not in text.lower()
        for line in text.splitlines():
            if re.match(r"^Figure [A-Z]\d+\.",line):
                label=line.split(".",1)[0]
                captions[label]=len(line.split())
                assert captions[label]<=40,(label,captions[label])
        for image_path in re.findall(r"!\[.*?\]\((.*?)\)",text):
            assert (HERE/image_path).is_file()
        OUTPUTS.append(HERE/name)
    OUTPUTS.append(HERE/"INTEGRATION_GUIDE.md")
    preview=render_pdf()
    for name,path in SOURCES.items():
        assert sha(path)==hashes[name],f"Source changed: {path}"
    data={"operation":"Replot and rewrite completed evidence; no fitting, threshold selection or test scoring",
          "scope":{"section_5_1":"unchanged and excluded","circular_shift":"unchanged and excluded",
                   "section_7":"host and input-audit analyses only; transfer and other discussion deferred"},
          "sources":[{"name":name,"path":p.relative_to(ROOT).as_posix(),"sha256":hashes[name]} for name,p in SOURCES.items()],
          "figures":PLOT_DATA,"layout_checks":LAYOUT,"caption_word_counts":captions,
          "outputs":[{"path":p.relative_to(ROOT).as_posix(),"sha256":sha(p)} for p in OUTPUTS]}
    (RESULTS/"provenance_and_checks.json").write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")
    for key in ["temporal","anchor_activity","web_seed_recalls","past_context"]:
        (RESULTS/f"{key}.csv").write_bytes(SOURCES[key].read_bytes())
    print(json.dumps({"bundle":str(HERE),"preview":str(preview),"source_files_unchanged":len(SOURCES),
                      "new_single_chart_figures":len(PLOT_DATA),"retained_D1":True,
                      "max_caption_words":max(captions.values())},indent=2))


if __name__=="__main__":
    main()
