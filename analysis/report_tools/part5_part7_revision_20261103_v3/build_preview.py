"""Validate preserved evidence and render the expanded Version 3 argument.

Saved values are preserved; H1-H3 use clearer labels. No training or rescoring.
"""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer
from redraw_host_labels import redraw_host_labels

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CIC_new").is_dir())
PREVIOUS = HERE.parent / "part5_part7_revision_20261103_v2"
DOCUMENTS = ["PART5_AFTER_5_1.md", "SECTION7_FROM_PART5.md"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_pdf() -> Path:
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportBody", fontName="Times-Roman", fontSize=10.8,
                              leading=14.2, spaceAfter=8, allowWidows=0, allowOrphans=0))
    styles.add(ParagraphStyle(name="ReportCaption", fontName="Times-Roman", fontSize=9.4,
                              leading=12, spaceAfter=11, textColor=colors.HexColor("#3E5060")))
    styles.add(ParagraphStyle(name="ReportSection", fontName="Times-Bold", fontSize=13.2,
                              leading=17, spaceBefore=8, spaceAfter=9, keepWithNext=True))
    styles.add(ParagraphStyle(name="ReportSubsection", fontName="Times-Bold", fontSize=11.6,
                              leading=15, spaceBefore=9, spaceAfter=8, keepWithNext=True))
    story = []
    for doc_index, name in enumerate(DOCUMENTS):
        if doc_index:
            story.append(Spacer(1, 12))
        blocks = re.split(r"\n\s*\n", (HERE / name).read_text(encoding="utf-8").strip())
        skip = False
        for i, block in enumerate(blocks):
            if skip:
                skip = False
                continue
            if block.startswith("!["):
                path = HERE / re.search(r"\]\((.*?)\)", block).group(1)
                iw, ih = ImageReader(str(path)).getSize()
                image_width = 456 if path.stem.startswith(("fig_i1_", "fig_i2_")) else 504
                picture = Image(str(path), width=image_width, height=image_width*ih/iw)
                caption = Paragraph(html.escape(blocks[i+1]), styles["ReportCaption"])
                story.append(KeepTogether([picture, Spacer(1, 5), caption]))
                skip = True
            elif block.startswith("### "):
                story.append(Paragraph(html.escape(block[4:]), styles["ReportSubsection"]))
            elif block.startswith("## "):
                story.append(Paragraph(html.escape(block[3:]), styles["ReportSection"]))
            else:
                story.append(Paragraph(html.escape(block), styles["ReportBody"]))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#CBD5DD"))
        canvas.line(54, 40, 558, 40)
        canvas.setFillColor(colors.HexColor("#617383"))
        canvas.setFont("Helvetica", 8)
        canvas.drawString(54, 28, "Version 3: Part 5 follow-up and Part 7 diagnostics | 3 November 2026")
        canvas.drawRightString(558, 28, str(doc.page))
        canvas.restoreState()

    path = ROOT / "output/pdf/part5_part7_revision_20261103_v3.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(path), pagesize=(612, 792), leftMargin=54, rightMargin=54,
                            topMargin=46, bottomMargin=53, title="Part 5 and Part 7: Developed Analysis",
                            author="DSA4266 Project Team")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return path


def main():
    previous_manifest = PREVIOUS / "results/provenance_and_checks.json"
    previous_hash = sha(previous_manifest)
    previous = json.loads(previous_manifest.read_text(encoding="utf-8"))
    for row in previous["sources"] + previous["outputs"]:
        assert sha(ROOT / row["path"]) == row["sha256"], row["path"]
    figure_dir = HERE / "figures"
    result_dir = HERE / "results"
    figure_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)
    outputs = []
    asset_checks = []
    expected_assets = {row["path"]:row["sha256"] for row in previous["outputs"]}
    for original in sorted((PREVIOUS / "figures").iterdir()):
        if original.suffix not in {".png", ".svg"}:
            continue
        assert sha(original) == expected_assets[original.relative_to(ROOT).as_posix()]
        target = figure_dir / original.name
        target.write_bytes(original.read_bytes())
        assert sha(target) == sha(original)
        outputs.append(target)
        asset_checks.append({"source":original.relative_to(ROOT).as_posix(),
                             "copy":target.relative_to(ROOT).as_posix(), "sha256":sha(target)})
    assert len(asset_checks) == 22
    for key in ["temporal", "anchor_activity", "web_seed_recalls", "past_context"]:
        source_row = next(r for r in previous["sources"] if r["name"] == key)
        target = result_dir / f"{key}.csv"
        target.write_bytes((ROOT / source_row["path"]).read_bytes())
        assert sha(target) == source_row["sha256"]
        outputs.append(target)
    plot_values = result_dir / "plotted_values.json"
    plot_values.write_text(json.dumps(previous["figures"], indent=2)+"\n", encoding="utf-8")
    outputs.append(plot_values)
    relabelled_paths = redraw_host_labels()
    relabelled_names = {p.name for p in relabelled_paths}
    relabelled_assets = [r for r in asset_checks if Path(r["copy"]).name in relabelled_names]
    asset_checks = [r for r in asset_checks if Path(r["copy"]).name not in relabelled_names]
    for row in relabelled_assets:
        row["original_sha256"] = row.pop("sha256")
        row["sha256"] = sha(ROOT / row["copy"])
        row["change"] = "Plain-language labels; plotted values unchanged"
    outputs.append(HERE / "redraw_host_labels.py")
    captions, word_counts, paragraphs = {}, {}, []
    for name in DOCUMENTS:
        text = (HERE / name).read_text(encoding="utf-8")
        old_text = (PREVIOUS / name).read_text(encoding="utf-8")
        assert "## 5.1 " not in text and "circular" not in text.lower()
        if name.startswith("SECTION7"):
            assert not any(t in text for t in ["Section 6", "Phase II", "zero-day", "holdout"])
        old_captions = [l for l in old_text.splitlines() if re.match(r"^Figure [A-Z]\d+\.", l)]
        new_captions = [l for l in text.splitlines() if re.match(r"^Figure [A-Z]\d+\.", l)]
        retained_old = [l for l in old_captions if not re.match(r"^Figure (?:H[123]|I[12])\.", l)]
        retained_new = [l for l in new_captions if not re.match(r"^Figure (?:H[123]|I[12])\.", l)]
        assert retained_old == retained_new
        for line in new_captions:
            captions[line.split(".", 1)[0]] = len(line.split())
            assert len(line.split()) <= 34
        for image_path in re.findall(r"!\[.*?\]\((.*?)\)", text):
            assert (HERE / image_path).is_file()
        body = [p for p in re.split(r"\n\s*\n", text) if not p.startswith(("#", "![", "Figure "))]
        paragraphs.append(set(body))
        word_counts[name] = {"including_captions":len(text.split()),
                             "body_only":sum(len(p.split()) for p in body),
                             "previous_including_captions":len(old_text.split())}
        outputs.append(HERE / name)
    repeated = [p for p in paragraphs[0] & paragraphs[1] if len(p.split()) > 10]
    assert not repeated
    outputs.append(HERE / "INTEGRATION_GUIDE.md")
    preview = render_pdf()
    outputs.append(preview)
    for row in previous["sources"]:
        assert sha(ROOT / row["path"]) == row["sha256"], row["path"]
    assert sha(previous_manifest) == previous_hash
    manifest = {
        "operation":"Plain-language diagnostic methods and labels; unchanged numerical evidence",
        "previous_manifest":{"path":previous_manifest.relative_to(ROOT).as_posix(), "sha256":previous_hash},
        "sources":previous["sources"], "unchanged_figure_assets":asset_checks,
        "relabelled_figure_assets":relabelled_assets,
        "caption_word_counts":captions, "word_counts":word_counts,
        "scope":{"section_5_1":"unchanged and excluded", "circular_shift":"unchanged and excluded",
                 "section_7":"host and input-ambiguity discussion only; Part 6 material excluded"},
        "identical_cross_section_body_paragraphs":len(repeated),
        "outputs":[{"path":p.relative_to(ROOT).as_posix(), "sha256":sha(p)} for p in outputs],
    }
    (result_dir / "provenance_and_checks.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"preview":str(preview), "source_files_unchanged":len(previous["sources"]),
                      "figure_assets_unchanged":len(asset_checks), "word_counts":word_counts,
                      "max_caption_words":max(captions.values())}, indent=2))


if __name__ == "__main__":
    main()
