"""Standardize Projector screenshot orientation and draft a word-neutral insertion."""
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import compose_part6_projector_screenshots as layout
from PIL import Image


def main():
    out = layout.OUT / "orientation_v2"
    with Image.open(out / "A_cloud.jpg") as source:
        reflected = source.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        # The operation is an exact pixel permutation; no point or color is edited.
        if reflected.transpose(Image.Transpose.FLIP_TOP_BOTTOM).tobytes() != source.tobytes():
            raise ValueError("Vertical reflection is not exactly reversible")
        reflected.save(out / "A_cloud_y_reflected.png")
    layout.OUT = out
    layout.compose("fig_part6_official_projector_v2.png",
                   [("A_cloud_y_reflected.png", "(a) Set A: XSS (n=62)"),
                    ("B1_cloud.jpg", "(b) Set B-1: Flooding (n=202)"),
                    ("B2_cloud.jpg", "(c) Set B-2: XSS (n=62)")],
                   ["Web siblings retained", "All flooding types removed from training", "Entire web family removed"],
                   "Website PCA 2D, Sphereize on | Same 4,851 test IDs | Seed 4266 | A reflected vertically; model projections remain independent")
    provenance = {"source": "fresh screenshots from the official TensorFlow Embedding Projector website",
                  "data_changed": False, "models_retrained": False,
                  "label_change": "display-only invisible plot_label field removes repeated Attack text; original class labels are retained",
                  "orientation_change": "vertical reflection of the entire label-free Set A crop; equivalent to reversing the displayed vertical axis",
                  "reflection_exactly_reversible": True,
                  "cross_model_projection_alignment": False,
                  "panel_order": ["A", "B1", "B2"],
                  "settings": {"PCA_components": [1, 2], "sphereize": True, "color_map": "none", "seed": 4266,
                               "displayed_test_n": 4851, "selected_n": {"A": 62, "B2": 62, "B1": 202}},
                  "scope": "exploratory single-seed visualization of already-inspected test records; no training references are displayed"}
    (out / "orientation_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    body = (
        "Across three seeds, transfer was stronger with related training coverage. The same 62 XSS test sequences were detected at 96.8% "
        "with web siblings retained (Set A), but at 16.1% after removing the web family (B-2). Removing all "
        "flooding types preserved detection of GoldenEye, SlowHTTPTest and HOIC, while LOIC-HTTP and LOIC-UDP deteriorated.\n\n"
        "Figure G1 locates these held-out groups within the learned test representations. Visible grouping alone "
        "does not establish detectability: XSS occupies local groups in both arms despite their different recall. "
        "The projection provides spatial context; the three-seed tables establish detection performance.\n\n"
        "Removed flow volume did not predict the loss: B-2 removed only 0.1% of attack flows yet caused the largest "
        "decline. Together, these results support transfer to related variants, while detection of families "
        "absent from training remains limited."
    )
    caption = (
        "Figure G1. Single-seed LSTM test representations in Embedding Projector. Red highlights XSS (A, B-2) or "
        "flooding (B-1); gray/black includes benign and other attacks. All panels use the same 4,851 test IDs. "
        "Sphereized PCA projections are fitted separately; A is reflected vertically for display. "
        "Neither cross-panel distances nor detector boundaries can be inferred."
    )
    source = (ROOT.parent / "docs/report/textest_ieee_text_only_20261103.md").read_text(encoding="utf-8")
    original = source.split("\\subsection{Findings}\\label{sec:6-4}", 1)[1].split("\\section{Discussion and Limitations}", 1)[0].strip()
    def words(s):
        return len(re.findall(r"\S+", s))
    counts = {"original_section_6_4_whitespace_tokens": words(original),
              "replacement_body_words": words(body), "caption_words": words(caption),
              "replacement_plus_caption_words": words(body + " " + caption),
              "counting_note": "whitespace-delimited count against the local text conversion matching the supplied report; no existing report file is modified"}
    if counts["replacement_plus_caption_words"] > counts["original_section_6_4_whitespace_tokens"]:
        raise ValueError("Replacement plus caption exceeds the original text budget")
    draft = ["# Section 6.4 replacement with Figure G1", "",
             "Insert the figure after the first paragraph. Replace the existing Section 6.4 prose with the text below; the report's three-seed numerical summaries are retained. Do not replace them with the new local single-seed recall. The website projection is descriptive and cannot establish a sibling-transfer mechanism or a volume-only detector.", "",
             "## Replacement body", "", body, "",
             "## Figure caption", "", caption, "",
             "## Why these panels", "",
             "Panels (a) and (c) hold the 62 queried XSS sequences fixed while changing the training exclusion. They visualize representation structure alongside the controlled performance contrast. Panel (b) locates the held-out flooding family; it does not explain its internal subtype or volume differences. The separate three-panel B-1 subtype figure can remain supplementary.", "",
             "Gray/black combines benign and other attacks, so overlap with gray/black does not demonstrate overlap with benign traffic. No training references are shown, so the plots do not measure proximity to retained training siblings. Sphereizing changes vector lengths; the resulting outline does not establish the detector's reliance on volume. Keep the tables as the primary evidence for transfer.", "",
             "## Text budget", "", json.dumps(counts, indent=2), ""]
    (out / "SECTION6_4_REPLACEMENT.md").write_text("\n".join(draft), encoding="utf-8")
    (out / "text_budget.json").write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
