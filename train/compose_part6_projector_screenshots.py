"""Arrange unmodified official Projector screenshot crops with report labels."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "cache/diagnostic_plot_dependencies"))
from PIL import Image, ImageDraw, ImageFont

OUT = ROOT / "runs/20261103_part6_new_holdouts_lstm_seed4266_v1/official_projector_website_v1"
FONT = Path("C:/Windows/Fonts")


def compose(filename, panels, subtitles, footnote):
    margin, gap, width, height = 24, 24, 590, 545
    canvas = Image.new("RGB", (margin * 2 + width * len(panels) + gap * (len(panels)-1), 720), "white")
    draw = ImageDraw.Draw(canvas)
    bold = ImageFont.truetype(str(FONT / "arialbd.ttf"), 25)
    regular = ImageFont.truetype(str(FONT / "arial.ttf"), 19)
    small = ImageFont.truetype(str(FONT / "arial.ttf"), 17)
    for i, ((image, title), subtitle) in enumerate(zip(panels, subtitles)):
        x = margin + i * (width + gap)
        draw.text((x + width / 2, 16), title, fill="black", font=bold, anchor="mt")
        draw.text((x + width / 2, 50), subtitle, fill="#536570", font=regular, anchor="mt")
        with Image.open(OUT / image) as crop:
            if crop.size != (width, height):
                raise ValueError(f"Unexpected website screenshot dimensions: {image}")
            canvas.paste(crop.convert("RGB"), (x, 86))
    midpoint = canvas.width / 2
    draw.text((midpoint, 650), "Red: selected held-out attacks. Gray/black: remaining test samples, including benign and other attacks.",
              fill="#536570", font=small, anchor="mt")
    draw.text((midpoint, 678), footnote, fill="#536570", font=small, anchor="mt")
    canvas.save(OUT / filename, dpi=(190, 190))


def main():
    compose("fig_part6_official_projector.png",
            [("A_xss_projector.jpg", "(a) Set A: XSS (n=62)"),
             ("B2_xss_projector.jpg", "(b) Set B-2: XSS (n=62)"),
             ("B1_flooding_projector.jpg", "(c) Set B-1: Flooding (n=202)")],
            ["Web siblings retained", "Entire web family removed", "All flooding types removed from training"],
            "Official Embedding Projector | PCA 2D, Sphereize on | Same 4,851 test IDs | Seed 4266 | Independent projection per model")
    compose("fig_part6_flooding_official_projector.png",
            [("B1_high_volume_projector.jpg", "(a) High-volume flood types (n=78)"),
             ("B1_loic_http_projector.jpg", "(b) LOIC-HTTP (n=110)"),
             ("B1_loic_udp_projector.jpg", "(c) LOIC-UDP (n=14)")],
            ["GoldenEye, SlowHTTPTest and HOIC", "Set B-1", "Set B-1"],
            "Official Embedding Projector | PCA 2D, Sphereize on | Same B-1 projection and 4,851 test samples in all panels")
    settings = {"url": "https://projector.tensorflow.org/", "capture_method": "browser screenshots from the official website",
                "tensor": "64D final classification-head input from new LSTM baseline fits, seed 4266",
                "cohort": "all 851 real test attacks and 4000 sampled real test benign sequences; identical sample IDs across arms",
                "training_rows_in_display": False, "projection": "website approximate PCA, components 1 and 2; 3D off",
                "sphereize_data": True, "color_by": "No color map", "label_by": "label (Attack/Benign)",
                "selection_method": "search in highlight or flood_group metadata; selected points shown red by the website",
                "explained_variance_2d": {"A": .707, "B2": .722, "B1": .700},
                "selection_counts": {"A_XSS": 62, "B2_XSS": 62, "B1_all_flooding": 202,
                                     "B1_high_volume_types": 78, "B1_LOIC_HTTP": 110, "B1_LOIC_UDP": 14},
                "viewport": {"width": 1280, "height": 720},
                "crop": {"x": 370, "y": 165, "width": 590, "height": 545},
                "compositing": "original screenshot pixels pasted without resizing, recoloring, point editing or added point labels; only panel headings and captions added",
                "limitations": ["Sphereizing changes vector lengths and displayed geometry; circular outlines do not establish a detector boundary or attack class.",
                                "Website PCA is fitted on this visualization cohort, including already-inspected test records; it is not the training-fitted PCA from the previous figures.",
                                "Independent per-model projections are not aligned, so cross-panel movement and distances cannot be quantified from these screenshots.",
                                "These are exploratory one-seed visualization figures, not new held-out evidence or the report's three-seed estimates."]}
    (OUT / "website_figure_provenance.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    notes = ["# Official TensorFlow Embedding Projector figures", "",
             "Both PNG figures contain actual screenshots taken in https://projector.tensorflow.org/. No local plot replaces the website rendering. Crops preserve the original pixels; only headings and captions were added. Full website screenshots are retained beside the figures.", "",
             "## Figure choices for Section 6.4", "",
             "- `fig_part6_official_projector.png`: XSS under Set A and B-2, followed by all held-out flooding under B-1.",
             "- `fig_part6_flooding_official_projector.png`: the same B-1 cloud with high-volume flood subtypes, LOIC-HTTP, and LOIC-UDP highlighted separately.", "",
             "## Website settings", "",
             "Load each arm's `vectors.tsv` followed by `metadata.tsv`. Keep Label by = label, Color by = No color map, Sphereize data checked, PCA selected, and Z unchecked. Search Selected by highlight for the summary views. For the B-1 subtype views, search High_volume_flood, LOIC_HTTP or LOIC_UDP by flood_group. The selected counts are 62, 62, 202, 78, 110 and 14 respectively.", "",
             "The circular form is influenced by sphereizing and must not be interpreted as a class or decision boundary. Different models have separately fitted display projections. These test records were already inspected; the website view is exploratory and one-seed only. No training reference rows are included in these website figures.", "",
             "## Suggested compact caption", "",
             "Official Embedding Projector views of LSTM test representations (seed 4266). Red highlights XSS under Set A and B-2, or flooding under B-1; gray/black shows remaining samples. All panels use the same 4,851 test IDs. PCA with sphereizing is fitted separately per model; coordinates are not aligned.", ""]
    (OUT / "README.md").write_text("\n".join(notes), encoding="utf-8")
    print("Composed official website screenshots without modifying the point clouds.")


if __name__ == "__main__":
    main()
