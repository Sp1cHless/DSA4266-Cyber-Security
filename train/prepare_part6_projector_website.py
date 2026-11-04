"""Prepare matched test-only native vectors for the official Projector website."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent / "runs/20261103_part6_new_holdouts_lstm_seed4266_v1"
OUT = ROOT / "official_projector_website_v1"
FLOODS = {"DoS_attacks-GoldenEye", "DoS_attacks-SlowHTTPTest", "DDOS_attack-HOIC",
          "DDoS_attacks-LOIC-HTTP", "DDOS_attack-LOIC-UDP"}


def main():
    OUT.mkdir(exist_ok=False)
    matched = None
    manifest = {"source": str(ROOT), "representation": "native 64D final-head input; no retraining",
                "split": "test", "seed": 4266, "training_reference_rows_included": False,
                "purpose": "official TensorFlow Embedding Projector browser screenshots",
                "limitations": ["Exploratory reuse of already-inspected test records.",
                                "One fixed-seed model per arm; report tables remain three-seed summaries.",
                                "Website PCA and optional sphereizing are display-only transforms fitted on this visualization cohort.",
                                "Each arm has its own model and projection; coordinates across arms are not aligned."]}
    for arm in ("A", "B2", "B1"):
        source = ROOT / f"{arm}_projector"
        with (source / "metadata.tsv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        ids = [i for i, row in enumerate(rows) if row["split"] == "test"]
        selected = [rows[i] for i in ids]
        identity = [(r["sample_id"], r["label"], r["primary_attack"], r["flow_count"]) for r in selected]
        if matched is None:
            matched = identity
        elif matched != identity:
            raise ValueError("Website cohorts differ across arms")
        with np.load(source / "representations.npz", allow_pickle=False) as arrays:
            vectors = arrays["vectors"][ids]
        folder = OUT / arm
        folder.mkdir()
        np.savetxt(folder / "vectors.tsv", vectors, delimiter="\t", fmt="%.9g")
        metadata = []
        for row in selected:
            attack = row["primary_attack"]
            focus = attack in FLOODS if arm == "B1" else attack == "Brute_Force_-XSS"
            group = ("LOIC_HTTP" if attack == "DDoS_attacks-LOIC-HTTP" else
                     "LOIC_UDP" if attack == "DDOS_attack-LOIC-UDP" else
                     "High_volume_flood" if attack in FLOODS else "Other")
            metadata.append({"label": row["label"], "primary_attack": attack,
                             "highlight": "Selected" if focus else "Other", "flood_group": group,
                             "outcome_1pct": row["outcome_1pct"], "flow_count": row["flow_count"],
                             "sample_id": row["sample_id"]})
        with (folder / "metadata.tsv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(metadata[0]), delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(metadata)
        manifest[arm] = {"rows": len(metadata), "dimensions": vectors.shape[1],
                         "selected_count": sum(r["highlight"] == "Selected" for r in metadata)}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(OUT)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
