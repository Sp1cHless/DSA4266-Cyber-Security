"""Train fixed-seed baseline LSTMs and export reproducible Part 6 geometry.

Disposable mmap caches live on C:, and only this invocation's cache is removed.
Original datasets, completed runs, and model checkpoints are never deleted.
"""
import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
from sklearn.neighbors import NearestNeighbors

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.common.data import Scheme2, check_disjoint
from train.common.experiment import main as train_main, export_evaluation
from train.common.io import PROJECT, fresh, read_json, sha256, write_json
from train.export_projector import neural_vectors


ARMS = {"A": "holdout_4_subtype", "B1": "holdout_all_flooding", "B2": "holdout_web_family"}
SEED = 4266


def disk():
    return {drive: round(shutil.disk_usage(drive).free / 2**30, 3) for drive in ("C:/", "D:/")}


def safe_cleanup(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if path == root or root not in path.parents or path.drive.upper() != "C:":
        raise ValueError(f"Unsafe disposable cache target: {path}")
    if path.exists():
        shutil.rmtree(path)


def digest_arrays(split):
    result = {}
    for key in ("ids", "y", "attack", "host", "flows"):
        a = split[key]
        h = hashlib.sha256()
        for start in range(0, len(a), 8192):
            h.update(a[start:start + 8192].tobytes())
        result[key] = h.hexdigest()
    return result


def sample(split):
    real = np.char.startswith(split["types"], "real_")
    positive = np.flatnonzero(real & (split["y"] == 1))
    negative = np.flatnonzero(real & (split["y"] == 0))
    chosen = np.random.default_rng(SEED).choice(negative, min(4000, len(negative)), replace=False)
    return np.sort(np.concatenate((positive, chosen)))


def extract(run, split, ids):
    return neural_vectors("lstm", read_json(run / "config.json"), run,
                          np.array(split["x"][ids]), np.array(split["mask"][ids]), "cuda")


def export_arm(arm, dataset, run, cache, folder):
    folder = fresh(folder)
    data = Scheme2(dataset, cache)
    training = data.load("train")
    test = None
    try:
        heldout = data.manifest["train_holdout_attacks"]
        real_attacks = (training["types"] == "real_malicious")
        if np.isin(training["attack"][real_attacks], heldout).any():
            raise ValueError("Withheld primary labels present in real training references")
        tr = sample(training)
        train_vectors, train_scores, details = extract(run, training, tr)
        train_rows = {k: np.array(training[k][tr]) for k in ("ids", "y", "types", "attack", "flows")}
        test = data.load("test")
        check_disjoint(training, test)
        population_hashes = digest_arrays(test)
        threshold = read_json(run / "thresholds.json")["values"]
        # This is the only inference pass over the full test population for this run.
        all_vectors, all_scores, test_details = extract(run, test, np.arange(len(test)))
        evaluation = fresh(folder / "test_evaluation")
        metrics = export_evaluation(evaluation, test, all_scores, threshold)
        te = sample(test)
        vectors = np.concatenate((train_vectors, all_vectors[te])).astype(np.float32)
        rows = []
        for split_name, source, indices, scores in (("train", train_rows, np.arange(len(tr)), train_scores),
                                                   ("test", test, te, all_scores[te])):
            for position, index in enumerate(indices):
                label = int(source["y"][index])
                score = float(scores[position])
                prediction = score >= threshold["validation_fpr_1pct"]
                rows.append({"sample_id": int(source["ids"][index]), "split": split_name,
                             "label": "Attack" if label else "Benign",
                             "primary_attack": str(source["attack"][index]),
                             "sample_type": str(source["types"][index]),
                             "flow_count": int(source["flows"][index]), "score": score,
                             "outcome_1pct": ("TP" if prediction else "FN") if label else ("FP" if prediction else "TN"),
                             "arm": arm, "seed": SEED})
        pca = PCA(n_components=3, svd_solver="full").fit(train_vectors)
        coordinates = pca.transform(vectors).astype(np.float32)
        np.savez_compressed(folder / "representations.npz", vectors=vectors, coordinates=coordinates,
                            pca_components=pca.components_, pca_mean=pca.mean_)
        np.savetxt(folder / "vectors.tsv", vectors, delimiter="\t", fmt="%.9g")
        # Precomputed PCA coordinates keep Projector and report views reproducible.
        np.savetxt(folder / "pca_coordinates.tsv", coordinates, delimiter="\t", fmt="%.9g")
        with (folder / "metadata.tsv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        restored = np.loadtxt(folder / "vectors.tsv", delimiter="\t", dtype=np.float32, ndmin=2)
        if not np.array_equal(restored, vectors):
            raise ValueError("Vector TSV round trip failed")
        # Native-dimensional neighbors support interpretation without using 2D distances.
        nn = NearestNeighbors(n_neighbors=20, algorithm="brute", n_jobs=4).fit(train_vectors)
        positive = np.flatnonzero(test["y"] == 1)
        near = nn.kneighbors(all_vectors[positive], return_distance=False)
        reference_y = train_rows["y"]
        reference_names = train_rows["attack"]
        benign_fraction = (reference_y[near] == 0).mean(axis=1)
        sibling_fraction = np.isin(reference_names[near], ["Brute_Force_-Web", "SQL_Injection"]).mean(axis=1)
        records = []
        for name in sorted(set(test["attack"][positive])):
            keep = test["attack"][positive] == name
            local = positive[keep]
            negative = np.flatnonzero(test["y"] == 0)
            ap_ids = np.concatenate((negative, local))
            records.append({"primary_attack": str(name), "n": int(keep.sum()),
                            "tp": int((all_scores[local] >= threshold["validation_fpr_1pct"]).sum()),
                            "recall": float((all_scores[local] >= threshold["validation_fpr_1pct"]).mean()),
                            "ap_vs_all_test_benign": float(average_precision_score(test["y"][ap_ids], all_scores[ap_ids])),
                            "median_test_flows": float(np.median(test["flows"][local])),
                            "mean_train_benign_fraction_20nn": float(benign_fraction[keep].mean()),
                            "mean_train_web_sibling_fraction_20nn": float(sibling_fraction[keep].mean())})
        flow_metadata = dataset / "flow_metadata_test.npz"
        with np.load(flow_metadata, allow_pickle=False) as flow:
            if not (np.array_equal(flow["sample_id"], test["ids"]) and
                    np.array_equal(flow["total_flows"], test["flows"])):
                raise ValueError("Test flow metadata mismatch")
        manifest = {"complete": True, "arm": arm, "seed": SEED, "dataset": str(dataset), "run": str(run),
                    "dataset_manifest_sha256": data.identity, "model_sha256": sha256(run / "model.pt"),
                    "thresholds_sha256": sha256(run / "thresholds.json"), "thresholds": threshold,
                    "heldout_primary_labels": heldout, "full_test_metrics": metrics,
                    "test_population_hashes": population_hashes,
                    "train_reference_n": len(tr), "test_plot_n": len(te),
                    "train_reference_benign_n": int((train_rows["y"] == 0).sum()),
                    "sampling_seed": SEED, "train_reference": "all real retained attacks and 4000 sampled real benign; no synthetic rows",
                    "pca_fit": "selected real training reference only; unstandardized; no sphereizing",
                    "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                    "representation": details, "test_representation": test_details, "per_subtype": records,
                    "training_data_reused_for_geometry": True,
                    "independent_test_evidence": False,
                    "limitations": ["Single fixed-seed visualization, not the report's three-seed estimates.",
                                    "Validation includes withheld labels for checkpoint and threshold selection.",
                                    "Historical test inspection and current geometry inspection are exploratory.",
                                    "Training reference is attack-enriched; densities do not represent natural prevalence.",
                                    "Each arm refits its representation and PCA; cross-arm coordinates are not aligned.",
                                    "No projected decision boundary is inferred from the two displayed PCs."]}
        write_json(folder / "manifest.json", manifest)
        return manifest
    finally:
        training.close()
        if test is not None:
            test.close()
        gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args()
    out = fresh(args.out)
    cache_root = fresh(args.cache_root)
    write_json(out / "status.json", {"status": "running", "disk_initial_gib": disk()})
    manifests = {}
    for arm, name in ARMS.items():
        dataset = PROJECT / "datasets" / name / "dataset"
        cache = cache_root / arm
        print(f"START {arm}; free GiB {disk()}", flush=True)
        started = time.perf_counter()
        run = train_main("lstm", ["--dataset", str(dataset), "--cache", str(cache), "--device", "cuda",
                                  "--seed", str(SEED), "--out", str(out / f"{arm}_lstm")])
        print(f"EXPORT {arm}; fit complete after {time.perf_counter()-started:.1f}s", flush=True)
        manifests[arm] = export_arm(arm, dataset, run, cache, out / f"{arm}_projector")
        safe_cleanup(cache, cache_root)
        write_json(out / "status.json", {"status": "running", "completed_arms": list(manifests), "disk_free_gib": disk()})
        print(f"CLEANED {arm} disposable cache; free GiB {disk()}", flush=True)
    if not all(m["test_population_hashes"] == manifests["A"]["test_population_hashes"] for m in manifests.values()):
        raise ValueError("Test IDs/labels/hosts/flow counts differ between arms")
    write_json(out / "manifest.json", {"complete": True, "seed": SEED, "arms": manifests,
                                       "matched_test_populations": True, "disk_final_gib": disk(),
                                       "script_sha256": sha256(__file__)})
    write_json(out / "status.json", {"status": "complete", "completed_arms": list(manifests), "disk_free_gib": disk()})
    print(f"DONE {out}", flush=True)


if __name__ == "__main__":
    main()
