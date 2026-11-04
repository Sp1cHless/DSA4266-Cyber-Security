"""Build Scheme2 train/validation/test inputs from the source database.

Fit a whole-bucket input encoder, freeze it, and export zero-shift vector mixtures.
This workflow does not train a downstream prediction model.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from CIC_new.preprocessing.run_preprocessing import prepare
from CIC_new.preprocessing.export_augmented_dataset import export_dataset
from CIC_new.preprocessing.augmentation import Encoded
from CIC_new.preprocessing.dataset import fingerprint
from CIC_new.preprocessing.donor_plan import make_plan, original_catalog
from CIC_new.preprocessing.sequence_index import signature, write_json, read_json


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def completed(folder, marker, resume):
    folder = Path(folder)
    if not folder.exists() or not any(folder.iterdir()):
        return False
    if not resume:
        raise FileExistsError(f"Fresh stage directory required: {folder}")
    path = folder/marker
    if not path.is_file() or not read_json(path).get("complete"):
        raise ValueError(f"Interrupted stage cannot be resumed: {folder}; retain it and use a new --out")
    print(f"[reuse completed stage] {folder}", flush=True)
    return True


def validate_export(dataset, encoded, donor_plan=None, holdout=()):
    """Check every output against frozen originals and the shared zero-shift plan."""
    dataset, encoded = map(Path, (dataset, encoded))
    manifest = read_json(dataset/"manifest.json")
    if donor_plan is not None and digest(dataset/"donor_plan.npz") != digest(donor_plan):
        raise ValueError("Export did not preserve the frozen donor plan")
    with np.load(dataset/"donor_plan.npz", allow_pickle=False) as archive:
        plan = {k: archive[k] for k in archive.files}
    planned_seen = np.zeros(len(plan["sample_id"]), bool)
    shards, checked = [], 0
    for split in ("train", "validation", "test"):
        store = Encoded(encoded, split)
        positives = int(store.meta["label"].sum())
        expected_counts = dict(real_benign=len(store.meta)-positives, real_malicious=positives)
        if split == "train":
            expected_counts.update(synthetic_benign=positives, synthetic_malicious=positives)
        if manifest["counts"][split] != expected_counts or manifest["total_samples"][split] != sum(expected_counts.values()):
            raise ValueError("Dataset budget differs from frozen original populations")
        originals_seen = np.zeros(len(store.meta), bool)
        counts = Counter()
        for path in sorted((dataset/split).glob("part_*.npz")):
            with np.load(path, allow_pickle=False) as archive:
                a = {k: archive[k] for k in archive.files}
            n = len(a["sample_id"])
            dimension = store.manifest["bucket_dim"]
            if a["X"].shape != (n, 10, dimension+1) or a["X"].dtype != np.float32 or not np.isfinite(a["X"]).all():
                raise ValueError("Invalid exported feature tensor")
            if np.any(a["right_shift_buckets"]):
                raise ValueError("Scheme2 requires zero displacement")
            synthetic = a["right_id"] >= 0
            if split != "train" and synthetic.any():
                raise ValueError("Synthetic held-out example")
            left = np.searchsorted(store.meta["id"], a["left_id"])
            right = np.searchsorted(store.meta["id"], np.where(synthetic, a["right_id"], a["left_id"]))
            if (left >= len(store.meta)).any() or (right >= len(store.meta)).any():
                raise ValueError("Donor outside its split")
            if not np.array_equal(store.meta["id"][left], a["left_id"]) or not np.array_equal(
                    store.meta["id"][right], np.where(synthetic, a["right_id"], a["left_id"])):
                raise ValueError("Unknown donor")
            real = ~synthetic
            if not np.array_equal(a["sample_id"][real], a["left_id"][real]) or np.any(a["right_id"][real] != -1):
                raise ValueError("Invalid original provenance")
            if originals_seen[left[real]].any() or len(np.unique(left[real])) != real.sum():
                raise ValueError("Duplicate original")
            originals_seen[left[real]] = True
            if synthetic.any():
                positions = a["sample_id"][synthetic]-int(plan["sample_id"][0])
                if (positions < 0).any() or (positions >= len(planned_seen)).any():
                    raise ValueError("Synthetic ID outside donor plan")
                if planned_seen[positions].any() or len(np.unique(positions)) != len(positions):
                    raise ValueError("Duplicate synthetic example")
                for key in ("sample_id", "left_id", "right_id", "sample_type", "right_shift_buckets"):
                    if not np.array_equal(a[key][synthetic], plan[key][positions]):
                        raise ValueError(f"Donor plan mismatch: {key}")
                if np.any(store.meta["label"][left[synthetic]]) or np.any(store.meta["ip"][left[synthetic]] == store.meta["ip"][right[synthetic]]):
                    raise ValueError("Invalid background donor or shared source")
                planned_seen[positions] = True
            present = synthetic[:, None]
            ma = np.asarray(store.arrays["mask"][left])
            mb = np.asarray(store.arrays["mask"][right])*present
            expected_mask = np.maximum(ma, mb)
            z = np.asarray(store.arrays["embeddings"][left])*ma[..., None]
            z = z+np.asarray(store.arrays["embeddings"][right])*mb[..., None]
            z /= np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-8)
            z *= expected_mask[..., None]
            flows = np.asarray(store.arrays["flow_counts"][left])+np.asarray(store.arrays["flow_counts"][right])*present
            labels = np.maximum(store.arrays["y_bucket"][left], store.arrays["y_bucket"][right]*present)
            volume = ((np.log1p(flows.astype(np.float32))-store.scaler["mean"][0])/store.scaler["scale"][0])*expected_mask
            if not np.allclose(a["X"][..., :dimension], z, rtol=2e-5, atol=2e-6) or not np.allclose(
                    a["X"][..., -1], volume, rtol=2e-5, atol=2e-6):
                raise ValueError("Exported vector addition/L2 or flow-intensity channel is incorrect")
            for actual, expected in ((a["mask"], expected_mask), (a["flow_counts"], flows),
                    (a["y_bucket"], labels), (a["y_sequence"], labels.max(axis=1))):
                if not np.array_equal(actual, expected):
                    raise ValueError("Labels, masks or counts do not match donors")
            if np.any(a["X"][a["mask"] == 0] != 0):
                raise ValueError("Nonzero padding")
            expected_types = np.where(a["y_sequence"] > 0, "malicious", "benign")
            expected_types = np.char.add(np.where(synthetic, "synthetic_", "real_"), expected_types)
            if not np.array_equal(a["sample_type"], expected_types):
                raise ValueError("Wrong class/sample type")
            if not np.array_equal(a["left_ip"], store.meta["ip"][left]) or not np.array_equal(
                    a["right_ip"], np.where(synthetic, store.meta["ip"][right], "")):
                raise ValueError("Source ownership mismatch")
            counts.update(map(str, a["sample_type"]))
            shards.append(dict(path=path.relative_to(dataset).as_posix(), count=n, sha256=digest(path)))
            checked += n
        if not originals_seen.all() or dict(counts) != manifest["counts"][split]:
            raise ValueError("Original coverage or split counts are incorrect")
        print(f"[validate {split}] all {sum(counts.values()):,} vectors/counts/labels/owners checked", flush=True)
    if not planned_seen.all():
        raise ValueError("Missing synthetic plan rows")
    manifest.update(representation="scheme2", target_ratios=[.6, .2, .2],
        donor_plan_sha256=digest(dataset/"donor_plan.npz"),
        shards=shards, schema_sha256=digest(dataset/"schema.json"), scaler_sha256=digest(dataset/"scaler.json"),
        train_holdout_attacks=list(holdout))
    write_json(dataset/"manifest.json", manifest)
    result = dict(passed=True, complete=True, feature_width=manifest["feature_width"], counts=manifest["counts"],
        target_ratios=[.6, .2, .2], P=manifest["P"], all_exported_samples_checked=checked,
        all_shards_checked=len(shards), shift_buckets=0, model_accuracy_evaluated=False,
        scope="all exported vectors checked against frozen donors; masks/labels/counts/ownership and original/plan coverage checked; not independent raw-flow reaggregation")
    write_json(dataset/"validation.json", result)
    return result


def build(args):
    import torch
    from CIC_new.representation_learning.learn_bucket_embeddings import train_bucket
    from CIC_new.representation_learning.encode_sequences import encode

    source_before = signature((args.db or args.csv).resolve())
    if args.prepared and args.holdout_train_attacks:
        raise ValueError("A training holdout must be applied while splitting; it cannot be added to an existing preparation")
    if args.out is None:
        parent = ROOT/"CIC_new/outputs/scheme2_runs"
        parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        args.out = Path(tempfile.mkdtemp(prefix=f"{stamp}_scheme2_bucket{args.bucket_dim}_shift0_seed{args.seed}_", dir=parent))
    out = args.out.resolve()
    settings = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "resume"}
    if out.exists() and any(out.iterdir()):
        if not args.resume or not (out/"workflow.json").is_file():
            raise FileExistsError(f"Fresh output required, or --resume for this completed-stage workflow: {out}")
        workflow = read_json(out/"workflow.json")
        if workflow["settings"] != settings:
            raise ValueError("Resume settings differ from the recorded workflow")
    else:
        out.mkdir(parents=True, exist_ok=True)
        workflow = dict(complete=False, status="running", started_at=datetime.now().astimezone().isoformat(),
            settings=settings, prediction_model_training_started=False,
            train_holdout_attacks=list(args.holdout_train_attacks), stages={})
    workflow.update(complete=False, status="running")
    write_json(out/"workflow.json", workflow)
    print(f"Output: {out}", flush=True)
    data = args.prepared.resolve() if args.prepared else out/"preprocessed"
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    base = dict(data=data, device=device, seed=args.seed, cpu_threads=args.cpu_threads, batch_size=args.batch_size)
    try:
        print("[stage 1/5] Real sequences, buckets, 6:2:2 split and train-only transforms", flush=True)
        if args.prepared:
            if not read_json(data/"manifest.json").get("complete") or read_json(data/"manifest.json").get("version") != 3:
                raise ValueError("--prepared must be completed V3 Scheme2 preparation")
        elif not completed(data, "manifest.json", args.resume):
            prepare(data, db=args.db, csv=args.csv, seed=args.seed, shard_size=args.shard_size,
                    holdout=args.holdout_train_attacks)
        if read_json(data/"manifest.json")["source"] != source_before:
            raise ValueError("Prepared data belongs to another source database")
        if list(read_json(data/"manifest.json").get("train_holdout_attacks", [])) != list(args.holdout_train_attacks):
            raise ValueError("Prepared data uses a different training holdout")
        # Preserve the established attack/host/activity sampling and seed offset.
        plan = out/"donor_plan.npz"
        expected_plan = make_plan(data, original_catalog(data), args.seed+1)
        if plan.exists():
            with np.load(plan, allow_pickle=False) as saved:
                if set(saved.files) != set(expected_plan) or any(not np.array_equal(saved[k], v) for k, v in expected_plan.items()):
                    raise ValueError("Frozen donor plan differs from deterministic sampling")
        else:
            np.savez_compressed(plan, **expected_plan)
        workflow["stages"]["preprocessed"] = digest(data/"manifest.json")
        write_json(out/"workflow.json", workflow)
        folder = out/"bucket_embeddings"
        print("[stage 2/5] Fit reusable whole-bucket input embeddings (not the final detector)", flush=True)
        checkpoint = args.bucket_checkpoint.resolve() if args.bucket_checkpoint else folder/"best.pt"
        if not args.bucket_checkpoint and not completed(folder, "summary.json", args.resume):
            train_bucket(SimpleNamespace(**base, out=folder, steps=args.embedding_steps,
                validate_every=args.validate_every, category_dim=args.category_dim, bucket_dim=args.bucket_dim,
                lr=.001, val_limit=args.val_limit))
        checkpoint_hash = digest(checkpoint)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["data_fingerprint"] != fingerprint(data):
            raise ValueError("Bucket checkpoint belongs to another preparation or changed transforms")
        if saved["model_args"]["category_dim"] != args.category_dim or saved["model_args"]["bucket_dim"] != args.bucket_dim:
            raise ValueError("Checkpoint dimensions differ from the requested workflow")
        if args.resume and "bucket_checkpoint_sha256" in workflow["stages"] and workflow["stages"]["bucket_checkpoint_sha256"] != checkpoint_hash:
            raise ValueError("Completed checkpoint changed since the previous stage")
        workflow["stages"]["bucket_checkpoint_sha256"] = checkpoint_hash
        write_json(out/"workflow.json", workflow)
        encoded = out/"encoded_sequences"
        print("[stage 3/5] Frozen encoding of real train/validation/test; no fitting on test", flush=True)
        if not completed(encoded, "manifest.json", args.resume):
            encode(SimpleNamespace(**base, out=encoded, checkpoint=checkpoint, splits=["train", "validation", "test"]))
        if read_json(encoded/"manifest.json")["checkpoint_sha256"] != checkpoint_hash:
            raise ValueError("Encoded data belongs to another checkpoint")
        if read_json(encoded/"manifest.json")["data_fingerprint"] != fingerprint(data):
            raise ValueError("Encoded data belongs to changed preprocessing transforms")
        if read_json(encoded/"manifest.json")["bucket_dim"] != args.bucket_dim:
            raise ValueError("Bucket dimension differs from this workflow")
        dataset = out/"dataset"
        print("[stage 4/5] Shared donor pairs, zero displacement, vector addition and L2", flush=True)
        if not completed(dataset, "manifest.json", args.resume):
            export_dataset(data, encoded, dataset, seed=args.seed, shard_size=args.shard_size,
                           shift_mode="none", donor_plan=plan)
        print("[stage 5/5] Check every exported example against frozen donors", flush=True)
        report = validate_export(dataset, encoded, donor_plan=plan, holdout=args.holdout_train_attacks)
        from CIC_new.preprocessing.flow_metadata import export_flow_metadata
        export_flow_metadata(data, dataset)
        if signature((args.db or args.csv).resolve()) != source_before:
            raise ValueError("Raw source changed during preparation")
    except BaseException as error:
        workflow.update(complete=False, status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        write_json(out/"workflow.json", workflow)
        if (out/"dataset").exists():
            write_json(out/"dataset/validation.json", dict(passed=False, complete=False, error=str(error)))
        raise
    workflow.update(complete=True, status="complete", finished_at=datetime.now().astimezone().isoformat(),
                    dataset=str(dataset), validation=report)
    write_json(out/"workflow.json", workflow)
    print(f"COMPLETE: final dataset = {dataset}", flush=True)
    return workflow


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--db", type=Path, help="Full-schema SQLite source; default datasets/cic_2018_traffic.db")
    source.add_argument("--csv", type=Path, help="Full-schema CSV source imported into the fresh workflow directory")
    p.add_argument("--out", type=Path, help="Fresh workflow directory; default reserves a unique scheme2_runs directory")
    p.add_argument("--holdout-train-attacks", default="",
                   help="Comma-separated primary attack labels removed from the training fold only; validation/test retained")
    p.add_argument("--prepared", type=Path, help="Optional existing completed V3 preparation")
    p.add_argument("--bucket-checkpoint", type=Path, help="Optional existing whole-bucket checkpoint for that exact V3 preparation")
    p.add_argument("--resume", action="store_true", help="Reuse completed stages only; requires the same explicit --out and settings")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--seed", type=int, default=4266)
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--shard-size", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--embedding-steps", type=int, default=5000)
    p.add_argument("--validate-every", type=int, default=200)
    p.add_argument("--val-limit", type=int, default=4096)
    p.add_argument("--category-dim", type=int, default=16)
    p.add_argument("--bucket-dim", type=int, default=64)
    args = p.parse_args()
    if args.db is None and args.csv is None:
        args.db = ROOT/"datasets/cic_2018_traffic.db"
    args.holdout_train_attacks = sorted({name.strip() for name in args.holdout_train_attacks.split(",") if name.strip()})
    if args.seed < 0 or any(getattr(args, k) <= 0 for k in ("cpu_threads", "shard_size", "batch_size", "embedding_steps", "validate_every", "val_limit", "category_dim", "bucket_dim")):
        p.error("Seed must be nonnegative and sizes/training budgets positive")
    if args.resume and args.out is None:
        p.error("--resume requires the previous --out")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    print(f"Python: {sys.executable}", flush=True)
    try:
        build(args)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        p.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
