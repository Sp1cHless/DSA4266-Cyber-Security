"""Train a mixed-class temporal AE on a completed scheme1 or scheme2 dataset."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from CIC_new.modelling.train_autoencoder import EarlyStopping, train_temporal
from CIC_new.training_utils import validate_arguments


def new_run_directory(args):
    """Atomically reserve an empty, identifiable directory even for simultaneous runs."""
    parent = ROOT / "CIC_new/outputs/ae_runs"
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    population = "real" if args.real_only else "aug"
    representation = getattr(args, "representation", "scheme1")
    if getattr(args, "input_transform", "none") == "bucket_l2" and representation == "scheme1":
        representation = "scheme1_l2"
    if representation not in ("scheme1", "scheme2", "scheme1_l2"):
        representation = "custom"
    prefix = f"{stamp}_{representation}_ae_{population}_e{args.epochs}_b{args.batch_size}_seed{args.seed}_"
    return Path(tempfile.mkdtemp(prefix=prefix, dir=parent))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, help="Default: latest completed Scheme2 dataset; pass a rebuilt Scheme1 dataset explicitly for historical reproduction")
    parser.add_argument("--input-transform", choices=("none", "bucket_l2"), default="none")
    parser.add_argument("--out", type=Path, help="Exact fresh output directory; default reserves a unique directory under CIC_new/outputs/ae_runs")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=4266)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=500, help="Maximum epochs; early stopping may finish sooner")
    parser.add_argument("--steps-per-epoch", type=int, default=500)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--latent", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--patience", type=int, default=20, help="Epochs without significant validation improvement; 0 disables")
    parser.add_argument("--min-delta", type=float, default=.001, help="Relative improvement fraction, 0.001 = 0.1 percent")
    parser.add_argument("--real-only", action="store_true", help="Balanced real benign/malicious control without synthetic samples")
    parser.add_argument("--val-limit", type=int, default=0, help="0 uses ALL real validation sequences; positive values select a fixed uniform subset")
    args = parser.parse_args()
    if args.val_limit < 0:
        parser.error("--val-limit must be nonnegative")
    if args.batch_size < 4 or args.batch_size % 4:
        parser.error("--batch-size must be a positive multiple of four")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    try:
        if args.dataset is None:
            from CIC_new.run_scheme2_experiment import latest_dataset
            args.dataset = latest_dataset()
        manifest = json.loads((args.dataset / "manifest.json").read_text())
        args.representation = manifest.get("representation", "custom")
        if args.val_limit == 0:
            args.val_limit = manifest["total_samples"]["validation"]
        validate_arguments(parser, args)
        EarlyStopping(args.patience, args.min_delta)
        if args.device == "auto":
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.out is None:
            args.out = new_run_directory(args)
        print(f"Python: {sys.executable}", flush=True)
        train_temporal(args)
    except (ValueError, KeyError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
