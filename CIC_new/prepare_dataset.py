"""Build scheme1 data from the full CIC SQLite database, with zero-shift synthesis."""
from pathlib import Path
import argparse
import logging
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from CIC_new.scheme1.pipeline import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=ROOT / "datasets/cic_2018_traffic.db")
    parser.add_argument("--out", type=Path, default=ROOT / "CIC_new/outputs/scheme1")
    parser.add_argument("--resume", action="store_true", help="Reuse completed stages only; no partial-stage or optimizer resume")
    parser.add_argument("--prepare-only", action="store_true", help="Stop after real buckets, split, vocabularies and fitting pools")
    parser.add_argument("--embeddings", type=Path, help="Directory from train_embeddings.py, fitted on this preparation")
    parser.add_argument("--seed", type=int, default=4266)
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--train-pool", type=int, default=50000, help="Uniform bucket reservoir size; all-category witnesses are added")
    parser.add_argument("--val-pool", type=int, default=2048)
    parser.add_argument("--embedding-steps", type=int, default=3000, help="Minimum updates; extended if needed to cover every training category")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--validate-every", type=int, default=200)
    parser.add_argument("--l7-dim", type=int, default=8)
    parser.add_argument("--src-port-dim", type=int, default=8)
    parser.add_argument("--dst-port-dim", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--skip-validation", action="store_true", help="Export only; run validate_dataset.py separately")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    print(f"Python: {sys.executable}", flush=True)
    try:
        build(**vars(args))
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, ImportError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
