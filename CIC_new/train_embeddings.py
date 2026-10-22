"""Independently fit L7/source-port/destination-port embeddings on real TRAIN."""
from pathlib import Path
import argparse
import logging
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "CIC_new/outputs/scheme1/prepared")
    parser.add_argument("--out", type=Path, default=ROOT / "CIC_new/outputs/scheme1/embeddings")
    parser.add_argument("--steps", type=int, default=3000, help="Minimum updates; at least one complete pass is required")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--validate-every", type=int, default=200)
    parser.add_argument("--l7-dim", type=int, default=8)
    parser.add_argument("--src-port-dim", type=int, default=8)
    parser.add_argument("--dst-port-dim", type=int, default=8)
    parser.add_argument("--seed", type=int, default=4266)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--lr", type=float, default=.001)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    try:
        from CIC_new.scheme1.embeddings import train
        settings = vars(args).copy()
        prepared = settings.pop("data")
        train(prepared, **settings)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, ImportError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
