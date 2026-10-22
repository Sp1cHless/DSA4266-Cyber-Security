"""Verify and cache only train/validation by default; never fit a model."""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.common.data import Scheme2, training_weights, check_disjoint
from train.common.io import ROOT, DEFAULT_DATASET


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--cache", type=Path, default=ROOT / "cache")
    parser.add_argument("--include-test", action="store_true", help="Also verify test data without fitting or scoring")
    args = parser.parse_args(argv)
    data = Scheme2(args.dataset, args.cache)
    opened = []
    try:
        for name in ("train", "validation", "test") if args.include_test else ("train", "validation"):
            split = data.load(name); opened.append(split)
            print(f"{name}: n={len(split)} groups={split.info['groups']}", flush=True)
        training_weights(opened[0])
        for i in range(len(opened)):
            for j in range(i):
                check_disjoint(opened[i], opened[j])
        print(f"PASS: source shard hashes, shapes, labels, padding, split counts and ownership; test opened={args.include_test}")
    finally:
        for split in opened:
            split.close()


if __name__ == "__main__":
    main()
