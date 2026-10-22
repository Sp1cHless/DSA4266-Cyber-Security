"""Check exported scheme1 data; this validates data construction, not model accuracy."""
from pathlib import Path
import argparse
import json
import logging
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from CIC_new.scheme1.validate import validate
from CIC_new.scheme1.storage import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "CIC_new/outputs/scheme1/dataset")
    parser.add_argument("--prepared", type=Path, help="Override prepared cache location after moving outputs")
    parser.add_argument("--deep-samples", type=int, default=8, help="Independent flow reconstructions per split/sample-type group; zero disables sampling")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    try:
        report = validate(args.data, args.prepared, deep_samples=args.deep_samples)
    except Exception as error:
        if args.data.is_dir():
            write_json(args.data / "validation.json", dict(passed=False, complete=False, error=str(error)))
        parser.exit(1, f"FAIL: {error}\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
