"""Run from the repository root or by absolute script path."""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train.common.experiment import main

if __name__ == "__main__":
    main("xgboost")
