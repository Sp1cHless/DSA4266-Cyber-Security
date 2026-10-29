"""Run the four temporal CNN comparisons with the saved CNN baseline."""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path[0] = str(Path(__file__).resolve().parents[1])

from train.common.temporal_experiment import main as run_main


def main(argv=None):
    return run_main(argv, model="cnn")


if __name__ == "__main__":
    main()
