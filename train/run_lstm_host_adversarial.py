"""Run the matched baseline-data LSTM host-adversarial experiment."""
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.lstm_host_adversarial.experiment import main

if __name__ == '__main__':
    main()
