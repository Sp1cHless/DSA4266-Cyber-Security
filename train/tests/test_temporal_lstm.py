"""Four-case LSTM coverage using explicit short artificial reference runs."""
from train.tests import test_temporal_cnn as shared
from train.run_temporal_lstm import main as run_lstm_batch


class TemporalLSTMTests(shared.TemporalCNNTests):
    MODEL = "lstm"
    run_batch = staticmethod(run_lstm_batch)
