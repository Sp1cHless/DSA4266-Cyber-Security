"""Focused checks of reversal, unchanged control, sampling and probe isolation."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from torch.nn import functional as F
from train.lstm.model import Model as Baseline
from train.common.torch_runner import setup, fit as baseline_fit, predict
from train.lstm_host_adversarial.model import Model, Reverse, host_pools, sample_hosts
from train.lstm_host_adversarial.experiment import fit, attack_rows
from train.lstm_host_adversarial.probe import probe_tasks, evaluate_probes


class Split(dict):
    def __len__(self):
        return len(self['y'])


def fixture():
    rng = np.random.default_rng(5)
    types = np.repeat(['real_benign', 'real_malicious', 'synthetic_benign', 'synthetic_malicious'], 16)
    y = np.char.endswith(types, 'malicious').astype(np.int64)
    return Split(x=rng.normal(size=(64, 10, 65)).astype(np.float32), mask=np.ones((64, 10), np.float32),
                 y=y, ids=np.arange(1, 65), host=np.tile(np.repeat(['hostA', 'hostB'], 8), 4),
                 types=types, attack=np.where(y == 1, 'Attack', ''))


class HostAdversarialTest(unittest.TestCase):
    def test_reversal_and_head_gradient(self):
        x = torch.tensor([[1., 2.]], requires_grad=True)
        head = torch.nn.Linear(2, 2)
        target = torch.tensor([0])
        ordinary = F.cross_entropy(head(x), target)
        gx, gh = torch.autograd.grad(ordinary, (x, head.weight))
        reverse = F.cross_entropy(head(Reverse.apply(x, .1)), target)
        rx, rh = torch.autograd.grad(reverse, (x, head.weight))
        torch.testing.assert_close(rx, -.1 * gx)
        torch.testing.assert_close(rh, gh)
        zero = Reverse.apply(x, 0.).sum()
        torch.testing.assert_close(torch.autograd.grad(zero, x)[0], torch.zeros_like(x))

    def test_initialization_and_rng(self):
        setup(4266, 1, 'cuda' if torch.cuda.is_available() else 'cpu')
        baseline = Baseline(hidden_size=8, dropout=.2)
        cpu_rng = torch.get_rng_state().clone()
        gpu_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        setup(4266, 1, 'cuda' if torch.cuda.is_available() else 'cpu')
        model = Model({'hidden_size': 8, 'dropout': .2}, [2, 2])
        self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
        for previous, current in zip(gpu_rng, torch.cuda.get_rng_state_all() if gpu_rng else []):
            self.assertTrue(torch.equal(previous, current))
        for key, value in baseline.state_dict().items():
            torch.testing.assert_close(value, model.detector.state_dict()[key], atol=0, rtol=0)
        split = fixture()
        model.eval()
        baseline.eval()
        x, mask = torch.from_numpy(split['x']), torch.from_numpy(split['mask'])
        torch.testing.assert_close(baseline(x, mask), model(x, mask), atol=0, rtol=0)
        torch.testing.assert_close(model.detector.head(model.encode(x, mask)).squeeze(-1), model(x, mask), atol=0, rtol=0)

    def test_real_only_host_sampling(self):
        split = fixture()
        pools, audit = host_pools(split, minimum=5)
        for label, pool in enumerate(pools):
            ids, targets = sample_hosts(pool, 1000, np.random.default_rng(5))
            self.assertTrue(np.all(split['types'][ids] == ('real_benign', 'real_malicious')[label]))
            self.assertTrue(np.all(split['host'][ids] == np.array(audit[label]['hosts'])[targets]))
        split['host'][split['types'] == 'synthetic_malicious'] = 'syntheticOnly'
        _, audit = host_pools(split, minimum=5)
        self.assertNotIn('syntheticOnly', audit[1]['hosts'])

    def test_three_strengths_control_and_independent_probe(self):
        split = fixture()
        config = {'architecture': {'hidden_size': 8, 'dropout': .2},
                  'training': {'epochs': 2, 'patience': 2, 'batch_size': 16,
                               'learning_rate': .001, 'weight_decay': .0001},
                  'device': 'cpu', 'seed': 4266, 'threads': 1,
                  'host_supervision': {'batch_per_class': 8}}
        pools, _ = host_pools(split, minimum=5)
        weights = np.ones(len(split), np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original_dir = root / 'original'; original_dir.mkdir()
            original, _ = baseline_fit('lstm', split, split, weights, config, 'cpu', 1, 4266, original_dir)
            for strength in (0., .01, .1):
                target = root / str(strength); target.mkdir()
                model, info = fit(split, split, weights, config, pools, strength, target)
                scores = predict(model, split, 'cpu')
                self.assertTrue(np.isfinite(scores).all())
                self.assertEqual(info['epochs_completed'], 2)
                if strength == 0:
                    np.testing.assert_array_equal(predict(original, split, 'cpu'), scores)
                    for key, value in original.state_dict().items():
                        torch.testing.assert_close(value, model.detector.state_dict()[key], atol=0, rtol=0)
                rows = attack_rows(split, scores, {'validation_fpr_1pct': .5, 'validation_fpr_0.1pct': .6}, str(strength))
                self.assertEqual(len(rows), 2)
                task = {'name': 'small', 'hosts': ['hostA', 'hostB'],
                        'fit': np.r_[0:5, 8:13], 'eval': np.r_[5:8, 13:16],
                        'fit_y': np.repeat([0, 1], 5), 'eval_y': np.repeat([0, 1], 3)}
                before = {key: value.clone() for key, value in model.state_dict().items()}
                rows = evaluate_probes(model, split, [task], 'cpu', 1, 4266, target, epochs=2)
                self.assertEqual(rows[0]['uniform_chance'], .5)
                for key, value in before.items():
                    torch.testing.assert_close(value, model.state_dict()[key], atol=0, rtol=0)

    def test_probe_support_and_fold_separation(self):
        split = fixture()
        codes = np.tile(np.array([0] * 5 + [1] * 3), 8)
        tasks, audit = probe_tasks(split, codes, 4266)
        self.assertEqual(tasks, [])  # At least ten fit records per host are required.
        self.assertTrue(all(row['status'] == 'not_estimable' for row in audit))


if __name__ == '__main__':
    unittest.main()
