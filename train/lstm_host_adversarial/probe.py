"""Independent frozen-representation probes on purged training time blocks.

Detector training data are deliberately reused here. Probe evaluation is a
training-derived diagnostic, never held-out detector validation/test evidence.
"""
import hashlib
import sqlite3
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from CIC_new.preprocessing.split_policy import exclude_boundaries
from train.common.io import write_json, sha256
from train.common.torch_runner import tensors, setup
from .model import sample_hosts


def time_partition(training, metadata, seed):
    n = len(training)
    times, first, ends = [np.full(n, -1, np.int64) for _ in range(3)]
    real = np.flatnonzero(np.char.startswith(training['types'], 'real_'))
    ids = training['ids'][real]
    lookup = np.full(int(ids.max()) + 1, -1, np.int64)
    lookup[ids] = real
    matched = 0
    with sqlite3.connect(metadata.resolve().as_uri() + '?mode=ro', uri=True) as db:
        for sid, host, start, earliest, latest in db.execute(
                "SELECT sequence_id,source_ip,window_start_ms,first_ms,max_end_ms "
                "FROM original_sequences WHERE split='train'"):
            if sid >= len(lookup) or lookup[sid] < 0:
                raise ValueError('Metadata does not match the baseline real training IDs')
            i = lookup[sid]
            if training['host'][i] != host or times[i] != -1:
                raise ValueError('Host identity mismatch or duplicate metadata row')
            times[i], first[i], ends[i] = start, earliest, latest
            matched += 1
    if matched != len(real) or np.any(times[real] < 0):
        raise ValueError('Incomplete real-training timestamp mapping')
    blocks, inverse = np.unique(times[real] // 900_000, return_inverse=True)
    rng = np.random.default_rng(seed + 7201)
    shuffled = rng.permutation(len(blocks))
    block_codes = np.zeros(len(blocks), np.uint8)
    block_codes[shuffled[:max(1, int(.2 * len(blocks)))]] = 1
    codes, reasons, _ = exclude_boundaries(times[real], first[real], ends[real],
                                           block_codes[inverse])
    all_codes = np.full(n, 3, np.uint8)
    all_codes[real] = codes
    return all_codes, {'metadata': str(metadata.resolve()), 'metadata_sha256': sha256(metadata),
        'assignment': 'seeded global 15-minute blocks; 80/20; 100s guard and cross-fold flow purge',
        'seed': seed + 7201, 'blocks': len(blocks),
        'fit_real_sequences': int((codes == 0).sum()),
        'evaluation_real_sequences': int((codes == 1).sum()),
        'excluded_real_sequences': int((codes == 3).sum()),
        'exclusion_reason_counts': {str(i): int((reasons == i).sum()) for i in np.unique(reasons)},
        'detector_training_data_reused': True, 'test_used': False}


def probe_tasks(training, codes, seed, fit_cap=64, eval_cap=32):
    """Common sampled tasks for all variants, chosen without model scores."""
    rng = np.random.default_rng(seed + 7301)
    groups = [('benign', training['types'] == 'real_benign'),
              ('attack', training['types'] == 'real_malicious')]
    for name in np.unique(training['attack'][training['types'] == 'real_malicious']):
        groups.append(('attack_type:' + str(name), (training['types'] == 'real_malicious') &
                       (training['attack'] == name)))
    tasks, audit = [], []
    for name, keep in groups:
        fit_all, eval_all = np.flatnonzero(keep & (codes == 0)), np.flatnonzero(keep & (codes == 1))
        fit_hosts, fit_counts = np.unique(training['host'][fit_all], return_counts=True)
        eval_hosts, eval_counts = np.unique(training['host'][eval_all], return_counts=True)
        fit_ok = set(fit_hosts[fit_counts >= 10])
        hosts = sorted(fit_ok.intersection(eval_hosts[eval_counts >= 3]))
        info = {'task': name, 'hosts': len(hosts), 'minimum_fit_per_host': 10,
                'minimum_evaluation_per_host': 3, 'fit_cap_per_host': fit_cap,
                'evaluation_cap_per_host': eval_cap, 'available_fit': len(fit_all),
                'available_evaluation': len(eval_all)}
        if len(hosts) < 2:
            audit.append({**info, 'status': 'not_estimable',
                          'reason': 'Fewer than two hosts supported in both purged folds'})
            continue
        indices, targets = [], []
        for fold, available, cap in ((0, fit_all, fit_cap), (1, eval_all, eval_cap)):
            selected, labels = [], []
            observed = training['host'][available]
            order = np.argsort(observed, kind='stable')
            sorted_hosts = observed[order]
            for target, host in enumerate(hosts):
                lo, hi = np.searchsorted(sorted_hosts, host, side='left'), np.searchsorted(sorted_hosts, host, side='right')
                members = available[order[lo:hi]]
                chosen = rng.choice(members, size=min(len(members), cap), replace=False)
                selected.extend(chosen.tolist())
                labels.extend([target] * len(chosen))
            indices.append(np.array(selected, np.int64))
            targets.append(np.array(labels, np.int64))
        task = {'name': name, 'hosts': hosts, 'fit': indices[0], 'eval': indices[1],
                'fit_y': targets[0], 'eval_y': targets[1]}
        tasks.append(task)
        audit.append({**info, 'status': 'ready', 'fit_n': len(indices[0]), 'evaluation_n': len(indices[1]),
                      'uniform_chance': 1 / len(hosts),
                      'sample_ids_sha256': hashlib.sha256(
                          training['ids'][np.r_[indices[0], indices[1]]].tobytes()).hexdigest()})
    return tasks, audit


@torch.no_grad()
def encode(model, training, indices, device):
    model.eval()
    result = []
    for offset in range(0, len(indices), 2048):
        x, mask = tensors(training, indices[offset:offset + 2048], device)
        result.append(model.encode(x, mask).cpu().numpy())
    return np.concatenate(result)


def evaluate_probes(model, training, tasks, device, threads, seed, out, epochs=25):
    rows = []
    for number, task in enumerate(tasks):
        setup(seed + 7401, threads, device)
        fit_features = encode(model, training, task['fit'], device)
        eval_features = encode(model, training, task['eval'], device)
        mean = fit_features.mean(0)
        scale = fit_features.std(0).clip(.001)
        fit_x = torch.as_tensor((fit_features - mean) / scale, device=device)
        eval_x = torch.as_tensor((eval_features - mean) / scale, device=device)
        fit_y = torch.as_tensor(task['fit_y'], device=device)
        eval_y = task['eval_y']
        k = len(task['hosts'])
        head = nn.Sequential(nn.Linear(fit_x.shape[1], 64), nn.ReLU(), nn.Linear(64, k)).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.0001)
        rng = np.random.default_rng(seed + 7501)
        pools = [np.flatnonzero(task['fit_y'] == h) for h in range(k)]
        losses = []
        for epoch in range(epochs):
            total = 0.
            for _ in range(max(1, int(np.ceil(len(fit_features) / 512)))):
                indices, _ = sample_hosts(pools, 512, rng)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(head(fit_x[indices]), fit_y[indices])
                loss.backward()
                optimizer.step()
                total += loss.item()
            losses.append(total / max(1, int(np.ceil(len(fit_features) / 512))))
        head.eval()
        with torch.no_grad():
            predicted = np.concatenate([head(eval_x[i:i + 2048]).argmax(1).cpu().numpy()
                                        for i in range(0, len(eval_x), 2048)])
        recalls = [float((predicted[eval_y == h] == h).mean()) for h in range(k)]
        balanced = float(np.mean(recalls))
        row = {'task': task['name'], 'hosts': k, 'fit_n': len(fit_features), 'evaluation_n': len(eval_x),
               'balanced_accuracy': balanced, 'accuracy': float((predicted == eval_y).mean()),
               'uniform_chance': 1 / k, 'chance_adjusted_balanced_accuracy': (balanced - 1 / k) / (1 - 1 / k),
               'fixed_epochs': epochs, 'final_train_ce': losses[-1],
               'evaluation_used_for_selection': False, 'detector_training_data_reused': True}
        torch.save({'state_dict': head.state_dict(), 'mean': torch.as_tensor(mean),
                    'scale': torch.as_tensor(scale), 'hosts': task['hosts'], 'epochs': epochs},
                   out / f'probe_{number:02d}.pt')
        np.savez_compressed(out / f'probe_{number:02d}_predictions.npz',
                            sample_id=training['ids'][task['eval']], y=eval_y, predicted=predicted,
                            per_host_recall=recalls)
        rows.append(row)
        write_json(out / 'probe_metrics.json', rows)
        print(f"probe {task['name']} hosts={k} balanced_accuracy={balanced:.5f}", flush=True)
    return rows
