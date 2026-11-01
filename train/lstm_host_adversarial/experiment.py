"""Matched LSTM host-adversarial experiment; no detector test shards are read."""
import argparse
import copy
import csv
import math
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score
from train.common.data import Scheme2, check_disjoint, training_weights
from train.common.experiment import export_evaluation
from train.common.io import ROOT, PROJECT, environment, fresh, read_json, sha256, unique_name, write_json
from train.common.metrics import POLICY, choose_thresholds
from train.common.torch_runner import setup, tensors, predict
from .model import Model, host_pools, sample_hosts
from .probe import time_partition, probe_tasks, evaluate_probes

REFERENCE = ROOT / 'runs/20261024_223102_671768_circular_multiseed_start5_ef34ba/seed4266/lstm/start0'
METADATA = PROJECT / 'CIC_new/outputs/scheme2_runs/20261022_115847_160650_scheme2_bucket64_shift0_seed4266_0fhh3gjt/preprocessed/manifest.sqlite'
REFERENCE_FILES = ('config.json', 'summary.json', 'model.pt', 'thresholds.json', 'predictions.npz', 'data_audit.json')
LIMITS = [
    'Single-seed exploratory intervention on reused baseline training data and previously inspected real validation.',
    'Validation AP selects checkpoints and validation labels select global FPR thresholds; no independent final test claim.',
    'The frozen Scheme2 encoder, scaler, donor plan, inputs and detector objective are unchanged; the encoder is not retrained.',
    'Hosts and campaigns may recur across baseline splits. Reduced host decodability does not establish unseen-host generalization or causality.',
    'Auxiliary host supervision uses real training hosts with at least 20 sequences; unsupported hosts are excluded only from host supervision.',
    'Synthetic examples still contribute to detector BCE, but are excluded from single-host adversarial supervision.',
    'Binary-class-conditional adversaries reduce benign/attack prevalence confounding, but cannot separate attack subtype, campaign and host effects.',
    'Single-source attack subtypes do not identify host-independent attack behavior. Within-subtype probes require multiple supported hosts.',
    'Independent probes reuse detector training examples in purged time blocks; their evaluation is diagnostic, not detector validation/test.',
    'Probe tasks, sampling, epochs and hyperparameters are fixed before model scoring; probe evaluation does not select checkpoints or strengths.',
    'A weaker independent probe is evidence of lower decodability for that probe family, not proof that all host information is absent.',
]


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def attack_rows(split, scores, thresholds, variant):
    rows = []
    for attack in np.unique(split['attack'][split['y'] == 1]):
        keep = (split['y'] == 1) & (split['attack'] == attack)
        hosts = split['host'][keep]
        for point in ('validation_fpr_1pct', 'validation_fpr_0.1pct'):
            detected = scores[keep] >= thresholds[point]
            per_host = [float(detected[hosts == h].mean()) for h in np.unique(hosts)]
            rows.append({'variant': variant, 'attack': str(attack), 'operating_point': point,
                         'n': int(keep.sum()), 'hosts': len(per_host), 'tp': int(detected.sum()),
                         'fn': int((~detected).sum()), 'recall': float(detected.mean()),
                         'macro_host_recall': float(np.mean(per_host)), 'threshold': thresholds[point]})
    return rows


def fit(training, validation, weights, config, pools, strength, out):
    seed, device = config['seed'], config['device']
    setup(seed, config['threads'], device)
    model = Model(config['architecture'], [len(p) for p in pools], seed).to(device)
    settings = config['training']
    detector_parameters = list(model.detector.parameters())
    host_parameters = list(model.adversaries.parameters())
    optimized = detector_parameters + (host_parameters if strength else [])
    optimizer = torch.optim.AdamW(optimized, lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    rng, auxiliary_rng = np.random.default_rng(seed), np.random.default_rng(seed + 7601)
    history, best, waiting, best_epoch = [], -1., 0, 0
    started = time.perf_counter()
    for epoch in range(1, settings['epochs'] + 1):
        model.train()
        detector_total, host_total, host_steps = 0., 0., 0
        host_correct, host_seen = [0, 0], [0, 0]
        order = rng.permutation(len(training))
        effective = strength * min(1., epoch / 3.)
        for start in range(0, len(order), settings['batch_size']):
            ids = order[start:start + settings['batch_size']]
            x, mask = tensors(training, ids, device)
            y = torch.as_tensor(np.array(training['y'][ids], dtype=np.float32), device=device)
            w = torch.as_tensor(weights[ids], device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, mask)
            detector_loss = (F.binary_cross_entropy_with_logits(logits, y, reduction='none') * w).mean()
            loss = detector_loss
            if strength:
                auxiliary_ids, auxiliary_targets = zip(*[
                    sample_hosts(pool, config['host_supervision']['batch_per_class'], auxiliary_rng)
                    for pool in pools])
                ax, am = tensors(training, np.concatenate(auxiliary_ids), device)
                representation = model.encode(ax, am)
                host_losses = []
                offset = 0
                for label, target in enumerate(auxiliary_targets):
                    target = torch.as_tensor(target, device=device)
                    host_logits = model.host_logits(representation[offset:offset + len(target)], label, effective)
                    # Normalize class counts so the two conditional tasks have comparable CE scales.
                    host_losses.append(F.cross_entropy(host_logits, target) / math.log(len(pools[label])))
                    host_correct[label] += int((host_logits.detach().argmax(1) == target).sum().item())
                    host_seen[label] += len(target)
                    offset += len(target)
                auxiliary_loss = torch.stack(host_losses).mean()
                # Reversal scales only representation gradients; host heads always learn to classify.
                loss = loss + auxiliary_loss
                host_total += auxiliary_loss.item()
                host_steps += 1
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite objective')
            loss.backward()
            # Preserve the baseline detector clipping budget; extra heads have a separate cap.
            torch.nn.utils.clip_grad_norm_(detector_parameters, max_norm=1.)
            if strength:
                torch.nn.utils.clip_grad_norm_(host_parameters, max_norm=1.)
            optimizer.step()
            detector_total += detector_loss.item() * len(ids)
        scores = predict(model, validation, device)
        ap = float(average_precision_score(validation['y'], scores))
        history.append({'epoch': epoch, 'weighted_train_bce': detector_total / len(training),
                        'normalized_host_ce': host_total / host_steps if host_steps else None,
                        'online_host_accuracy': [c / n if n else None for c, n in zip(host_correct, host_seen)],
                        'effective_reversal_strength': effective, 'validation_ap': ap,
                        'elapsed_seconds': time.perf_counter() - started})
        print(f"lambda={strength:g} epoch={epoch} BCE={detector_total / len(training):.6f} AP={ap:.6f} "
              f"host_CE={history[-1]['normalized_host_ce']} seconds={time.perf_counter()-started:.1f}", flush=True)
        if ap > best:
            best, waiting, best_epoch = ap, 0, epoch
            torch.save({'model_name': 'lstm_host_adversarial', 'architecture': config['architecture'],
                        'host_sizes': [len(p) for p in pools], 'strength': strength,
                        'state_dict': model.state_dict(), 'best_epoch': epoch}, out / 'model.pt')
        else:
            waiting += 1
        write_json(out / 'history.json', history)
        if waiting >= settings['patience']:
            break
    saved = torch.load(out / 'model.pt', weights_only=True, map_location=device)
    model.load_state_dict(saved['state_dict'])
    return model, {'best_epoch': best_epoch, 'best_validation_ap': best, 'epochs_completed': len(history),
                   'fit_seconds': time.perf_counter() - started,
                   'detector_parameters': sum(p.numel() for p in detector_parameters),
                   'host_parameters': sum(p.numel() for p in host_parameters)}


def compare(out, variants, all_attacks, all_probes):
    baseline = variants[0]
    attack_base = {(r['attack'], r['operating_point']): r for r in all_attacks if r['variant'] == 'lambda0'}
    for row in all_attacks:
        previous = attack_base[row['attack'], row['operating_point']]
        row['delta_recall_pp'] = 100 * (row['recall'] - previous['recall'])
        row['delta_macro_host_recall_pp'] = 100 * (row['macro_host_recall'] - previous['macro_host_recall'])
    probe_base = {r['task']: r for r in all_probes if r['variant'] == 'lambda0'}
    for row in all_probes:
        row['delta_balanced_accuracy_pp'] = 100 * (row['balanced_accuracy'] - probe_base[row['task']]['balanced_accuracy'])
    write_csv(out / 'attack_comparison.csv', all_attacks)
    write_csv(out / 'probe_comparison.csv', all_probes)
    write_csv(out / 'overall_comparison.csv', variants)
    write_json(out / 'comparison.json', {'variants': variants, 'attacks': all_attacks, 'probes': all_probes,
                                       'limitations': LIMITS, 'test_used': False})
    lines = ['# LSTM host-adversarial experiment', '',
             'Single-seed exploratory real-validation comparison on frozen baseline Scheme2 inputs.',
             'The lambda=0 detector was refitted with the same initialization and detection minibatch order.', '',
             '| Variant | AP | ROC-AUC | Recall at 1% FPR | Recall at 0.1% FPR | Best epoch |',
             '|---|---:|---:|---:|---:|---:|']
    for r in variants:
        lines.append(f"| {r['variant']} | {r['ap']:.4%} | {r['roc_auc']:.6f} | {r['recall_1pct']:.2%} | {r['recall_0.1pct']:.2%} | {r['best_epoch']} |")
    lines += ['', '## Attack-specific real-validation results', '',
              'All attack rows use the same global threshold within each variant; no attack-specific calibration.', '',
              '| Variant | Attack | FPR budget | N | Hosts | Recall | Host-macro recall | Recall delta (pp) |',
              '|---|---|---|---:|---:|---:|---:|---:|']
    for r in all_attacks:
        lines.append(f"| {r['variant']} | {r['attack']} | {r['operating_point']} | {r['n']} | {r['hosts']} | {r['recall']:.2%} | {r['macro_host_recall']:.2%} | {r['delta_recall_pp']:+.2f} |")
    lines += ['', '## Independent host probes: training-derived diagnostics', '',
              'Fixed 25-epoch MLPs trained from scratch on frozen sequence representations. '
              'Fit-only standardization and uniform-host sampling; identical purged time folds and IDs across variants. '
              'Evaluation was never used for probe epoch selection. Detector training records are reused.', '',
              '| Variant | Task | Hosts | Fit / evaluation N | Balanced accuracy | Chance | Delta (pp) |',
              '|---|---|---:|---:|---:|---:|---:|']
    for r in all_probes:
        lines.append(f"| {r['variant']} | {r['task']} | {r['hosts']} | {r['fit_n']} / {r['evaluation_n']} | {r['balanced_accuracy']:.2%} | {r['uniform_chance']:.2%} | {r['delta_balanced_accuracy_pp']:+.2f} |")
    lines += ['', '## Interpretation limits', ''] + ['- ' + item for item in LIMITS]
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run', type=Path, default=REFERENCE)
    parser.add_argument('--metadata', type=Path, default=METADATA)
    parser.add_argument('--cache', type=Path, default=ROOT / 'cache')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    reference = args.reference_run.resolve()
    config, summary = [read_json(reference / name) for name in ('config.json', 'summary.json')]
    if (config['model'] != 'lstm' or not summary['complete'] or summary['test_used'] or
            summary['split'] != 'validation' or config['evaluation_policy'] != POLICY or
            config['input_transform'] != {'name': 'identity'} or
            config['weighting'] != 'equal_four_groups_mean_weight_one' or
            config['architecture'].get('circular_shift_start', 0) != 0 or
            read_json(reference / 'data_audit.json')['test_opened']):
        raise ValueError('Expected completed chronological baseline LSTM on real validation')
    reference_hashes = {name: sha256(reference / name) for name in REFERENCE_FILES}
    if reference_hashes['model.pt'] != summary['model_sha256']:
        raise ValueError('Reference checkpoint changed')
    if reference_hashes['thresholds.json'] != summary['thresholds_sha256']:
        raise ValueError('Reference thresholds changed')
    current_environment = environment()
    for key in ('python', 'packages', 'machine'):
        if config['environment'][key] != current_environment[key]:
            raise ValueError('Use the baseline runtime: ' + key)
    config = copy.deepcopy(config)
    config['environment'] = current_environment
    config['host_supervision'] = {'minimum_real_sequences': 20, 'batch_per_class': 128,
        'reduction': 'mean_of_two_conditional_cross_entropies_divided_by_log_host_count',
        'reversal_strengths': [0., .01, .1], 'linear_ramp_epochs': 3,
        'sampling': 'uniform host then uniform sequence with replacement; separate RNG from detector',
        'head': '192->64 ReLU -> host count; no dropout', 'clip': 'detector and auxiliary heads separately at norm 1'}
    dataset = Scheme2(config['dataset'], args.cache)
    if dataset.identity != config['dataset_manifest_sha256'] or dataset.timesteps != 10:
        raise ValueError('Baseline Scheme2 dataset mismatch')
    if not args.metadata.is_file():
        raise FileNotFoundError(args.metadata)
    if args.dry_run:
        print({'reference': str(reference), 'dataset': str(dataset.path), 'seed': config['seed'],
               'device': config['device'], 'training': config['training'],
               'host_supervision': config['host_supervision'], 'test_used': False}, flush=True)
        return
    out = fresh(args.out or ROOT / 'runs' / unique_name('lstm_host_adversarial_seed' + str(config['seed'])))
    print('Experiment directory: ' + str(out), flush=True)
    write_json(out / 'config.json', config)
    plan = {'reference': str(reference), 'reference_sha256': reference_hashes,
            'dataset_manifest_sha256': dataset.identity, 'strengths': [0., .01, .1],
            'training_data_reused': True, 'encoder_retrained': False, 'test_used': False,
            'source_sha256': {str(p.relative_to(PROJECT)): sha256(p) for p in
                list((ROOT / 'lstm_host_adversarial').glob('*.py')) +
                [ROOT / 'run_lstm_host_adversarial.py', ROOT / 'lstm/model.py', ROOT / 'common/torch_runner.py',
                 ROOT / 'common/data.py', ROOT / 'common/metrics.py', ROOT / 'common/networks.py',
                 PROJECT / 'CIC_new/preprocessing/split_policy.py']},
            'limitations': LIMITS}
    write_json(out / 'plan.json', plan)
    write_json(out / 'status.json', {'status': 'preparing', 'test_used': False})
    opened = []
    try:
        training, validation = dataset.load('train'), dataset.load('validation')
        opened.extend((training, validation))
        check_disjoint(training, validation)
        for split, key in ((training, 'train_cohort_sha256'), (validation, 'validation_cohort_sha256')):
            if split.info['cohort_sha256'] != summary[key]:
                raise ValueError('Baseline cohort changed: ' + key)
        weights = training_weights(training)
        pools, coverage = host_pools(training)
        write_json(out / 'host_supervision_coverage.json', coverage)
        codes, partition_audit = time_partition(training, args.metadata, config['seed'])
        tasks, task_audit = probe_tasks(training, codes, config['seed'])
        write_json(out / 'probe_partition.json', partition_audit)
        write_json(out / 'probe_tasks.json', task_audit)
        np.savez_compressed(out / 'probe_partition.npz', sample_id=training['ids'], fold=codes)
        np.savez_compressed(out / 'probe_cohorts.npz', **{
            f'task{i}_{key}': training['ids'][task[key]] for i, task in enumerate(tasks) for key in ('fit', 'eval')})
        write_json(out / 'data_audit.json', {'train': training.info, 'validation': validation.info,
                   'dataset_manifest_sha256': dataset.identity, 'test_opened': False,
                   'host_supervision_coverage': [{k: v for k, v in c.items() if k not in ('hosts', 'counts')} for c in coverage]})
        print('Host coverage: ' + str([{k: v for k, v in c.items() if k not in ('hosts', 'counts')} for c in coverage]), flush=True)
        print('Probe tasks: ' + str(task_audit), flush=True)
        variants, all_attacks, all_probes = [], [], []
        for strength in config['host_supervision']['reversal_strengths']:
            variant = f'lambda{strength:g}'
            target = fresh(out / variant)
            write_json(target / 'config.json', {**config, 'reversal_strength': strength})
            write_json(out / 'status.json', {'status': 'training', 'variant': variant, 'test_used': False})
            model, fit_info = fit(training, validation, weights, config, pools, strength, target)
            scores = predict(model, validation, config['device'])
            thresholds = choose_thresholds(validation['y'], scores)
            write_json(target / 'thresholds.json', {'source': 'real_validation_only', 'values': thresholds})
            metrics = export_evaluation(target, validation, scores, thresholds)
            identity = {}
            if strength == 0:
                with np.load(reference / 'predictions.npz', allow_pickle=False) as saved:
                    for key, array in (('sample_id', validation['ids']), ('y', validation['y']),
                                       ('primary_attack', validation['attack']), ('source_host', validation['host'])):
                        np.testing.assert_array_equal(saved[key], array)
                    identity = {'scores_bitwise_equal_to_reference': bool(np.array_equal(scores, saved['score'])),
                                'max_absolute_score_difference': float(np.max(np.abs(scores - saved['score'])))}
                write_json(target / 'baseline_reproduction.json', identity)
                # A refit that fails to reproduce is retained for diagnosis, not silently accepted.
                if not identity['scores_bitwise_equal_to_reference']:
                    raise ValueError('Zero-strength refit failed exact same-runtime baseline reproduction')
            write_json(out / 'status.json', {'status': 'independent_probes', 'variant': variant, 'test_used': False})
            probe_results = evaluate_probes(model, training, tasks, config['device'], config['threads'], config['seed'], target)
            operating = metrics['operating_points']
            variants.append({'variant': variant, 'strength': strength, 'ap': metrics['average_precision'],
                             'roc_auc': metrics['roc_auc'], 'recall_1pct': operating['validation_fpr_1pct']['recall'],
                             'recall_0.1pct': operating['validation_fpr_0.1pct']['recall'],
                             'fpr_1pct': operating['validation_fpr_1pct']['fpr'],
                             'fpr_0.1pct': operating['validation_fpr_0.1pct']['fpr'],
                             'best_epoch': fit_info['best_epoch']})
            all_attacks.extend(attack_rows(validation, scores, thresholds, variant))
            all_probes.extend({'variant': variant, **row} for row in probe_results)
            write_json(target / 'summary.json', {'complete': True, 'split': 'validation', 'fit': fit_info,
                       'metrics': metrics, 'probes': probe_results, 'baseline_reproduction': identity,
                       'model_sha256': sha256(target / 'model.pt'), 'test_used': False, 'limitations': LIMITS})
            compare(out, variants, all_attacks, all_probes)
            del model
            torch.cuda.empty_cache()
        if reference_hashes != {name: sha256(reference / name) for name in REFERENCE_FILES}:
            raise ValueError('Reference artifacts changed')
        if dataset.identity != sha256(dataset.path / 'manifest.json'):
            raise ValueError('Dataset manifest changed')
        write_json(out / 'status.json', {'status': 'complete', 'test_used': False, 'completed_variants': len(variants)})
        print('Completed experiment: ' + str(out), flush=True)
        return out
    except BaseException as error:
        write_json(out / 'status.json', {'status': 'failed', 'error': repr(error),
                   'traceback': traceback.format_exc(), 'test_used': False})
        raise
    finally:
        for split in opened:
            split.close()
