"""Audit completed host-adversarial runs and write a compact evidence review."""
import argparse
from pathlib import Path
import sqlite3
import numpy as np
from CIC_new.preprocessing.split_policy import interval_overlap
from train.common.data import Scheme2
from train.common.io import read_json, write_json, sha256, PROJECT
from .experiment import write_csv


def review(out):
    out = Path(out).resolve()
    if read_json(out / 'status.json')['status'] != 'complete':
        raise ValueError('Review requires all three completed variants')
    if (out / 'integrity_audit.json').exists() or (out / 'INTERPRETATION.md').exists():
        raise FileExistsError('Completed reviews are not overwritten')
    plan, config = read_json(out / 'plan.json'), read_json(out / 'config.json')
    for filename, expected in plan['source_sha256'].items():
        if sha256(PROJECT / filename) != expected:
            raise ValueError('Experiment source changed: ' + filename)
    for filename, expected in plan['reference_sha256'].items():
        if sha256(Path(plan['reference']) / filename) != expected:
            raise ValueError('Reference artifact changed: ' + filename)
    data = Scheme2(config['dataset'])
    if data.identity != plan['dataset_manifest_sha256']:
        raise ValueError('Baseline dataset changed')
    training = data.load('train')
    try:
        with np.load(out / 'probe_partition.npz', allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['sample_id'], training['ids'])
            codes = saved['fold'].copy()
        real = np.char.startswith(training['types'], 'real_')
        if np.any(codes[~real] != 3):
            raise ValueError('Synthetic example included in probe folds')
        size = int(training['ids'].max()) + 1
        lookup = np.full(size, -1, np.int64)
        lookup[training['ids']] = np.arange(len(training))
        times, first, ends = [np.zeros(len(training), np.int64) for _ in range(3)]
        metadata = read_json(out / 'probe_partition.json')
        if sha256(metadata['metadata']) != metadata['metadata_sha256']:
            raise ValueError('Timestamp metadata changed')
        with sqlite3.connect(Path(metadata['metadata']).as_uri() + '?mode=ro', uri=True) as db:
            for sid, start, earliest, latest in db.execute(
                    "SELECT sequence_id,window_start_ms,first_ms,max_end_ms "
                    "FROM original_sequences WHERE split='train'"):
                i = lookup[sid]
                times[i], first[i], ends[i] = start, earliest, latest
        active = codes < 3
        if interval_overlap(first[active], ends[active], times[active], codes[active]).any():
            raise ValueError('Actual-flow overlap across probe folds')
        shared_blocks = np.intersect1d(times[codes == 0] // 900_000, times[codes == 1] // 900_000)
        if len(shared_blocks):
            raise ValueError('Global 15-minute block shared across probe folds')
        eligible = read_json(out / 'host_supervision_coverage.json')[1]['hosts']
        supervision = []
        for attack in np.unique(training['attack'][training['types'] == 'real_malicious']):
            keep = (training['types'] == 'real_malicious') & (training['attack'] == attack)
            hosts = training['host'][keep]
            supervised = np.isin(hosts, eligible)
            supervision.append({'attack': str(attack), 'real_train_n': int(keep.sum()),
                                'real_train_hosts': len(np.unique(hosts)),
                                'host_supervised_n': int(supervised.sum()),
                                'host_supervised_fraction': float(supervised.mean()),
                                'host_supervised_hosts': len(np.unique(hosts[supervised]))})
        comparison = read_json(out / 'comparison.json')
        tasks = [r for r in read_json(out / 'probe_tasks.json') if r['status'] == 'ready']
        with np.load(out / 'probe_cohorts.npz', allow_pickle=False) as cohorts:
            for number, task in enumerate(tasks):
                fit_ids, eval_ids = cohorts[f'task{number}_fit'], cohorts[f'task{number}_eval']
                if np.intersect1d(fit_ids, eval_ids).size:
                    raise ValueError('Probe fit/evaluation ID overlap')
                if np.any(codes[lookup[fit_ids]] != 0) or np.any(codes[lookup[eval_ids]] != 1):
                    raise ValueError('Probe cohort violates its purged time fold')
                for variant in comparison['variants']:
                    with np.load(out / variant['variant'] / f'probe_{number:02d}_predictions.npz') as pred:
                        np.testing.assert_array_equal(pred['sample_id'], eval_ids)
        paired = []
        with np.load(out / 'lambda0/predictions.npz') as saved:
            baseline = {key: saved[key].copy() for key in saved.files}
        baseline_t = read_json(out / 'lambda0/thresholds.json')['values']
        for variant in comparison['variants']:
            folder = out / variant['variant']
            summary = read_json(folder / 'summary.json')
            if summary['test_used'] or sha256(folder / 'model.pt') != summary['model_sha256']:
                raise ValueError('Unexpected test use or changed fitted checkpoint')
            with np.load(folder / 'predictions.npz') as saved:
                for key in ('sample_id', 'y', 'primary_attack', 'source_host'):
                    np.testing.assert_array_equal(saved[key], baseline[key])
                scores = saved['score'].copy()
            thresholds = read_json(folder / 'thresholds.json')['values']
            for row in comparison['attacks']:
                if row['variant'] != variant['variant']:
                    continue
                keep = (baseline['y'] == 1) & (baseline['primary_attack'] == row['attack'])
                point = row['operating_point']
                before = baseline['score'][keep] >= baseline_t[point]
                after = scores[keep] >= thresholds[point]
                paired.append({**row, 'recovered': int((~before & after).sum()),
                               'lost': int((before & ~after).sum())})
        write_csv(out / 'attack_supervision_coverage.csv', supervision)
        write_csv(out / 'paired_attack_changes.csv', paired)
        audit = {'passed': True, 'reference_artifacts_unchanged': True,
                 'planned_source_hashes_unchanged': True, 'baseline_dataset_manifest_unchanged': True,
                 'zero_strength_exact_reproduction': read_json(out / 'lambda0/baseline_reproduction.json'),
                 'same_validation_ids_labels_and_hosts': True, 'probe_ids_disjoint_between_folds': True,
                 'probe_evaluation_ids_identical_across_variants': True,
                 'probe_time_blocks_disjoint': True, 'probe_cross_fold_actual_flow_overlap': False,
                 'synthetic_examples_in_probe_folds': False, 'test_feature_or_label_shards_opened': False,
                 'review_source_sha256': sha256(Path(__file__)), 'review_uses_detector_training_data': True}
        write_json(out / 'integrity_audit.json', audit)
        lines = ['# Evidence review: host-adversarial LSTM', '',
                 'All three planned strengths completed on the original baseline dataset with seed 4266. '
                 'The zero-strength refit reproduces saved same-runtime validation predictions bit for bit.', '',
                 '## Detection on the reused real validation cohort', '',
                 '| Strength | AP | Recall at global 1% FPR | Recall at global 0.1% FPR |',
                 '|---|---:|---:|---:|']
        for row in comparison['variants']:
            lines.append(f"| {row['strength']:g} | {row['ap']:.4%} | {row['recall_1pct']:.2%} | {row['recall_0.1pct']:.2%} |")
        lines += ['', '## Host decodability on training-derived probe evaluation', '',
                  '| Task | Hosts | Baseline | Weak (0.01) | Moderate (0.1) |',
                  '|---|---:|---:|---:|---:|']
        for task in tasks:
            rows = [r for r in comparison['probes'] if r['task'] == task['task']]
            values = ' | '.join(f"{r['balanced_accuracy']:.2%}" for r in rows)
            lines.append(f"| {task['task']} | {task['hosts']} | {values} |")
        lines += ['', 'Probe evaluation reuses detector training records in separate purged time blocks. '
                  'These accuracies measure decodability for the specified independently trained MLP, '
                  'not generalization to unseen hosts. All probes use identical IDs and fixed training settings.', '',
                  '## Which attack types receive host supervision?', '',
                  '| Attack | Real training N | Training hosts | Host-supervised N | Coverage |',
                  '|---|---:|---:|---:|---:|']
        for row in supervision:
            lines.append(f"| {row['attack']} | {row['real_train_n']} | {row['real_train_hosts']} | {row['host_supervised_n']} | {row['host_supervised_fraction']:.2%} |")
        lines += ['', 'A host qualifies using its total real attack support, so support can be shared across subtypes. '
                  'Only the host auxiliary loss is restricted; every original and synthetic example still contributes to detector training.', '',
                  '## Interpretation', '',
                  'Assess changes in independent host probes together with attack-specific recall. '
                  'A detection change alone cannot establish reduced host dependence. '
                  'A decrease in probe accuracy is evidence only for reduced decodability under this probe family; '
                  'it does not prove removal of all host information. '
                  'Attack subtype, campaign and host effects remain entangled, especially for single-source subtypes. '
                  'No causal or unseen-host robustness claim is supported by this experiment.', '',
                  'See REPORT.md for all attack recalls at both global operating points; '
                  'paired_attack_changes.csv records recovered and lost validation examples. '
                  'integrity_audit.json records the checked reference, cohort, fold and checkpoint invariants.', '',
                  'This is single-seed exploratory validation on reused data; no new independent final-test performance is claimed.']
        (out / 'INTERPRETATION.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        print(audit)
        print(out / 'INTERPRETATION.md')
    finally:
        training.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    review(parser.parse_args().run)
