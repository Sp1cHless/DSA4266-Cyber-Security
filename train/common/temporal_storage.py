"""Bound temporal runs to one case's disposable caches without changing inputs."""
import os
from pathlib import Path
import re
import shutil
import sqlite3
import stat
import subprocess
from contextlib import closing

from .io import read_json, write_json, sha256, unique_name

GIB = 1024 ** 3
RESERVE = 2 * GIB
STORAGE_FILES = {'train/common/temporal_experiment.py', 'train/common/temporal_storage.py',
                 'train/common/remove_temporal_cache.ps1'}
LEGACY_DRIVER = '895e93f730265ea2c6f8d83a0ed8034633ec0e861e805604910b64586d63b88b'


def normalized_pipeline(settings):
    result = {}
    for key, value in settings['pipeline_sha256'].items():
        key = key.replace('\\', '/')
        if key in result and result[key] != value:
            raise ValueError('Conflicting recorded source hashes: ' + key)
        result[key] = value
    return result


def compatible_storage_upgrade(previous, current):
    """Permit only the reviewed old orchestration code to adopt cache management."""
    if {k: v for k, v in previous.items() if k != 'pipeline_sha256'} != {
            k: v for k, v in current.items() if k != 'pipeline_sha256'}:
        return False
    old, new = normalized_pipeline(previous), normalized_pipeline(current)
    if (old.get('train/common/temporal_experiment.py') != LEGACY_DRIVER and
            old.get('train/common/temporal_experiment.py') != new.get('train/common/temporal_experiment.py')):
        return False
    return ({k: v for k, v in old.items() if k not in STORAGE_FILES} ==
            {k: v for k, v in new.items() if k not in STORAGE_FILES})


def tree_files(path):
    """Reject links before measuring or removing a cache."""
    files = []
    for folder, directories, names in os.walk(path, followlinks=False):
        for name in directories + names:
            entry = Path(folder) / name
            info = entry.lstat()
            if entry.is_symlink() or getattr(info, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                raise ValueError('Refusing a linked cache entry: ' + str(entry))
        files.extend(Path(folder) / name for name in names)
    return files


def remove_cache(target, owner, audit, kind):
    """Delete one explicit cache, retaining all embedded weights and JSON metadata."""
    target, owner = Path(target).absolute(), Path(owner).resolve()
    if not target.is_relative_to(owner) or target == owner:
        raise ValueError('Cache target escaped its owner: ' + str(target))
    relative = target.relative_to(owner)
    allowed = (kind == 'prepared' and relative.as_posix() in
               {'mmap/train', 'mmap/validation', 'mmap/train.partial', 'mmap/validation.partial'})
    allowed |= kind == 'encoded' and relative.as_posix() == 'encoded_sequences'
    allowed |= kind == 'detector' and re.fullmatch(r'(cnn|lstm)_attempt_\d{4}_cache', relative.as_posix()) is not None
    if not allowed:
        raise ValueError('Unrecognized disposable cache: ' + str(target))
    for ancestor in [target, *target.parents]:
        if ancestor.is_symlink() or (ancestor.exists() and
                getattr(ancestor.lstat(), 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
            raise ValueError('Refusing a linked cache path: ' + str(ancestor))
        if ancestor == owner:
            break
    if not target.exists():
        return 0
    files = tree_files(target)
    size = sum(file.stat().st_size for file in files)
    record = {'target': str(target), 'owner': str(owner), 'kind': kind,
              'bytes': size, 'files': len(files), 'status': 'planned', 'preserved': []}
    audit = Path(audit)
    audit.mkdir(parents=True, exist_ok=True)
    record_path = audit / (unique_name('cache') + '.json')
    for file in files:
        if file.suffix.lower() not in {'.json', '.pt', '.pth', '.ckpt', '.ubj', '.onnx', '.safetensors'}:
            continue
        destination = record_path.with_suffix('') / file.relative_to(target)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, destination)
        digest = sha256(file)
        if sha256(destination) != digest:
            raise ValueError('Cache provenance copy differs: ' + str(file))
        record['preserved'].append({'source': str(file), 'copy': str(destination), 'sha256': digest})
    write_json(record_path, record)
    if os.name == 'nt':
        shell = shutil.which('pwsh') or shutil.which('powershell')
        if not shell:
            raise RuntimeError('PowerShell is required for native Windows cache removal')
        subprocess.run([shell, '-NoProfile', '-NonInteractive', '-File',
                        str(Path(__file__).with_name('remove_temporal_cache.ps1')),
                        '-Root', str(owner), '-Target', str(target)], check=True, capture_output=True, text=True)
    else:
        shutil.rmtree(target)
    if target.exists():
        raise RuntimeError('Cache directory remains after cleanup: ' + str(target))
    record['status'] = 'removed'
    write_json(record_path, record)
    print(f'[storage] removed {size / GIB:.3f} GiB: {target}', flush=True)
    return size


def cleanup_case(case, item, model, output, representation_verified=False, detector_verified=False):
    prepared = Path(case['prepared']['path'])
    audit = Path(output) / 'cache_cleanup' / case['name']
    removed = 0
    names = ('train', 'validation', 'train.partial', 'validation.partial') if representation_verified else (
        'train.partial', 'validation.partial')
    for name in names:
        removed += remove_cache(prepared / 'mmap' / name, prepared, audit, 'prepared')
    if representation_verified:
        representation = Path(item['representation']['path'])
        removed += remove_cache(representation / 'encoded_sequences', representation, audit, 'encoded')
    for attempt in item.get('attempts', []):
        if attempt['stage'] != model:
            continue
        folder = Path(attempt['path'])
        if detector_verified or not attempt.get('complete'):
            removed += remove_cache(folder.with_name(folder.name + '_cache'), folder.parent, audit, 'detector')
    return removed


def population(case):
    prepared = Path(case['prepared']['path'])
    with closing(sqlite3.connect((prepared / 'manifest.sqlite').resolve().as_uri() + '?mode=ro', uri=True)) as connection:
        rows = connection.execute("SELECT split,COUNT(*),SUM(attack_flows>0) FROM original_sequences "
                                  "WHERE split IN ('train','validation') GROUP BY split").fetchall()
    return {split: (int(n), int(positive)) for split, n, positive in rows}


def estimate_stage(case, stage, dataset=None):
    """Conservative new-allocation estimate, including a 2 GiB reserve."""
    if stage == 'representation':
        prepared = Path(case['prepared']['path'])
        schema = read_json(prepared / 'schema.json')
        counts = population(case)
        times = schema['timesteps']
        cache = 0
        total = sum(n for n, _ in counts.values())
        for split, (n, _) in counts.items():
            if not (prepared / 'mmap' / split / 'identity.json').is_file():
                cache += n * times * (4 * (84 + sum(schema['category_slots'].values())) + 8) + 8192
        encoded = total * (times * (64 * 4 + 4 + 8 + 8) + 400)
        exported_n = total + 2 * counts['train'][1]
        # Uncompressed output bound plus ample per-row provenance allowance.
        exported = exported_n * (times * (65 * 4 + 1 + 1 + 8) + 1400)
        return {'prepared_cache': cache, 'encoded_arrays': encoded,
                'export_bound': exported, 'reserve': RESERVE}
    manifest = read_json(Path(dataset) / 'manifest.json')
    total = sum(manifest['total_samples'][split] for split in ('train', 'validation'))
    times = manifest['timesteps']
    # The loader stores x, mask, y, ids, types, attack, host, flows.
    cache = total * (times * (65 * 4 + 4) + 8 + 8 + 96 + 320 + 256 + 8)
    # Saved scores, curves, checkpoints, logs and model working files.
    return {'detector_cache': cache, 'result_bound': total * 800, 'reserve': RESERVE}


def require_space(case, stage, output, dataset=None):
    components = estimate_stage(case, stage, dataset)
    required = sum(components.values())
    owner = Path(case['prepared']['path']) if stage == 'representation' else Path(output)
    # Each allocation drive must independently fit. The supported normal run uses D only.
    for folder in {owner.resolve(), Path(output).resolve()}:
        while not folder.exists():
            folder = folder.parent
        free = shutil.disk_usage(folder).free
        print(f'[storage] {case["name"]}/{stage}: need <= {required / GIB:.2f} GiB '
              f'including reserve; free={free / GIB:.2f} GiB at {folder}', flush=True)
        if free < required:
            raise RuntimeError(f'Insufficient disk space before {case["name"]}/{stage}: '
                               f'{required / GIB:.2f} GiB required, {free / GIB:.2f} GiB free. '
                               'No new stage was started; completed results are retained.')
    return components
