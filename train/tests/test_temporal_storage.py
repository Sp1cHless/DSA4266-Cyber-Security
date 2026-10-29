"""Storage bounds, strict resume migration, and cache-only removal checks."""
from copy import deepcopy
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from train.common.io import write_json, sha256, read_json
from train.common.temporal_storage import (LEGACY_DRIVER, GIB, RESERVE,
    compatible_storage_upgrade, estimate_stage, remove_cache, require_space)


class TemporalStorageTests(unittest.TestCase):
    def test_reviewed_storage_migration_rejects_scientific_or_runtime_changes(self):
        previous = {'seed': 4266, 'packages': {'torch': 'same'}, 'pipeline_sha256': {
            'train\\common\\temporal_experiment.py': LEGACY_DRIVER,
            'train/common/temporal_experiment.py': LEGACY_DRIVER,
            'train/common/experiment.py': 'scientific'}}
        current = deepcopy(previous)
        current['pipeline_sha256'].update({'train\\common\\temporal_experiment.py': 'new-driver',
            'train/common/temporal_experiment.py': 'new-driver',
            'train/common/temporal_storage.py': 'new-storage',
            'train/common/remove_temporal_cache.ps1': 'native-cleanup'})
        self.assertTrue(compatible_storage_upgrade(previous, current))
        for mutation in ('seed', 'scientific', 'runtime', 'unknown_driver'):
            altered = deepcopy(current)
            original = deepcopy(previous)
            if mutation == 'seed':
                altered['seed'] = 1
            elif mutation == 'scientific':
                altered['pipeline_sha256']['train/common/experiment.py'] = 'changed'
            elif mutation == 'runtime':
                altered['packages']['torch'] = 'different'
            else:
                original['pipeline_sha256']['train\\common\\temporal_experiment.py'] = 'unknown'
                original['pipeline_sha256']['train/common/temporal_experiment.py'] = 'unknown'
            self.assertFalse(compatible_storage_upgrade(original, altered))

    def test_cache_removal_preserves_weights_and_metadata_and_rejects_other_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            owner = root / 'representation'
            target = owner / 'encoded_sequences'
            (target / 'train').mkdir(parents=True)
            (target / 'train/embeddings.npy').write_bytes(b'fixture-cache')
            (target / 'frozen_bucket_encoder.pt').write_bytes(b'fixture-weight')
            write_json(target / 'manifest.json', {'complete': True})
            (owner / 'dataset').mkdir()
            protected = owner / 'dataset/part.npz'
            protected.write_bytes(b'final-data')
            before = sha256(protected)
            audit = root / 'audit'
            removed = remove_cache(target, owner, audit, 'encoded')
            self.assertGreater(removed, 0)
            self.assertFalse(target.exists())
            records = [read_json(p) for p in audit.glob('*.json')]
            self.assertEqual(records[0]['status'], 'removed')
            self.assertEqual(len(records[0]['preserved']), 2)
            for saved in records[0]['preserved']:
                self.assertEqual(sha256(Path(saved['copy'])), saved['sha256'])
            self.assertEqual(sha256(protected), before)
            self.assertEqual(remove_cache(target, owner, audit, 'encoded'), 0)
            for path in (owner, owner / 'dataset', root / 'elsewhere'):
                with self.assertRaises(ValueError):
                    remove_cache(path, owner, audit, 'encoded')

    def test_space_estimates_and_low_disk_stop_before_allocation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = root / 'prepared'
            prepared.mkdir()
            write_json(prepared / 'schema.json', {'timesteps': 40,
                       'category_slots': {'protocol': 7, 'l7': 134, 'service': 9}})
            with closing(sqlite3.connect(prepared / 'manifest.sqlite')) as connection:
                connection.execute('CREATE TABLE original_sequences(split TEXT,attack_flows INTEGER)')
                connection.executemany('INSERT INTO original_sequences VALUES(?,?)',
                                       [('train', 0), ('train', 1), ('validation', 1), ('test', 1)])
                connection.commit()
            case = {'name': 'fixture', 'prepared': {'path': str(prepared)}}
            components = estimate_stage(case, 'representation')
            self.assertEqual(components['prepared_cache'], 3 * 40 * (4 * (84 + 150) + 8) + 2 * 8192)
            self.assertEqual(components['reserve'], RESERVE)
            with patch('train.common.temporal_storage.shutil.disk_usage', return_value=SimpleNamespace(free=0)):
                with self.assertRaisesRegex(RuntimeError, 'No new stage was started'):
                    require_space(case, 'representation', root / 'future-run')
            self.assertFalse((root / 'future-run').exists())
            with patch('train.common.temporal_storage.shutil.disk_usage', return_value=SimpleNamespace(free=100 * GIB)):
                self.assertEqual(require_space(case, 'representation', root), components)
            dataset = root / 'dataset'
            dataset.mkdir()
            write_json(dataset / 'manifest.json', {'timesteps': 40, 'total_samples': {'train': 4, 'validation': 1}})
            detector = estimate_stage(case, 'lstm', dataset)
            self.assertEqual(detector['detector_cache'], 5 * (40 * 264 + 696))


if __name__ == '__main__':
    unittest.main()
