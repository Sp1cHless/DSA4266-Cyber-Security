"""Exact dilution, train-only reads and both representation routes on fixtures."""
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np
import torch

from CIC_new.diagnose_controlled_dilution import (
    RATIOS, ROUTES, EventReader, Representations, controlled_subsets,
    host_balanced, build_cases, score_cases, summarize, baseline_transitions, report_markdown,
)
from CIC_new.evaluate_autoencoder import write, digest
from CIC_new.modelling.autoencoder import TemporalAE
from CIC_new.preprocessing import bucket_statistics as bf
from CIC_new.preprocessing.sequence_index import signature
from CIC_new.representation_learning.bucket_autoencoder import BucketAE
from CIC_new.scheme1.statistics import FIELDS, LEARNED, raw_bucket
from CIC_new.tests.fixtures import event


def events(n, label, row_start):
    return np.stack([event(start=(i%10)*10000+1000+i, label=label, raw_rowid=row_start+i,
        l4_src_port=100+i%3, protocol=6 if label else 17, l7=7 if label else 5) for i in range(n)])


class ControlledDilutionTests(unittest.TestCase):
    def test_exact_nested_subsets_no_duplicates_or_mutation(self):
        a, b, c = events(3, 1, 1), events(300, 0, 100), events(8, 0, 1000)
        before = b.copy()
        parts = list(controlled_subsets(a, b, c, np.random.default_rng(4)))
        self.assertEqual([p for p, _, _ in parts], list(RATIOS))
        previous = None
        for percent, bg, control in parts:
            self.assertEqual(len(a)*100, percent*(len(a)+len(bg)))
            self.assertEqual(len(control), len(a))
            self.assertEqual(len(np.unique(bg[:, bf.E["raw_rowid"]])), len(bg))
            if previous is not None:
                np.testing.assert_array_equal(bg, previous[:len(bg)])
            previous = bg
        np.testing.assert_array_equal(b, before)
        with self.assertRaises(ValueError):
            list(controlled_subsets(a, b[:2], c, np.random.default_rng(1)))
        rows = [(1, "a", 2), (2, "a", 2), (3, "b", 2)]
        self.assertEqual(len({r[1] for r in host_balanced(rows, 2, np.random.default_rng(2))}), 2)

    def test_readonly_event_membership_and_exact_attack_type(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root/"source.sqlite"
            conn = sqlite3.connect(raw)
            conn.execute("CREATE TABLE network_flows(attack TEXT)")
            conn.executemany("INSERT INTO network_flows VALUES(?)", [("Alpha",), ("Beta",), ("Benign",)])
            conn.commit(); conn.close()
            work = sqlite3.connect(root/"work.sqlite")
            work.execute("CREATE TABLE events(sequence_id INTEGER,"+",".join(n+" INTEGER" for n in bf.EVENT_COLUMNS)+")")
            work.execute("CREATE INDEX events_sequence ON events(sequence_id)")
            for sid, row in ((1, event(label=1, raw_rowid=1)), (1, event(label=1, raw_rowid=2)), (2, event(raw_rowid=3))):
                work.execute("INSERT INTO events VALUES("+",".join("?" for _ in range(1+len(row)))+")", (sid, *map(int, row)))
            work.commit(); work.close()
            conn = sqlite3.connect(root/"manifest.sqlite")
            conn.execute("CREATE TABLE original_sequences(sequence_id INTEGER,source_ip TEXT,window_start_ms INTEGER,split TEXT,attack_flows INTEGER)")
            conn.executemany("INSERT INTO original_sequences VALUES(?,?,?,?,?)", [(1, "a", 0, "train", 2), (2, "b", 0, "validation", 0)])
            conn.commit(); conn.close()
            write(root/"manifest.json", dict(source=signature(raw)))
            before = digest(raw)
            reader = EventReader(root)
            try:
                a, _ = reader.events(1, "Alpha")
                self.assertEqual(a[:, bf.E["raw_rowid"]].tolist(), [1])
                with self.assertRaisesRegex(ValueError, "training donors"):
                    reader.events(2)
            finally:
                reader.close()
            self.assertEqual(digest(raw), before)

    def test_physical_vector_routes_scores_and_unchanged_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            one, two = root/"one", root/"two"
            one.mkdir(); two.mkdir()
            a, b, c = events(5, 1, 1), events(500, 0, 100), events(100, 0, 1000)
            state = raw_bucket(np.concatenate((a, b, c)))
            vocab = {field: sorted(set(map(int, state["hist_"+field][:, 1]))) for field in FIELDS}
            tables = {field: np.random.default_rng(4).normal(size=(len(vocab[field])+1, 2)).astype(np.float32) for field in LEARNED}
            for table in tables.values():
                table[0] = 0
            np.savez(one/"embeddings.npz", **tables)
            width = 82+sum(len(vocab[f])+1 if f not in LEARNED else 2 for f in FIELDS)
            write(one/"schema.json", dict(vocabularies=vocab))
            write(one/"scaler.json", dict(mean=[0]*width, scale=[1]*width))
            write(one/"manifest.json", dict(complete=True))
            vocab2 = dict(protocol=[6, 17], l7=[5, 7], service=[1])
            write(two/"schema.json", dict(vocabularies=vocab2))
            write(two/"scaler.json", dict(mean=[0]*82, scale=[1]*82))
            write(two/"manifest.json", dict(complete=True))
            args = dict(slots={f:len(v)+1 for f,v in vocab2.items()}, category_dim=2, bucket_dim=4)
            bucket = BucketAE(**args)
            torch.save(dict(model_args=args, model_state=bucket.state_dict()), two/"frozen_bucket_encoder.pt")
            models = {}
            for name, folder, d in (("scheme1_l2", one, width), ("scheme2", two, 5)):
                run = root/name
                run.mkdir()
                model_args = dict(features=d, width=4, latent=3, timesteps=10)
                model = TemporalAE(**model_args)
                torch.save(dict(model_args=model_args, model_state=model.state_dict()), run/"best.pt")
                write(run/"linear_probe.json", dict(mean=[0]*3, scale=[1]*3, weights=[1]*3, bias=0))
                models[name] = dict(config=dict(dataset=str(folder)), run=run, report_path=run/"report.json",
                    report=dict(linear_probe=dict(operating_points={p:dict(threshold_logit=0.) for p in ("train_benign_99pct", "train_benign_99.9pct")})))
            class Reader:
                def events(self, sid, attack=None):
                    matrix = {1:a, 2:b, 3:c}[sid]
                    return matrix.copy(), (str(sid), 0, "train", int(matrix[:, bf.E["label"]].sum()))
            plan = [dict(pair_id=0, attack="Alpha", attack_id=1, attack_host="1", background_id=2,
                background_host="2", benign_replacement_id=3, attack_budget=5, attack_flows_available=5)]
            reps = Representations(models)
            before = {p:digest(p) for p in root.rglob("*") if p.is_file()}
            out = root/"out"
            out.mkdir()
            values, metadata = build_cases(reps, Reader(), plan, out, 4)
            self.assertEqual(len(metadata), 12)
            for i in range(0, 12, 2):
                self.assertEqual(metadata[i]["total_flows"], metadata[i+1]["total_flows"])
                self.assertEqual(metadata[i]["attack_flows"], 5)
                self.assertEqual(metadata[i+1]["attack_flows"], 0)
            self.assertEqual(values[ROUTES[0]].shape, (12, 10, width))
            np.testing.assert_allclose(values[ROUTES[1]][-2:], values[ROUTES[2]][-2:], atol=1e-6)
            predictions = score_cases(models, values, metadata, out)
            rows = summarize(metadata, predictions, out)
            self.assertTrue(rows)
            report_markdown(out, rows, plan, {})
            report_markdown(out, [r for r in rows if r["target_attack_percent"] != 100], plan, {})
            transitions = baseline_transitions(metadata, predictions)
            self.assertEqual(len(transitions), len(RATIOS)*len(ROUTES)*2)
            for row in transitions:
                if row["target_attack_percent"] == 100:
                    self.assertEqual(row["lost"], 0)
                    self.assertEqual(row["gained"], 0)
            reps.assert_unchanged()
            for path, expected in before.items():
                self.assertEqual(digest(path), expected)


if __name__ == "__main__":
    unittest.main()
