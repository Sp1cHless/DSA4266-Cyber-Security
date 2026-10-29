"""Focused checks for candidate extraction and grouped training-only screening."""
import tempfile
from pathlib import Path
import unittest
import numpy as np

from analysis.audit_infilteration_features import OMITTED, extract, candidate_names, probes, split_indices
from CIC_new.tests.fixtures import event
from CIC_new.preprocessing import bucket_statistics as bf


def row(start=1000, **changes):
    values = dict(zip(bf.EVENT_COLUMNS, map(int, event(start))))
    values.pop("destination_id")
    values.update({name: 0 for name in OMITTED})
    values.update(ipv4_src_addr="10.0.0.1", ipv4_dst_addr="192.0.2.1", attack="Benign")
    values.update(changes)
    return values


class FeatureAuditTest(unittest.TestCase):
    def test_added_fields_and_distribution_preserve_old_representation(self):
        schema = {"vocabularies": {"protocol": [6, 17], "l7": [5], "service": list(range(8))}}
        first = [row(1000, in_bytes=50), row(2000, in_bytes=150)]
        second = [row(1000, in_bytes=100), row(2000, in_bytes=100)]
        old_a, new_a, _ = extract(first, 0, schema)
        old_b, new_b, _ = extract(second, 0, schema)
        np.testing.assert_array_equal(old_a, old_b)
        self.assertFalse(np.array_equal(new_a["distribution"], new_b["distribution"]))
        third = [dict(v, src_to_dst_iat_avg=25, min_ttl=64) for v in second]
        old_c, new_c, _ = extract(third, 0, schema)
        np.testing.assert_array_equal(old_b, old_c)
        self.assertFalse(np.array_equal(new_b["iat"], new_c["iat"]))
        self.assertFalse(np.array_equal(new_b["ttl"], new_c["ttl"]))
        for family in new_c:
            self.assertEqual(new_c[family].shape, (10, len(candidate_names()[family])))
            self.assertTrue(np.all(new_c[family][1:] == 0))
            self.assertTrue(np.isfinite(new_c[family]).all())

    def test_fixed_group_splits_and_probe_outputs(self):
        rows = [{"id": i, "target": i % 2, "host": str(i // 8), "start": (i // 4) * 900000,
                 "active": 1} for i in range(160)]
        for mode in ("time_block", "source_host"):
            fit, hold = split_indices(rows, mode)
            key = "host" if mode == "source_host" else "start"
            self.assertFalse({rows[i][key] for i in fit} & {rows[i][key] for i in hold})
        rng = np.random.default_rng(4266)
        base = rng.random((160, 10))
        extras = {k: rng.random((160, 10, len(v))) for k, v in candidate_names().items()}
        with tempfile.TemporaryDirectory() as temporary:
            result = probes(rows, base, extras, Path(temporary), trees=4)
            self.assertEqual(len(result), 16)
            self.assertTrue(all(0 <= r["holdout_roc_auc"] <= 1 for r in result))
            self.assertTrue((Path(temporary) / "univariate_screening.csv").exists())


if __name__ == "__main__":
    unittest.main()
