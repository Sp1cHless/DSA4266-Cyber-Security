"""Checks for bucket statistics and training-only scaling."""
import unittest
import numpy as np
from CIC_new.preprocessing import bucket_statistics as bf
from CIC_new.tests.fixtures import event


class FeatureTests(unittest.TestCase):
    def test_known_bucket_categories_flags_and_timing(self):
        schema=bf.make_schema([5,88])
        events=np.stack([event(1000,l7=5),event(1000,label=1,l7=88,destination=2),event(3000,protocol=17,l7=999,tcp_flags=255)])
        x,mask,y,flows,attacks=bf.aggregate(events,schema)
        get=lambda name:x[0,schema["feature_names"].index(name)]
        self.assertEqual(len(bf.NUMERIC_NAMES),82)
        self.assertEqual(get("flow_count"),3)
        self.assertEqual(get("in_bytes_sum"),300)
        self.assertAlmostEqual(get("protocol_tcp_ratio"),2/3)
        self.assertEqual(get("tcp_flag_syn_ratio"),1)
        self.assertEqual(get("tcp_flag_fin_ratio"),0)  # Non-TCP mask ignored.
        self.assertAlmostEqual(get("src_port_reuse_ratio"),2/3)
        self.assertEqual(get("dst_ip_nunique"),2)
        self.assertAlmostEqual(get("l7_UNK_ratio"),1/3)
        self.assertEqual(get("l7_proto_nunique"),3)
        self.assertEqual(get("flow_start_gap_mean_ms"),1000)
        self.assertEqual(get("flow_start_gap_max_ms"),2000)
        self.assertEqual(get("flow_start_span_ms"),2000)
        self.assertEqual((mask.sum(),y.sum(),flows.sum(),attacks.sum()),(1,1,3,1))
        self.assertTrue(np.all(x[1:]==0))

    def test_conditioned_categories_and_normalization(self):
        s=bf.make_schema([5])
        rows=np.stack([event(1000,protocol=1,icmp_ipv4_type=0),event(10000,protocol=17,dns_query_id=10,dns_query_type=28)])
        x,mask,*_=bf.aggregate(rows,s)
        self.assertEqual(x[0,s["feature_names"].index("icmp_type_nunique")],1)
        self.assertEqual(x[1,s["feature_names"].index("dns_query_type_top1_ratio")],1)
        moments=bf.Moments(s["feature_count"])
        moments.add(bf.transform_log(x[mask==1],s))
        scaler=moments.finish()
        result=bf.normalize(x[None],mask[None],s,scaler)
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(np.all(result[0,2:]==0))
        self.assertEqual(result.dtype,np.float32)
