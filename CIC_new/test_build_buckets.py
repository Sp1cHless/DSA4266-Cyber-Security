"""Small deterministic checks; never reads the full CIC database."""
from collections import Counter
from contextlib import nullcontext, redirect_stderr
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

import bucket_features as bf
import build_buckets as build
import split_policy as sp


def event(start=1000,label=0,protocol=6,l7=5,destination=1,**fields):
    row=np.zeros(len(bf.EVENT_COLUMNS),dtype=np.int64)
    values={"raw_rowid":1,"flow_start_milliseconds":start,"flow_end_milliseconds":start+120000,
            "destination_id":destination,"l4_src_port":100,"l4_dst_port":80,"protocol":protocol,"l7_proto":l7,
            "in_bytes":100,"out_bytes":20,"in_pkts":2,"out_pkts":1,"flow_duration_milliseconds":120000,
            "tcp_flags":2,"client_tcp_flags":2,"server_tcp_flags":16,"min_ip_pkt_len":40,"max_ip_pkt_len":100,
            "num_pkts_up_to_128_bytes":3,"tcp_win_max_in":1000,"tcp_win_max_out":2000,"label":label,**fields}
    for name,value in values.items():
        row[bf.E[name]]=value
    return row


def fixture_database(path):
    fields=[n for n in bf.EVENT_COLUMNS if n not in {"raw_rowid","destination_id"}]
    conn=sqlite3.connect(path)
    conn.execute("CREATE TABLE network_flows("+",".join(f'"{n}" INTEGER' for n in fields)+",ipv4_src_addr TEXT,ipv4_dst_addr TEXT,attack TEXT)")
    rows=[]
    # All five sources share global times in 20 blocks; Beta includes background.
    base=17574*86400000
    starts_array,first_array,end_array,sources,positives,cells=[],[],[],[],[],{}
    for source in range(5):
        ip=f"10.0.{0 if source<3 else 1}.{source+1}"
        for i in range(60):
            window=base+(i//3)*900000+200000+(i%3)*100000
            starts=[1000] if source==0 else [1000,21000,41000] if source==1 else [1000,11000,31000,51000,71000,91000] if source==2 else [1000,11000]
            for j,offset in enumerate(starts):
                label=int(source>=3 and (source==3 or j==1))
                attack="Alpha" if source==3 else "Beta" if label else "Benign"
                r=event(window+offset,label,6 if source>=3 else 17,7 if label else 5,in_bytes=100+source*10+i)
                rows.append(tuple(int(r[bf.E[n]]) for n in fields)+(ip,"192.0.2.1",attack))
            index=len(starts_array)
            starts_array.append(window)
            first_array.append(window+starts[0])
            end_array.append(window+starts[-1]+120000)
            sources.append(source)
            positives.append(2 if source==3 else 1 if source==4 else 0)
            if source>=3:
                cells[(index,"Alpha" if source==3 else "Beta")]=positives[-1]
    conn.executemany("INSERT INTO network_flows VALUES("+",".join("?" for _ in range(len(fields)+3))+")",rows)
    # Put a held-out-only L7 code in a test window under the new block policy.
    kept,*_=sp.split(*map(np.asarray,(starts_array,first_array,end_array,sources,positives)),cells)
    index=int(np.flatnonzero((kept==2)&(np.asarray(sources)==3))[0])
    test_window=starts_array[index]
    conn.execute("UPDATE network_flows SET l7_proto=999 WHERE ipv4_src_addr='10.0.1.4' AND flow_start_milliseconds>=? AND flow_start_milliseconds<?",(test_window,test_window+100000))
    conn.commit()
    conn.close()


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

    def test_shift_preserves_events_and_long_endpoints(self):
        a=np.stack([event(20000,label=1),event(70000,label=1)])
        b=np.stack([event(5000)])
        self.assertEqual(bf.shift_bounds(a),(-20000,29999))
        merged=bf.superimpose(b,a,10000)
        malicious=merged[merged[:,bf.E["label"]]==1]
        np.testing.assert_array_equal(malicious[:,bf.E["flow_start_milliseconds"]],[30000,80000])
        np.testing.assert_array_equal(malicious[:,bf.E["flow_end_milliseconds"]]-malicious[:,bf.E["flow_start_milliseconds"]],[120000,120000])
        unchanged=[i for i,n in enumerate(bf.EVENT_COLUMNS) if n not in {"flow_start_milliseconds","flow_end_milliseconds"}]
        np.testing.assert_array_equal(malicious[:,unchanged],a[:,unchanged])
        self.assertEqual(len(merged),3)
        with self.assertRaises(ValueError):
            bf.superimpose(b,a,30000)

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

    def test_allocation_and_no_same_source_sampler(self):
        pool=build.Pool({"1":{"A":[1],"B":[2]},"5-10":{"A":[3]}})
        rng=np.random.default_rng(0)
        for _ in range(10):
            sid,source,_=pool.pick(rng,exclude="A")
            self.assertEqual((sid,source),(2,"B"))


class PipelineTests(unittest.TestCase):
    def test_complete_cli_resume_and_report_only(self):
        kept=os.environ.get("CIC_BUCKET_TEST_DIR")
        context=nullcontext(kept) if kept else tempfile.TemporaryDirectory()
        with context as folder:
            folder=Path(folder).resolve()
            folder.mkdir(parents=True,exist_ok=True)
            db=folder/"fixture.db"
            fixture_database(db)
            before=hashlib.sha256(db.read_bytes()).hexdigest()
            out,reports=folder/"dataset",folder/"reports"
            args=["--db",str(db),"--output",str(out),"--reports-dir",str(reports),"--shard-sequences","7","--donor-cache-mb","1"]
            self.assertEqual(build.main(args),0)
            plan=build.read_json(out/"augmentation_plan.json")
            conn=sqlite3.connect(out/"manifest.sqlite")
            originals=dict(conn.execute("SELECT split,COUNT(*) FROM original_sequences GROUP BY split"))
            real_positive=dict(conn.execute("SELECT split,SUM(attack_flows>0) FROM original_sequences GROUP BY split"))
            n,p=originals["train"],real_positive["train"]
            self.assertEqual((plan["original_train_N"],plan["original_train_P"],plan["final_train"]),(n,p,n+2*p))
            self.assertGreater(originals.get("excluded",0),0)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM samples p JOIN original_sequences s ON p.left_id=s.sequence_id WHERE s.split='excluded'").fetchone()[0],0)
            schema=build.read_json(out/"feature_schema.json")
            self.assertNotIn(999,schema["l7_vocabulary"])
            counts,positive=Counter(),Counter()
            unknown=0
            original_training_log_counts=[]
            for split in build.SPLITS:
                for file in sorted((out/"splits"/split).glob("*.npz")):
                    with np.load(file,allow_pickle=False) as arrays:
                        self.assertEqual(arrays["X"].shape[1:],(10,schema["feature_count"]))
                        self.assertTrue(np.all(arrays["X"][arrays["mask"]==0]==0))
                        counts[split]+=len(arrays["sample_id"])
                        positive[split]+=int(arrays["y_sequence"].sum())
                    with np.load(out/"raw_shards"/split/file.name,allow_pickle=False) as raw:
                        if split=="test":
                            unknown+=raw["X_raw"][:,:,-1].sum()
                        if split=="train":
                            for i,sid in enumerate(raw["sample_id"]):
                                kind=conn.execute("SELECT sample_type FROM samples WHERE sample_id=?",(int(sid),)).fetchone()[0]
                                if kind.startswith("real_"):
                                    original_training_log_counts.extend(np.log1p(raw["X_raw"][i,raw["mask"][i]!=0,0]))
            self.assertEqual(dict(counts),{"train":n+2*p,"validation":originals["validation"],"test":originals["test"]})
            self.assertEqual(dict(positive),{"train":2*p,"validation":real_positive["validation"],"test":real_positive["test"]})
            self.assertGreater(unknown,0)
            self.assertAlmostEqual(build.read_json(out/"scaler.json")["mean"][0],float(np.mean(original_training_log_counts)))
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM samples WHERE right_id IS NOT NULL AND split!='train'").fetchone()[0],0)
            conn.close()
            self.assertTrue(all(r["passed"] for r in build.read_json(reports/"summary.json")["checks"]))
            with zipfile.ZipFile(reports/"bucket_summary.xlsx") as z:
                ns={"s":"http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
                root=ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
                cached=lambda address:float(root.find(f".//s:c[@r='{address}']/s:v",ns).text)
                self.assertEqual(cached("B5"),n+2*p)
                self.assertEqual(cached("C5"),2*p)
                self.assertAlmostEqual(cached("D5"),2*p/(n+2*p))
                split_root=ET.fromstring(z.read("xl/worksheets/sheet2.xml"))
                split_cached=lambda address:float(split_root.find(f".//s:c[@r='{address}']/s:v",ns).text)
                self.assertEqual(split_cached("B5"),n)
                self.assertAlmostEqual(split_cached("D5"),n/sum(originals[s] for s in build.SPLITS))
                self.assertAlmostEqual(sum(split_cached(f"E{i}") for i in range(5,9)),1)
            with patch.dict(os.environ,{"CIC_BUCKET_PREVIEW_SHEETS":"Summary,Split,Method"}):
                self.assertEqual(build.main(args+["--resume"]),0)
                # Exercise split-stage restart and aggregation-stage restart separately.
                for completed in (["index"],["index","split","plan"]):
                    state=build.read_json(out/"run_status.json")
                    state.update(complete=False,completed_stages=completed)
                    build.write_json(out/"run_status.json",state)
                    self.assertEqual(build.main(args+["--resume"]),0)
                self.assertEqual(build.main(["--report-only",str(out)]),0)
            state=build.read_json(out/"run_status.json")
            old=json.loads(json.dumps(state))
            old["config"]["pipeline_version"]=1
            build.write_json(out/"run_status.json",old)
            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    build.main(args+["--resume"])
                with self.assertRaises(SystemExit):
                    build.main(["--report-only",str(out)])
            build.write_json(out/"run_status.json",state)
            self.assertEqual(hashlib.sha256(db.read_bytes()).hexdigest(),before)
            self.assertTrue(build.read_json(out/"run_status.json")["complete"])
            self.assertEqual(len(list((out/"_excel_runtime/previews").glob("*.png"))),7)


if __name__=="__main__":
    unittest.main()
