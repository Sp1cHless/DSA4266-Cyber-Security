"""End-to-end preprocessing: raw data -> sequences -> buckets -> train/val/test.

No neural network is trained here. Raw histograms, train-only vocabularies and
scaling are exported for any downstream representation or prediction model.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sqlite3

import numpy as np

from . import bucket_statistics as bf

from . import sequence_index
from .temporal import DEFAULT_GRID, TemporalGrid

FIELDS = ("protocol", "l7", "service")
NUMERIC_SCHEMA = bf.make_schema([])
NUMERIC_SCHEMA["feature_names"] = bf.NUMERIC_NAMES
NUMERIC_SCHEMA["feature_count"] = 82
NUMERIC_SCHEMA["log1p_columns"] = [i for i,n in enumerate(bf.NUMERIC_NAMES)
                                  if not n.endswith(("_ratio", "_entropy"))]


def service_ids(ports):
    result=np.full(len(ports),7,dtype=np.int64)
    for code,values in enumerate(((53,), (80,443), (22,), (20,21),
                                  (25,110,143,465,587,993,995), (3389,), (139,445))):
        result[np.isin(ports,values)]=code
    return result


def raw_bucket(events, grid=DEFAULT_GRID):
    x,mask,y,flows,attacks=bf.aggregate(events,bf.make_schema([], grid))
    bucket=events[:,bf.E["flow_start_milliseconds"]]//grid.bucket_ms
    categories={"protocol":events[:,bf.E["protocol"]],"l7":events[:,bf.E["l7_proto"]],
                "service":service_ids(events[:,bf.E["l4_dst_port"]])}
    hist={}
    for field,codes in categories.items():
        pairs,counts=np.unique(np.column_stack([bucket,codes]),axis=0,return_counts=True)
        hist[field]=np.column_stack([pairs,counts]).astype(np.int64)
    return dict(numeric=x[:,:82],mask=mask,y_bucket=y,y_sequence=np.uint8(y.max()),
                flow_counts=flows,attack_flow_counts=attacks),hist


def aggregate_originals(out,shard_size,grid=DEFAULT_GRID):
    """All original buckets are computed BEFORE any split/vocabulary/scaler."""
    conn=sequence_index.connect(out)
    try:
        rawdir=out/"raw_originals";rawdir.mkdir()
        conn.execute("CREATE TABLE v3_raw_locations(sample_id INTEGER PRIMARY KEY,shard TEXT,row_index INTEGER)")
        query="SELECT sequence_id,"+",".join(bf.EVENT_COLUMNS)+" FROM work.events ORDER BY sequence_id,flow_start_milliseconds,raw_rowid"
        meta=iter(conn.execute("SELECT sequence_id,window_start_ms FROM original_sequences ORDER BY sequence_id"))
        buffer=[];part=0;total=0
        def flush():
            nonlocal part
            ids=np.array([a[0] for a in buffer],dtype=np.int64)
            payload={key:np.stack([a[1][key] for a in buffer]) for key in buffer[0][1]}
            for field in FIELDS:
                coos=[]
                for i,(_,_,hist) in enumerate(buffer):
                    coo=hist[field].copy();coo[:,0]+=i*grid.timesteps;coos.append(coo)
                payload["categories_"+field]=np.concatenate(coos)
            payload["sample_id"]=ids
            filename=f"part_{part:06d}.npz"
            np.savez_compressed(rawdir/filename,**payload)
            conn.executemany("INSERT INTO v3_raw_locations VALUES(?,?,?)",
                             [(int(sid),filename,i) for i,sid in enumerate(ids)])
            conn.commit();buffer.clear();part+=1
        for sid,group in itertools.groupby(conn.execute(query),key=lambda r:r[0]):
            msid,start=next(meta)
            if sid!=msid:raise AssertionError("Original sequence ordering mismatch")
            events=bf.local_events(np.asarray([r[1:] for r in group],dtype=np.int64),start)
            values,hist=raw_bucket(events,grid);buffer.append((sid,values,hist));total+=1
            if len(buffer)>=shard_size:flush()
            if total%10000==0:print(f"[bucket-before-split] {total:,} originals",flush=True)
        if buffer:flush()
        print(f"[bucket-before-split] completed {total:,} originals",flush=True)
    finally:conn.close()


def selected_distribution(coo,rows,vocabulary,n,timesteps=10):
    """Map raw category histograms AFTER train-only vocabulary is fixed."""
    mapping={int(code):i+1 for i,code in enumerate(vocabulary)}  # slot 0 = UNK
    old_to_new=np.full(int(rows.max())+1 if len(rows) else 0,-1,dtype=int)
    old_to_new[rows]=np.arange(len(rows))
    old=coo[:,0]//timesteps
    eligible=(old<len(old_to_new))
    coo=coo[eligible];old=old[eligible]
    eligible=old_to_new[old]>=0;coo=coo[eligible];old=old[eligible]
    result=np.zeros((n,timesteps,len(vocabulary)+1),dtype=np.float32)
    new=old_to_new[old];b=coo[:,0]%timesteps
    categories=np.fromiter((mapping.get(int(c),0) for c in coo[:,1]),dtype=np.int64,count=len(coo))
    np.add.at(result,(new,b,categories),coo[:,2])
    denom=result.sum(axis=-1,keepdims=True)
    np.divide(result,denom,out=result,where=denom>0)
    return result


def fit_and_export(out,grid=DEFAULT_GRID):
    conn=sequence_index.connect(out)
    try:
        split_lookup={int(sid):name for sid,name in conn.execute("SELECT sequence_id,split FROM original_sequences")}
        vocab={f:set() for f in FIELDS};moments=bf.Moments(82)
        files=sorted((out/"raw_originals").glob("part_*.npz"))
        for file in files:
            with np.load(file,allow_pickle=False) as a:
                train=np.array([split_lookup[int(s)]=="train" for s in a["sample_id"]])
                active=a["mask"].astype(bool)&train[:,None]
                moments.add(bf.transform_log(a["numeric"][active],NUMERIC_SCHEMA))
                for field in FIELDS:
                    coo=a["categories_"+field]
                    vocab[field].update(map(int,coo[train[coo[:,0]//grid.timesteps],1]))
        vocab={f:sorted(v) for f,v in vocab.items()}
        scaler=moments.finish()
        sequence_index.write_json(out/"scaler.json",{**scaler,"population":"real retained training active buckets only"})
        schema=dict(version=3,numeric_names=bf.NUMERIC_NAMES,vocabularies=vocab,
                    category_slots={f:len(v)+1 for f,v in vocab.items()},unknown_index=0,
                    **grid.settings(),
                    log1p_columns=NUMERIC_SCHEMA["log1p_columns"],mask_separate=True,
                    augmentation="training-only circular shift then frozen-embedding addition and L2")
        sequence_index.write_json(out/"schema.json",schema)
        conn.execute("CREATE TABLE v3_locations(sample_id INTEGER PRIMARY KEY,split TEXT,shard TEXT,row_index INTEGER)")
        counts={s:0 for s in sequence_index.SPLITS};parts={s:0 for s in sequence_index.SPLITS}
        for split in sequence_index.SPLITS:(out/"splits"/split).mkdir(parents=True)
        for file in files:
            with np.load(file,allow_pickle=False) as a:
                allocation=np.array([split_lookup[int(s)] for s in a["sample_id"]])
                for split in sequence_index.SPLITS:
                    rows=np.flatnonzero(allocation==split)
                    if not len(rows):continue
                    payload={key:a[key][rows] for key in ("mask","y_bucket","y_sequence","flow_counts","attack_flow_counts","sample_id")}
                    payload["numeric"]=bf.normalize(a["numeric"][rows],payload["mask"],NUMERIC_SCHEMA,scaler)
                    for field in FIELDS:
                        h=selected_distribution(a["categories_"+field],rows,vocab[field],len(rows),grid.timesteps)
                        if not np.allclose(h.sum(axis=-1),payload["mask"]):raise AssertionError("Histogram/mask mismatch")
                        payload["hist_"+field]=h
                    filename=f"part_{parts[split]:06d}.npz";parts[split]+=1;counts[split]+=len(rows)
                    np.savez_compressed(out/"splits"/split/filename,**payload)
                    conn.executemany("INSERT INTO v3_locations VALUES(?,?,?,?)",
                        [(int(sid),split,filename,i) for i,sid in enumerate(payload["sample_id"])])
        conn.commit()
        for split,count in counts.items():
            expected=conn.execute("SELECT COUNT(*) FROM original_sequences WHERE split=?",(split,)).fetchone()[0]
            if count!=expected:raise AssertionError("Real original export mismatch")
        sequence_index.write_json(out/"v3_counts.json",counts)
        return counts
    finally:conn.close()


def import_csv(source,target,chunksize=250000):
    import pandas as pd
    required=[n for n in bf.EVENT_COLUMNS if n not in ("raw_rowid","destination_id")]+["ipv4_src_addr","ipv4_dst_addr","attack"]
    numeric=set(required)-{"ipv4_src_addr","ipv4_dst_addr","attack"}
    conn=sqlite3.connect(target)
    try:
        for df in pd.read_csv(source,usecols=lambda n:n.lower() in required,chunksize=chunksize):
            df.columns=df.columns.str.lower()
            if set(required)-set(df):raise ValueError("CSV missing required columns")
            if df.isna().any().any():raise ValueError("Missing raw values")
            for n in numeric:
                a=pd.to_numeric(df[n],errors="raise")
                if not np.isfinite(a).all() or (a<0).any() or np.any(a!=np.floor(a)):
                    raise ValueError(f"Invalid integer field {n}")
                df[n]=a.astype(np.int64)
            df.to_sql("network_flows",conn,if_exists="append",index=False)
            print(f"[CSV import] appended {len(df):,} flows",flush=True)
    finally:conn.close()


def prepare(out,db=None,csv=None,seed=4266,shard_size=2048,
            sequence_seconds=100,bucket_seconds=10):
    grid=TemporalGrid(sequence_seconds,bucket_seconds)
    out=Path(out).resolve()
    if bool(db)==bool(csv):raise ValueError("Supply exactly one full-schema --db or --csv")
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    supplied=Path(db or csv).resolve();signature=sequence_index.signature(supplied)
    manifest=dict(version=3,complete=False,seed=seed,source=signature,
                  order=["sequence_construction","bucket_aggregation","split","training_only_fitting","embedding_space_training_augmentation"],
                  categorical_fields=list(FIELDS),model_training_started=False,
                  temporal_grid=grid.settings())
    sequence_index.write_json(out/"manifest.json",manifest)
    if csv:
        db=out/"raw_source.sqlite";import_csv(Path(csv),db)
    else:db=Path(db).resolve()
    sequence_index.index_source(out,db,grid)
    aggregate_originals(out,shard_size,grid)
    sequence_index.split_originals(out,seed,grid)
    counts=fit_and_export(out,grid)
    if signature!=sequence_index.signature(supplied):raise AssertionError("Source changed")
    manifest.update(complete=True,counts=counts,
                    split_policy="global 15min blocks; 100s guards and actual-end envelope exclusions",
                    augmentation_materialized=False)
    sequence_index.write_json(out/"manifest.json",manifest)
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument("--db",type=Path);group.add_argument("--csv",type=Path)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--seed",type=int,default=4266)
    p.add_argument("--shard-size",type=int,default=2048)
    p.add_argument("--sequence-seconds",type=int,default=100)
    p.add_argument("--bucket-seconds",type=int,default=10)
    args=p.parse_args()
    if args.shard_size<1:p.error("Positive shard size required")
    try:
        TemporalGrid(args.sequence_seconds,args.bucket_seconds)
    except ValueError as error:
        p.error(str(error))
    import logging
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    prepare(args.out,args.db,args.csv,args.seed,args.shard_size,args.sequence_seconds,args.bucket_seconds)
