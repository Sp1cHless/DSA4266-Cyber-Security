"""Model-independent original-data loading and training-only donor selection."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

FIELDS=("protocol","l7","service")
META=np.dtype([("id","<i8"),("ip","U15"),("start","<i8"),("active","u1"),
               ("label","u1"),("attack","U64")])


def read_json(path):return json.loads(Path(path).read_text())


def fingerprint(data):
    digest=hashlib.sha256()
    for name in ("manifest.json","schema.json","scaler.json"):
        digest.update((Path(data)/name).read_bytes())
    return digest.hexdigest()


def metadata(data,split):
    if split not in ("train","validation","test"):raise ValueError(split)
    conn=sqlite3.connect(f"file:{Path(data).resolve()/'manifest.sqlite'}?mode=ro",uri=True)
    try:
        rows=conn.execute("""SELECT sequence_id,source_ip,window_start_ms,active_buckets,
            attack_flows>0,primary_attack FROM original_sequences WHERE split=? ORDER BY sequence_id""",(split,)).fetchall()
        return np.array(rows,dtype=META)
    finally:conn.close()


class MappedArrays:
    """Explicit mmap lifetime, particularly for Windows directory cleanup."""
    def close(self):
        for array in getattr(self,"arrays",{}).values():
            mapping=getattr(array,"_mmap",None)
            if mapping is not None and not mapping.closed:mapping.close()
        self.arrays={}

    def __enter__(self):return self

    def __exit__(self,*exc):self.close()


class Originals(MappedArrays):
    """Normalised originals cached once as mmap arrays for fast random batches.

    NPZ decompression happens once, not for every random donor. Caches can be
    large on disk; they do not load the full corpus into RAM.
    """
    def __init__(self,data,split):
        self.data=Path(data).resolve();self.split=split
        manifest=read_json(self.data/"manifest.json")
        if manifest.get("version")!=3 or not manifest.get("complete"):raise ValueError("Completed V3 data required")
        self.schema=read_json(self.data/"schema.json");self.meta=metadata(self.data,split)
        n=len(self.meta)
        folder=self.data/"mmap"/split;identity=fingerprint(self.data)
        shapes={"numeric":(n,10,82),"mask":(n,10),"y_bucket":(n,10),"flow_counts":(n,10),
                **{"hist_"+f:(n,10,self.schema["category_slots"][f]) for f in FIELDS}}
        if not folder.exists():
            partial=folder.with_name(split+".partial")
            if partial.exists():raise FileExistsError(f"Interrupted/concurrent cache: {partial}; use a fresh data copy or inspect it")
            partial.mkdir(parents=True)
            arrays={k:np.lib.format.open_memmap(partial/(k+".npy"),mode="w+",dtype=np.int64 if k=="flow_counts" else np.float32,shape=s)
                    for k,s in shapes.items()}
            seen=np.zeros(n,dtype=bool)
            for file in sorted((self.data/"splits"/split).glob("part_*.npz")):
                with np.load(file,allow_pickle=False) as a:
                    positions=np.searchsorted(self.meta["id"],a["sample_id"])
                    if np.any(positions>=n) or not np.array_equal(self.meta["id"][positions],a["sample_id"]):raise AssertionError("Wrong split IDs")
                    if seen[positions].any():raise AssertionError("Duplicate original")
                    for k,target in arrays.items():target[positions]=a[k]
                    seen[positions]=True
            if not seen.all():raise AssertionError("Missing originals")
            for a in arrays.values():
                a.flush();a._mmap.close()
            del arrays
            (partial/"identity.json").write_text(json.dumps({"fingerprint":identity,"n":n}))
            partial.rename(folder)
        if read_json(folder/"identity.json")!={"fingerprint":identity,"n":n}:raise ValueError("Stale mmap cache")
        self.arrays={k:np.load(folder/(k+".npy"),mmap_mode="r") for k in shapes}

    def take(self,indices):return {k:np.array(a[indices],copy=True) for k,a in self.arrays.items()}


class Pool:
    def __init__(self,groups):
        self.groups={g:{h:np.array(ids,dtype=np.int64) for h,ids in hosts.items()} for g,hosts in groups.items()}
        self.hosts={g:list(hosts) for g,hosts in self.groups.items()}

    def pick(self,rng,exclude=None):
        valid=[g for g,h in self.hosts.items() if len(h)>1 or h[0]!=exclude]
        if not valid:raise ValueError("No different-source training donor")
        g=valid[int(rng.integers(len(valid)))];hosts=self.hosts[g]
        if exclude in self.groups[g]:
            # A small dictionary map avoids rebuilding all host lists per draw.
            if not hasattr(self,"host_index"):self.host_index={g:{h:i for i,h in enumerate(hs)} for g,hs in self.hosts.items()}
            skip=self.host_index[g][exclude];j=int(rng.integers(len(hosts)-1));j+=j>=skip
        else:j=int(rng.integers(len(hosts)))
        host=hosts[j];ids=self.groups[g][host]
        return int(ids[int(rng.integers(len(ids)))]),host,g


class Donors:
    def __init__(self,data,train_meta,seed=4266):
        self.meta=train_meta;self.rng=np.random.default_rng(seed)
        self.benign=np.flatnonzero(train_meta["label"]==0);self.positive=np.flatnonzero(train_meta["label"]==1)
        if not len(self.benign) or not len(self.positive):raise ValueError("Both original training classes required")
        bg=defaultdict(lambda:defaultdict(list));atk=defaultdict(lambda:defaultdict(list))
        for i in self.benign:
            m=train_meta[i];band=0 if m["active"]==1 else 1 if m["active"]<=4 else 2
            bg[band][str(m["ip"])].append(int(i))
        conn=sqlite3.connect(f"file:{Path(data).resolve()/'manifest.sqlite'}?mode=ro",uri=True)
        try:
            for sid,attack in conn.execute("SELECT sequence_id,attack FROM sequence_attacks"):
                i=int(np.searchsorted(train_meta["id"],sid))
                if i<len(train_meta) and train_meta[i]["id"]==sid:atk[attack][str(train_meta[i]["ip"])].append(i)
        finally:conn.close()
        self.bg=Pool(bg);self.atk=Pool(atk);self.draw=0

    def batch(self,size,real_only=False,circular=True):
        if size<4 or size%4:raise ValueError("Batch size must be a positive multiple of four")
        left=[];right=[];shift=[];provenance=[]
        for _ in range(size//4):
            for label,pool in ((0,self.benign),(1,self.positive)):
                repetitions=2 if real_only else 1
                for _ in range(repetitions):
                    i=int(self.rng.choice(pool));left.append(i);right.append(-1);shift.append(0)
                    provenance.append(dict(synthetic=False,label=label,left_id=int(self.meta[i]["id"])))
                if not real_only:
                    b,host,category=(self.atk if label else self.bg).pick(self.rng)
                    a,_,band=self.bg.pick(self.rng,exclude=host)
                    # Consume the same RNG draw for a zero-shift control, so
                    # donor identities match when only rotation is changed.
                    k=int(self.rng.integers(10));k=k if circular else 0;self.draw+=1
                    left.append(a);right.append(b);shift.append(k)
                    provenance.append(dict(synthetic=True,label=label,left_id=int(self.meta[a]["id"]),
                        right_id=int(self.meta[b]["id"]),left_ip=str(self.meta[a]["ip"]),right_ip=str(self.meta[b]["ip"]),
                        right_shift_buckets=k,background_band=int(band),target_attack=str(category) if label else None,
                        generation_cycle=(self.draw-1)//(2*len(self.positive))))
        order=self.rng.permutation(size)
        return np.array(left)[order],np.array(right)[order],np.array(shift)[order],[provenance[i] for i in order]
