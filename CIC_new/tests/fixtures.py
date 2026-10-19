"""Small synthetic CIC flow records; never reads the full dataset."""
import sqlite3
import numpy as np
from CIC_new.preprocessing import bucket_statistics as bf, split_policy as sp


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
