"""CIC v1 feature definitions and event-level aggregation. No learned embedding."""
from __future__ import annotations

import numpy as np

WINDOW_MS = 100_000
BUCKET_MS = 10_000
BUCKETS = 10
FLAG_BITS = {"fin": 1, "syn": 2, "rst": 4, "psh": 8, "ack": 16, "urg": 32, "ece": 64, "cwr": 128}
PACKET_BINS = ["num_pkts_up_to_128_bytes", "num_pkts_128_to_256_bytes", "num_pkts_256_to_512_bytes",
               "num_pkts_512_to_1024_bytes", "num_pkts_1024_to_1514_bytes"]
RETRANSMISSION = ["retransmitted_in_bytes", "retransmitted_in_pkts", "retransmitted_out_bytes", "retransmitted_out_pkts"]
# Numeric staging rows use this fixed order. Destination addresses are dictionary
# IDs used exclusively for equality/diversity, never as numeric model inputs.
EVENT_COLUMNS = ["raw_rowid", "flow_start_milliseconds", "flow_end_milliseconds", "destination_id",
                 "l4_src_port", "l4_dst_port", "protocol", "l7_proto", "in_bytes", "out_bytes", "in_pkts", "out_pkts",
                 "flow_duration_milliseconds", "tcp_flags", "client_tcp_flags", "server_tcp_flags",
                 "icmp_ipv4_type", "dns_query_id", "dns_query_type", "dns_ttl_answer", "duration_in", "duration_out",
                 "min_ip_pkt_len", "max_ip_pkt_len", *RETRANSMISSION, *PACKET_BINS, "tcp_win_max_in", "tcp_win_max_out", "label"]
E = {name: i for i, name in enumerate(EVENT_COLUMNS)}


def base_names():
    names = ["flow_count", "in_bytes_sum", "out_bytes_sum", "in_pkts_sum", "out_pkts_sum",
             "duration_mean", "duration_min", "duration_max", "protocol_tcp_ratio", "protocol_udp_ratio",
             "protocol_other_ratio", "protocol_nunique", "l7_proto_nunique", "l7_top1_ratio", "l7_entropy"]
    for prefix in ("src_port", "dst_port"):
        names.extend(f"{prefix}_{suffix}" for suffix in ("nunique", "reuse_ratio", "top1_ratio", "entropy"))
    for prefix in ("tcp_flag", "client_tcp_flag", "server_tcp_flag"):
        names.extend(f"{prefix}_{flag}_ratio" for flag in FLAG_BITS if not (prefix == "server_tcp_flag" and flag == "urg"))
    for prefix, field in (("icmp", "icmp_type"), ("dns", "dns_query_type")):
        names.append(f"{prefix}_flow_ratio")
        names.extend(f"{field}_{suffix}" for suffix in ("nunique", "top1_ratio", "entropy"))
    names.extend(f"dst_ip_{suffix}" for suffix in ("nunique", "reuse_ratio", "top1_ratio", "entropy"))
    assert len(names) == 58
    return names


BASE_NAMES = base_names()
EXTRA_NAMES = ["duration_in_mean", "duration_in_max", "duration_out_mean", "duration_out_max",
               "min_ip_pkt_len_min", "min_ip_pkt_len_mean", "max_ip_pkt_len_mean", "max_ip_pkt_len_max",
               *(f"{name}_sum" for name in RETRANSMISSION), *(f"{name}_ratio" for name in PACKET_BINS),
               "tcp_win_max_in_mean", "tcp_win_max_in_max", "tcp_win_max_out_mean", "tcp_win_max_out_max",
               "flow_start_gap_mean_ms", "flow_start_gap_max_ms", "flow_start_span_ms"]
assert len(EXTRA_NAMES) == 24
NUMERIC_NAMES = BASE_NAMES + EXTRA_NAMES


def make_schema(vocabulary):
    vocabulary = sorted(set(int(v) for v in vocabulary))
    names = NUMERIC_NAMES + [f"l7_code_{v}_ratio" for v in vocabulary] + ["l7_UNK_ratio"]
    return {"version": 1, "window_seconds": 100, "bucket_seconds": 10, "buckets_per_sequence": 10,
            "numeric_feature_count": 82, "l7_vocabulary": vocabulary, "l7_ratio_count": len(vocabulary)+1,
            "feature_names": names, "feature_count": len(names), "mask_separate": True,
            "model_width_with_mask": len(names)+1,
            "log1p_columns": [i for i, name in enumerate(names) if not name.endswith(("_ratio", "_entropy"))],
            "category_rule": "L7 ratios use training vocabulary plus UNK; diversity uses original codes",
            "normalization_population": "original training nonempty buckets only",
            "flag_denominator": "TCP flow count", "dns_predicate": "query_id != 0 OR query_type != 0 OR answer_ttl != 0"}


def diversity(values, denominator=None):
    """nunique, reuse against supplied denominator, top1, normalized entropy."""
    if not len(values):
        return 0., 0., 0., 0.
    _, counts = np.unique(values, return_counts=True)
    n, k = len(values), len(counts)
    denominator = n if denominator is None else denominator
    probabilities = counts / n
    entropy = float(-np.sum(probabilities*np.log(probabilities))/np.log(k)) if k > 1 else 0.
    return float(k), 1-k/denominator, float(counts.max()/denominator), entropy


def aggregate(events, schema):
    """Aggregate local-time events into exactly ten buckets; return unscaled data."""
    if events.ndim != 2 or events.shape[1] != len(EVENT_COLUMNS) or not len(events):
        raise ValueError("Expected a nonempty event matrix with the registered column order")
    starts = events[:, E["flow_start_milliseconds"]]
    if starts.min() < 0 or starts.max() >= WINDOW_MS:
        raise ValueError("Flow start outside the local 100-second window")
    if not np.all(np.isin(events[:, E["label"]], [0, 1])):
        raise ValueError("Nonbinary flow label")
    x = np.zeros((BUCKETS, schema["feature_count"]), dtype=np.float64)
    mask = np.zeros(BUCKETS, dtype=np.uint8)
    y = np.zeros(BUCKETS, dtype=np.uint8)
    flow_counts = np.zeros(BUCKETS, dtype=np.int64)
    attacks = np.zeros(BUCKETS, dtype=np.int64)
    vocabulary_index = {int(v): 82+i for i, v in enumerate(schema["l7_vocabulary"])}
    bucket_indices = starts // BUCKET_MS
    for bucket in np.unique(bucket_indices):
        bucket = int(bucket)
        rows = events[bucket_indices == bucket]
        get = lambda name: rows[:, E[name]]
        n = len(rows)
        proto, l7, duration = get("protocol"), get("l7_proto"), get("flow_duration_milliseconds")
        tcp, icmp = proto == 6, proto == 1
        dns = (get("dns_query_id") != 0) | (get("dns_query_type") != 0) | (get("dns_ttl_answer") != 0)
        # Cast before summing counters to avoid overflow in int64 accumulators.
        sums = lambda name: float(get(name).astype(np.float64).sum())
        l7_k, _, l7_top, l7_entropy = diversity(l7)
        values = [n, *(sums(name) for name in ("in_bytes", "out_bytes", "in_pkts", "out_pkts")),
                  float(duration.mean()), float(duration.min()), float(duration.max()),
                  float(tcp.mean()), float((proto == 17).mean()), float(((proto != 6) & (proto != 17)).mean()),
                  len(np.unique(proto)), l7_k, l7_top, l7_entropy]
        for name in ("l4_src_port", "l4_dst_port"):
            values.extend(diversity(get(name), n))
        for field, prefix in (("tcp_flags", "tcp_flag"), ("client_tcp_flags", "client_tcp_flag"), ("server_tcp_flags", "server_tcp_flag")):
            flags = get(field)[tcp]
            values.extend(float(((flags & bit) != 0).mean()) if len(flags) else 0.
                          for flag, bit in FLAG_BITS.items() if not (prefix == "server_tcp_flag" and flag == "urg"))
        for selection, field in ((icmp, "icmp_ipv4_type"), (dns, "dns_query_type")):
            k, _, top, entropy = diversity(get(field)[selection])
            values.extend([float(selection.mean()), k, top, entropy])
        values.extend(diversity(get("destination_id"), n))
        for field in ("duration_in", "duration_out"):
            v = get(field)
            values.extend([float(v.mean()), float(v.max())])
        values.extend([float(get("min_ip_pkt_len").min()), float(get("min_ip_pkt_len").mean()),
                       float(get("max_ip_pkt_len").mean()), float(get("max_ip_pkt_len").max())])
        values.extend(sums(field) for field in RETRANSMISSION)
        bins = np.array([sums(field) for field in PACKET_BINS])
        values.extend((bins/bins.sum()).tolist() if bins.sum() else [0.]*5)
        for field in ("tcp_win_max_in", "tcp_win_max_out"):
            v = get(field)[tcp]
            values.extend([float(v.mean()), float(v.max())] if len(v) else [0., 0.])
        ordered = np.sort(get("flow_start_milliseconds"))
        gaps = np.diff(ordered)
        values.extend([float(gaps.mean()), float(gaps.max()), float(ordered[-1]-ordered[0])] if len(gaps) else [0., 0., 0.])
        if len(values) != 82:
            raise AssertionError(f"Expected 82 fixed features, got {len(values)}")
        x[bucket, :82] = values
        codes, counts = np.unique(l7, return_counts=True)
        for code, count in zip(codes, counts):
            x[bucket, vocabulary_index.get(int(code), schema["feature_count"]-1)] += count/n
        mask[bucket] = 1
        attacks[bucket] = int(get("label").sum())
        flow_counts[bucket] = n
        y[bucket] = attacks[bucket] > 0
    if not np.isfinite(x).all() or np.any(x < -1e-12):
        raise AssertionError("Invalid unscaled features")
    ratio_cols = [i for i, name in enumerate(schema["feature_names"]) if name.endswith(("_ratio", "_entropy"))]
    if np.any(x[:, ratio_cols] > 1+1e-10):
        raise AssertionError("Ratio/entropy exceeds one")
    if not np.allclose(x[:, 82:].sum(axis=1), mask, atol=1e-10):
        raise AssertionError("L7 ratio/mask reconciliation failed")
    return x, mask, y, flow_counts, attacks


def local_events(events, window_start):
    copied = events.copy()
    copied[:, E["flow_start_milliseconds"]] -= window_start
    copied[:, E["flow_end_milliseconds"]] -= window_start
    return copied


def shift_bounds(events):
    v = events[:, E["flow_start_milliseconds"]]
    return -int(v.min()), WINDOW_MS-1-int(v.max())


def superimpose(left, right, shift):
    lo, hi = shift_bounds(right)
    if not lo <= shift <= hi:
        raise ValueError("Shift would discard a flow start")
    shifted = right.copy()
    shifted[:, E["flow_start_milliseconds"]] += shift
    shifted[:, E["flow_end_milliseconds"]] += shift
    merged = np.concatenate([left, shifted])
    return merged[np.argsort(merged[:, E["flow_start_milliseconds"]], kind="stable")]


class Moments:
    def __init__(self, width):
        self.n = 0
        self.mean = np.zeros(width, dtype=np.float64)
        self.m2 = np.zeros(width, dtype=np.float64)

    def add(self, rows):
        if not len(rows):
            return
        n, mu = len(rows), rows.mean(axis=0)
        delta = mu-self.mean
        total = self.n+n
        self.m2 += np.square(rows-mu).sum(axis=0) + delta*delta*self.n*n/total
        self.mean += delta*n/total
        self.n = total

    def finish(self):
        if not self.n:
            raise ValueError("No original training active buckets for scaler fitting")
        std = np.sqrt(self.m2/self.n)
        constant = std < 1e-12
        std[constant] = 1.
        return {"n": self.n, "mean": self.mean.tolist(), "scale": std.tolist(), "constant": constant.tolist()}


def transform_log(x, schema):
    copied = x.astype(np.float64, copy=True)
    indices = schema["log1p_columns"]
    if np.any(copied[:, indices] < 0):
        raise ValueError("Negative value passed to log1p")
    copied[:, indices] = np.log1p(copied[:, indices])
    return copied


def normalize(x, mask, schema, scaler):
    shape = x.shape
    flat = transform_log(x.reshape(-1, shape[-1]), schema)
    flat = (flat-np.asarray(scaler["mean"]))/np.asarray(scaler["scale"])
    flat[mask.reshape(-1) == 0] = 0
    result = flat.reshape(shape).astype(np.float32)
    if not np.isfinite(result).all():
        raise AssertionError("Nonfinite normalized feature")
    return result
