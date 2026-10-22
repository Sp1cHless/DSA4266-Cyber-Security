"""Exact zero-shift bucket composition; no neural-network dependency.

Frequency maps and flow-start offsets are retained alongside the 82 historical
statistics. Numerical features are recomputed using their original definitions.
"""
from __future__ import annotations

import numpy as np

from CIC_new.preprocessing import bucket_statistics as bf
from CIC_new.preprocessing.run_preprocessing import service_ids

FIXED = ("protocol", "service", "icmp_type", "dns_type")
LEARNED = ("l7", "src_port", "dst_port")
FIELDS = FIXED + LEARNED
HISTOGRAMS = FIELDS + ("src_port_all", "dst_port_all", "destination")
DENSE = ("numeric", "mask", "y_bucket", "flow_counts", "attack_flow_counts", "packet_bins")
NAMES = {name: i for i, name in enumerate(bf.NUMERIC_NAMES)}
NUMERIC_SCHEMA = {
    "feature_names": bf.NUMERIC_NAMES,
    "log1p_columns": [i for i, name in enumerate(bf.NUMERIC_NAMES)
                      if not name.endswith(("_ratio", "_entropy"))],
}


def histogram(bucket, codes):
    if not len(codes):
        return np.empty((0, 3), np.int64)
    pairs, counts = np.unique(np.column_stack((bucket, codes)), axis=0, return_counts=True)
    return np.column_stack((pairs, counts)).astype(np.int64)


def raw_bucket(events):
    numeric, mask, labels, counts, attacks = bf.aggregate(events, bf.make_schema([]))
    get = lambda name: events[:, bf.E[name]]
    if np.any(get("l4_src_port") > 65535) or np.any(get("l4_dst_port") > 65535):
        raise ValueError("Port outside 0..65535")
    bucket = get("flow_start_milliseconds") // bf.BUCKET_MS
    protocol = get("protocol")
    ports = np.isin(protocol, (6, 17))
    icmp = protocol == 1
    dns = (get("dns_query_id") != 0) | (get("dns_query_type") != 0) | (get("dns_ttl_answer") != 0)
    selections = {
        "protocol": (np.ones(len(events), bool), protocol),
        "l7": (np.ones(len(events), bool), get("l7_proto")),
        "service": (ports, service_ids(get("l4_dst_port"))),
        "src_port": (ports, get("l4_src_port")),
        "dst_port": (ports, get("l4_dst_port")),
        "src_port_all": (np.ones(len(events), bool), get("l4_src_port")),
        "dst_port_all": (np.ones(len(events), bool), get("l4_dst_port")),
        "icmp_type": (icmp, get("icmp_ipv4_type")),
        "dns_type": (dns, get("dns_query_type")),
        "destination": (np.ones(len(events), bool), get("destination_id")),
    }
    result = dict(numeric=numeric[:, :82], mask=mask, y_bucket=labels,
                  flow_counts=counts, attack_flow_counts=attacks,
                  starts=np.column_stack((bucket, get("flow_start_milliseconds") % bf.BUCKET_MS)),
                  packet_bins=np.zeros((10, 5), np.float64))
    for field, (eligible, codes) in selections.items():
        result["hist_" + field] = histogram(bucket[eligible], codes[eligible])
    for j, name in enumerate(bf.PACKET_BINS):
        np.add.at(result["packet_bins"][:, j], bucket, get(name).astype(np.float64))
    return result


def merge_histograms(a, b):
    joined = np.concatenate((a, b))
    if not len(joined):
        return np.empty((0, 3), np.int64)
    pairs, inverse = np.unique(joined[:, :2], axis=0, return_inverse=True)
    counts = np.zeros(len(pairs), np.int64)
    np.add.at(counts, inverse, joined[:, 2])
    return np.column_stack((pairs, counts))


def category_counts(hist, code=None):
    chosen = hist if code is None else hist[hist[:, 1] == code]
    out = np.zeros(10, np.int64)
    np.add.at(out, chosen[:, 0], chosen[:, 2])
    return out


def merge_buckets(a, b):
    """Compose prepared bucket states, preserving every original feature meaning."""
    result = {"hist_" + field: merge_histograms(a["hist_" + field], b["hist_" + field])
              for field in HISTOGRAMS}
    result["starts"] = np.concatenate((a["starts"], b["starts"]))
    result["packet_bins"] = a["packet_bins"] + b["packet_bins"]
    result["flow_counts"] = a["flow_counts"] + b["flow_counts"]
    result["attack_flow_counts"] = a["attack_flow_counts"] + b["attack_flow_counts"]
    result["mask"] = (result["flow_counts"] > 0).astype(np.uint8)
    result["y_bucket"] = (result["attack_flow_counts"] > 0).astype(np.uint8)
    n_a, n_b = a["flow_counts"], b["flow_counts"]
    n = result["flow_counts"]
    tcp_a = category_counts(a["hist_protocol"], 6)
    tcp_b = category_counts(b["hist_protocol"], 6)
    x = a["numeric"] + b["numeric"]
    for name, column in NAMES.items():
        if name.endswith("_mean") or name.endswith("_ratio"):
            weights_a, weights_b = (tcp_a, tcp_b) if (
                "tcp_flag_" in name or name.startswith("tcp_win_")) else (n_a, n_b)
            numerator = a["numeric"][:, column] * weights_a + b["numeric"][:, column] * weights_b
            x[:, column] = np.divide(numerator, weights_a + weights_b,
                                     out=np.zeros(10), where=(weights_a + weights_b) > 0)
        elif name.endswith("_min"):
            x[:, column] = np.minimum(np.where(n_a > 0, a["numeric"][:, column], np.inf),
                                      np.where(n_b > 0, b["numeric"][:, column], np.inf))
        elif name.endswith("_max"):
            x[:, column] = np.maximum(a["numeric"][:, column], b["numeric"][:, column])
    for field, prefix in (("l7", "l7"), ("src_port_all", "src_port"),
                          ("dst_port_all", "dst_port"), ("destination", "dst_ip"),
                          ("icmp_type", "icmp_type"), ("dns_type", "dns_query_type")):
        hist = result["hist_" + field]
        for t in range(10):
            freq = hist[hist[:, 0] == t, 2]
            k, total = len(freq), int(freq.sum())
            probabilities = freq / total if total else np.empty(0)
            denominator = total if field in ("icmp_type", "dns_type") else n[t]
            values = {
                "nunique": k,
                "reuse_ratio": 1 - k / denominator if denominator else 0.,
                "top1_ratio": float(freq.max() / denominator) if denominator else 0.,
                "entropy": float(-(probabilities * np.log(probabilities)).sum() / np.log(k)) if k > 1 else 0.,
            }
            for suffix, value in values.items():
                feature = prefix + "_" + suffix
                if field == "l7" and suffix == "nunique":
                    feature = "l7_proto_nunique"
                if feature in NAMES:
                    x[t, NAMES[feature]] = value
    for t in range(10):
        x[t, NAMES["protocol_nunique"]] = np.count_nonzero(result["hist_protocol"][:, 0] == t)
        bins = result["packet_bins"][t]
        for j, name in enumerate(bf.PACKET_BINS):
            x[t, NAMES[name + "_ratio"]] = bins[j] / bins.sum() if bins.sum() else 0.
        starts = np.sort(result["starts"][result["starts"][:, 0] == t, 1])
        gaps = np.diff(starts)
        timing = (gaps.mean(), gaps.max(), starts[-1] - starts[0]) if len(gaps) else (0., 0., 0.)
        for name, value in zip(("flow_start_gap_mean_ms", "flow_start_gap_max_ms", "flow_start_span_ms"), timing):
            x[t, NAMES[name]] = value
    x[n == 0] = 0
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite composed statistics")
    result["numeric"] = x
    return result


def category_indices(codes, vocabulary):
    """Slot zero is UNK, including for genuinely unseen port numbers."""
    vocabulary = np.asarray(vocabulary, np.int64)
    if not len(vocabulary):
        return np.zeros(len(codes), np.int64)
    positions = np.searchsorted(vocabulary, codes)
    inside = positions < len(vocabulary)
    known = inside & (vocabulary[np.minimum(positions, len(vocabulary) - 1)] == codes)
    return np.where(known, positions + 1, 0)


def project_histogram(coo, rows, vocabulary, table=None):
    """Mean of per-flow category vectors; supports flattened multi-sequence COO."""
    width = len(vocabulary) + 1 if table is None else table.shape[1]
    values = np.zeros((rows, width), np.float64)
    denominator = np.zeros(rows, np.float64)
    indices = category_indices(coo[:, 1], vocabulary)
    np.add.at(denominator, coo[:, 0], coo[:, 2])
    if table is None:
        np.add.at(values, (coo[:, 0], indices), coo[:, 2])
    else:
        # Bound temporary allocation for large, high-cardinality port histograms.
        for start in range(0, len(coo), 65536):
            part = coo[start:start + 65536]
            np.add.at(values, part[:, 0], table[indices[start:start + len(part)]] * part[:, 2, None])
    np.divide(values, denominator[:, None], out=values, where=denominator[:, None] > 0)
    return values


def categorical_features(state, schema, tables):
    rows = state["mask"].size
    blocks = [project_histogram(state["hist_" + field], rows, schema["vocabularies"][field],
                                tables.get(field)) for field in FIELDS]
    return np.concatenate(blocks, axis=1).reshape(*state["mask"].shape, -1)


def model_features(state, schema, tables, scaler):
    numeric = bf.transform_log(state["numeric"].reshape(-1, 82), NUMERIC_SCHEMA)
    categorical = categorical_features(state, schema, tables).reshape(len(numeric), -1)
    x = np.concatenate((numeric, categorical), axis=1)
    x = ((x - np.asarray(scaler["mean"])) / np.asarray(scaler["scale"])).astype(np.float32)
    x[state["mask"].reshape(-1) == 0] = 0
    return x.reshape(*state["mask"].shape, -1)
