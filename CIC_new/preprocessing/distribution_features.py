"""A fixed 15-feature distribution/joint-statistics bundle for Scheme2 ablation."""
import numpy as np
from . import bucket_statistics as bf

NAMES = [f"{field}_std" for field in ("in_bytes", "out_bytes", "in_pkts", "out_pkts", "flow_duration_milliseconds")]
NAMES += [f"{field}_q90" for field in ("in_bytes", "out_bytes", "flow_duration_milliseconds")]
NAMES += ["zero_reply_fraction", "in_byte_share_mean", "in_byte_share_std", "bytes_per_packet_mean",
          "bytes_per_packet_std", "destination_port_pair_unique_fraction", "flow_duration_milliseconds_median"]
LOG_COLUMNS = [i for i, name in enumerate(NAMES) if name not in
               ("zero_reply_fraction", "in_byte_share_mean", "in_byte_share_std", "destination_port_pair_unique_fraction")]
BUNDLE = "distribution_joint_v1"


def aggregate(events):
    if events.ndim != 2 or events.shape[1] != len(bf.EVENT_COLUMNS) or not len(events):
        raise ValueError("Expected nonempty staged flow records")
    starts = events[:, bf.E["flow_start_milliseconds"]]
    if starts.min() < 0 or starts.max() >= 100000:
        raise ValueError("The fixed bundle requires original 100-second windows")
    result = np.zeros((10, len(NAMES)), np.float64)
    for bucket in np.unique(starts // 10000):
        rows = events[starts // 10000 == bucket]
        get = lambda name: rows[:, bf.E[name]].astype(np.float64)
        values = [get(field).std() for field in ("in_bytes", "out_bytes", "in_pkts", "out_pkts", "flow_duration_milliseconds")]
        values += [np.quantile(get(field), .9) for field in ("in_bytes", "out_bytes", "flow_duration_milliseconds")]
        total_bytes = get("in_bytes") + get("out_bytes")
        shares = get("in_bytes") / np.maximum(total_bytes, 1)
        bytes_per_packet = total_bytes / np.maximum(get("in_pkts") + get("out_pkts"), 1)
        pairs = np.unique(rows[:, [bf.E["destination_id"], bf.E["l4_dst_port"]]], axis=0)
        values += [float((get("out_pkts") == 0).mean()), shares.mean(), shares.std(), bytes_per_packet.mean(),
                   bytes_per_packet.std(), len(pairs) / len(rows), np.median(get("flow_duration_milliseconds"))]
        result[int(bucket)] = values
    if not np.isfinite(result).all() or (result < 0).any():
        raise AssertionError("Invalid distribution features")
    return result


def logged(values):
    result = values.copy()
    result[..., LOG_COLUMNS] = np.log1p(result[..., LOG_COLUMNS])
    return result
