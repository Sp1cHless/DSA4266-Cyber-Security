"""Deterministic global time-block split, coverage reservations and temporal purge.

Label metadata designs the split; no fitted features or model scores are used.
This is a same-environment, held-out-time-block evaluation, not a claim that
attack campaigns or source IPs are independent between folds.
"""
from __future__ import annotations

import numpy as np

WINDOW_MS = 100_000
BLOCK_MS = 900_000
EMBARGO_MS = 100_000
NAMES = ("train", "validation", "test", "excluded")
POLICY = {"method": "global_15m_coverage_v2", "block_seconds": 900,
          "embargo_seconds": 100, "target_ratios": [.6, .2, .2],
          "coverage": "train all types; greedy feasible validation/test coverage; complete blocks only",
          "balance": "original-count allocation plus three retained-ratio corrections; choose closest retained proportions, no model scores",
          "evaluation": "same environment, held-out global time blocks; campaigns may recur"}


def interval_overlap(first, max_end, start, codes):
    """Conservative closed flow envelope vs other folds' half-open 100s windows."""
    result = np.zeros(len(start), bool)
    for code in range(3):
        members = np.flatnonzero(codes == code)
        other = np.unique(start[(codes < 3) & (codes != code)])
        if not len(other) or not len(members):
            continue
        j = np.searchsorted(other, first[members]-WINDOW_MS, side="right")
        valid = j < len(other)
        result[members[valid]] = other[j[valid]] <= max_end[members[valid]]
    return result


def exclude_boundaries(start, first, max_end, codes, embargo_ms=EMBARGO_MS):
    times, inverse = np.unique(start, return_inverse=True)
    global_codes = np.empty(len(times), np.uint8)
    global_codes[inverse] = codes
    if not np.array_equal(global_codes[inverse], codes):
        raise ValueError("One global 100s timestamp was assigned to multiple folds")
    boundaries = times[1:][global_codes[1:] != global_codes[:-1]]
    support_end = np.maximum(start+WINDOW_MS, max_end+1)
    near = np.zeros(len(start), bool)
    if embargo_ms and len(boundaries):
        j = np.searchsorted(boundaries, start-embargo_ms, side="right")
        valid = j < len(boundaries)
        near[valid] = boundaries[j[valid]]-embargo_ms < support_end[valid]
    crossing = interval_overlap(first, max_end, start, codes)
    reasons = near.astype(np.uint8) | (crossing.astype(np.uint8) << 1)
    retained = codes.copy()
    retained[reasons != 0] = 3
    return retained, reasons, boundaries


def split(start, first, max_end, source, attack_flows, cells, seed=4266, primary=None, holdout_types=()):
    """Return kept/original fold codes, exclusion bits, groups and audit metadata.

    cells maps (zero-based sequence array index, attack name) to exact type count.
    Reservations protect a whole-block footprint around one actual sequence.
    The footprint also includes the preceding observed time group, so a first
    observed window after a capture gap is not accidentally purged on entry.
    Greedy reservations are feasible coverage, not a global optimality claim.

    holdout_types lists attack labels removed from the training fold only, by
    primary sequence label. Fold assignment, coverage reservations, the seeded
    RNG and the temporal purge run exactly as without a holdout, so validation
    and test populations are unchanged; only training membership is reduced.
    """
    n = len(start)
    if not n or not all(len(a) == n for a in (first, max_end, source, attack_flows)):
        raise ValueError("Split input arrays are empty or have inconsistent lengths")
    holdout_types = tuple(sorted(set(holdout_types)))
    if primary is not None:
        primary = np.asarray(primary)
        if len(primary) != n:
            raise ValueError("Primary attack labels have a different length")
    elif holdout_types:
        raise ValueError("Training holdout requires one primary attack label per sequence")
    blocks, group_index, weights = np.unique(start//BLOCK_MS, return_inverse=True, return_counts=True)
    if len(blocks) < 3:
        raise ValueError("Need at least three observed global 15min blocks for train/validation/test")
    times = np.unique(start)
    time_groups = np.searchsorted(blocks, times//BLOCK_MS)
    rng = np.random.default_rng(seed)
    random_order = rng.permutation(len(blocks))
    ranks = np.empty(len(blocks), np.int64)
    ranks[random_order] = np.arange(len(blocks))
    # Initial seeded mass split supplies a label-independent preference order.
    cumulative = np.cumsum(weights[random_order])
    cut1 = int(np.argmin(np.abs(cumulative[:-2]-.6*n)))+1
    cut2 = int(np.argmin(np.abs(cumulative[cut1:-1]-.8*n)))+cut1+1
    assigned = np.empty(len(blocks), np.uint8)
    assigned[random_order[:cut1]], assigned[random_order[cut1:cut2]], assigned[random_order[cut2:]] = 0, 1, 2
    initial, _, _ = exclude_boundaries(start, first, max_end, assigned[group_index])
    reserved = np.full(len(blocks), -1, np.int8)
    by_type = {}
    for (index, name), count in cells.items():
        by_type.setdefault(name, []).append((int(index), int(count)))
    if not by_type:
        raise ValueError("No malicious attack types in the original dataset")
    footprints = {}
    def footprint(index):
        if index not in footprints:
            lo = max(0, int(np.searchsorted(times, start[index]-EMBARGO_MS))-1)
            end = max(int(start[index])+WINDOW_MS, int(max_end[index])+1)
            hi = int(np.searchsorted(times, end+EMBARGO_MS))
            footprints[index] = np.unique(time_groups[lo:hi])
        return footprints[index]

    reservations = []
    def reserve(entries, code, name):
        choices = []
        for index, count in entries:
            index = int(index)
            groups = footprint(index)
            if np.any((reserved[groups] >= 0) & (reserved[groups] != code)):
                continue
            # Prefer an already retained member; then substantial attack support.
            # Remaining ties use seeded group rank and stable catalog index.
            score = (initial[index] != code, -count,
                     int(weights[groups[assigned[groups] != code]].sum()),
                     int(ranks[group_index[index]]), index)
            choices.append((score, index, groups))
        if not choices:
            return False
        _, index, groups = min(choices, key=lambda c:c[0])
        reserved[groups], assigned[groups] = code, code
        reservations.append({"purpose": name, "split": NAMES[code], "sequence_index": index,
                             "block_starts_ms": (blocks[groups]*BLOCK_MS).tolist()})
        return True

    # Rare types receive first access to indivisible support groups.
    ordered_types = sorted(by_type, key=lambda a:(len({int(group_index[i]) for i,_ in by_type[a]}), a))
    for name in ordered_types:
        if not reserve(by_type[name], 0, name):
            raise ValueError(f"Cannot preserve training coverage for {name}")
    # Event superposition requires at least two distinct pure-benign sources.
    backgrounds = np.flatnonzero(attack_flows == 0)
    if len(np.unique(source[backgrounds])) < 2:
        raise ValueError("Need at least two original pure-benign sources")
    picked_sources = set()
    background_order = np.lexsort((backgrounds, ranks[group_index[backgrounds]], initial[backgrounds] != 0))
    for index in backgrounds[background_order]:
        if int(source[index]) in picked_sources:
            continue
        if reserve([(index, 1)], 0, "pure-benign donor source"):
            picked_sources.add(int(source[index]))
        if len(picked_sources) == 2:
            break
    missing_reservations = []
    for k, name in enumerate(ordered_types):
        # Alternate the first held-out fold to avoid always favoring validation.
        for code in ((1, 2) if k % 2 == 0 else (2, 1)):
            if not reserve(by_type[name], code, name):
                missing_reservations.append({"attack_type": name, "split": NAMES[code],
                                             "reason": "No witness footprint fits existing coverage reservations"})

    # Balance original sequence counts around fixed coverage reservations.
    free = sorted(np.flatnonzero(reserved < 0), key=lambda g:(-int(weights[g]), int(ranks[g])))
    def allocate(target):
        totals = np.array([weights[reserved == code].sum() for code in range(3)], np.int64)
        assigned[reserved >= 0] = reserved[reserved >= 0]
        for group in free:
            code = int(np.argmin(2*(totals-target)*weights[group]+weights[group]**2))
            assigned[group] = code
            totals[code] += weights[group]
        original = assigned[group_index]
        kept, reasons, boundaries = exclude_boundaries(start, first, max_end, original)
        return original, kept, reasons, boundaries
    ratios = np.array([.6, .2, .2])
    target = ratios*n
    best, balance = None, []
    for attempt in range(4):
        original, kept, reasons, boundaries = allocate(target)
        retained_counts = np.bincount(kept, minlength=4)[:3]
        score = float(np.sum((retained_counts/max(1, retained_counts.sum())-ratios)**2))
        balance.append({"pass": attempt, "retained_sequences": retained_counts.tolist(), "squared_ratio_error": score})
        if best is None or score < best[0]:
            best = (score, original.copy(), kept.copy(), reasons.copy(), boundaries.copy(), assigned.copy())
        rates = retained_counts/np.maximum(1, np.bincount(original, minlength=3))
        corrected = ratios/np.maximum(rates, .01)
        target = corrected/corrected.sum()*n
    _, original, kept, reasons, boundaries, assigned = best
    held = np.zeros(n, bool)
    removed_total = 0
    train_holdout = []
    if holdout_types:
        held = np.isin(primary, holdout_types)
        missing_types = sorted(set(holdout_types)-set(np.unique(primary).tolist()))
        if missing_types:
            raise ValueError(f"Holdout attack types absent from the source: {missing_types}")
        for name in holdout_types:
            indices = np.array([i for i,_ in by_type.get(name, [])], np.int64)
            counts = np.array([c for _,c in by_type.get(name, [])], np.int64)
            primary_here = held[indices]
            dropped = (kept[indices] == 0) & primary_here
            train_holdout.append({"attack_type": name,
                "train_sequences_removed": int(np.count_nonzero(dropped)),
                "train_attack_flows_removed": int(counts[dropped].sum()),
                "removed_sources": int(len(np.unique(source[indices[dropped]]))),
                "validation_sequences_primary": int(np.count_nonzero(primary_here & (kept[indices] == 1))),
                "test_sequences_primary": int(np.count_nonzero(primary_here & (kept[indices] == 2))),
                "validation_sequences_containing_type": int(np.count_nonzero(kept[indices] == 1)),
                "test_sequences_containing_type": int(np.count_nonzero(kept[indices] == 2)),
                "policy": "primary attack label removed from the training fold only; validation/test retained"})
        removed_total = int(np.count_nonzero(held & (kept == 0)))
        kept[held & (kept == 0)] = 3
    for witness in reservations:
        if witness["purpose"] in holdout_types and witness["split"] == "train":
            continue
        if kept[witness["sequence_index"]] != NAMES.index(witness["split"]):
            raise AssertionError("Coverage witness did not survive temporal exclusions")
    coverage = []
    for name in sorted(by_type):
        indices = np.array([i for i,_ in by_type[name]], np.int64)
        counts = np.array([c for _,c in by_type[name]], np.int64)
        row = {"attack_type": name, "original_blocks": len(np.unique(group_index[indices]))}
        for code, split_name in enumerate(NAMES):
            row[split_name+"_sequences"] = int(np.count_nonzero(kept[indices] == code))
            row[split_name+"_attack_flows"] = int(counts[kept[indices] == code].sum())
        row["held_out_from_training"] = name in holdout_types
        row["validation_unseen_in_train"] = row["validation_sequences"] > 0 and not row["train_sequences"]
        row["test_unseen_in_train"] = row["test_sequences"] > 0 and not row["train_sequences"]
        if not row["train_sequences"] and name not in holdout_types:
            raise AssertionError(f"Training type missing after exclusions: {name}")
        coverage.append(row)
    if any(not np.any(kept == code) for code in range(3)):
        raise ValueError("Coverage reservations and temporal purge leave an empty split; need more independent time blocks")
    overlap = int(interval_overlap(first, max_end, start, kept).sum())
    if overlap:
        raise AssertionError("Retained flow envelopes overlap another fold's 100s windows")
    # Check source adjacency in source/time order, independently of input ordering.
    order = np.lexsort((start, source))
    adjacent = (source[order[1:]] == source[order[:-1]]) & (np.diff(start[order]) == WINDOW_MS)
    crossing = adjacent & (kept[order[1:]] < 3) & (kept[order[:-1]] < 3) & (kept[order[1:]] != kept[order[:-1]])
    notes = {**POLICY, "seed": seed, "original_sequences": n, "observed_blocks": len(blocks),
             "assigned_blocks": {NAMES[c]: int(np.count_nonzero(assigned == c)) for c in range(3)},
             "sequences_before_purge": {NAMES[c]: int(np.count_nonzero(original == c)) for c in range(3)},
             "retained_sequences": {NAMES[c]: int(np.count_nonzero(kept == c)) for c in range(3)},
             "retained_positive_sequences": {NAMES[c]: int(np.count_nonzero((kept == c) & (attack_flows > 0))) for c in range(3)},
             "excluded_sequences": int(np.count_nonzero(kept == 3)),
             "excluded_positive_sequences": int(np.count_nonzero((kept == 3) & (attack_flows > 0))),
             "embargo_exclusions": int(np.count_nonzero(reasons & 1)), "interval_exclusions": int(np.count_nonzero(reasons & 2)),
             "fold_transitions": len(boundaries), "shared_global_100s": 0,
             "retained_balance_attempts": balance,
             "retained_interval_overlap_sequences": overlap, "adjacent_source_pairs_cross_split": int(crossing.sum()),
             "coverage": coverage, "unreserved_coverage_requests": missing_reservations,
             "coverage_reservations": reservations,
             "train_holdout_types": list(holdout_types), "train_holdout": train_holdout,
             "train_holdout_sequences_removed": removed_total,
             "coverage_policy": ("train all types except the declared training holdout" if holdout_types
                                 else POLICY["coverage"]),
             "train_holdout_scope": ("Fold assignment, reservations and purge are identical to a run without a holdout; "
                                     "only training membership of the declared primary labels is removed, so validation/test "
                                     "populations are unchanged. Removed training sequences count as excluded, not as temporal purge."),
             "duplicate_record_scope": "Identical complete raw records have the same source IP/start and therefore the same original window. No within-window duplicate removal or session/campaign independence claim.",
             "ratios_denominator": "original retained sequences, before augmentation; 60/20/20 is approximate after purge"}
    if np.any(crossing):
        raise AssertionError("Adjacent retained source windows cross sets")
    return kept, original, reasons, blocks*BLOCK_MS, assigned, reserved, notes
