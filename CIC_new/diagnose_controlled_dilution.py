"""Train-derived controlled flow-dilution stress test of frozen pipelines.

Donor originals were available during fitting. New pairs/subsets are not an
unseen-donor holdout. No parameter fitting, test access or threshold tuning.
"""
import argparse
from collections import defaultdict
from datetime import datetime
from fractions import Fraction
from pathlib import Path
import sqlite3
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from CIC_new.diagnose_superimposition import load_models, write_csv, percent
from CIC_new.evaluate_autoencoder import read, write, digest
from CIC_new.modelling.autoencoder import TemporalAE
from CIC_new.preprocessing import bucket_statistics as bf
from CIC_new.preprocessing.dataset import ModelInputs, transform_model_inputs
from CIC_new.preprocessing.run_preprocessing import raw_bucket as v3_bucket, selected_distribution, NUMERIC_SCHEMA
from CIC_new.preprocessing.sequence_index import signature
from CIC_new.representation_learning.bucket_autoencoder import BucketAE, FIELDS
from CIC_new.run_scheme1_l2_experiment import latest_l2_run
from CIC_new.scheme1.statistics import raw_bucket, model_features, merge_buckets

RATIOS = (1, 5, 10, 25, 50, 100)
ROUTES = ("scheme1_l2_physical", "scheme2_physical", "scheme2_vector_add")
THRESHOLDS = ("train_benign_99pct", "train_benign_99.9pct")


def host_balanced(rows, limit, rng):
    groups = defaultdict(list)
    for row in rows:
        groups[row[1]].append(row)
    hosts = list(groups)
    rng.shuffle(hosts)
    for items in groups.values():
        rng.shuffle(items)
    result = []
    while hosts and len(result) < limit:
        remaining = []
        for host in hosts:
            result.append(groups[host].pop())
            if groups[host]:
                remaining.append(host)
            if len(result) == limit:
                break
        hosts = remaining
    return result


def controlled_subsets(attack, background, replacement, rng):
    """Keep attack rows fixed; nested benign prefixes vary exact flow ratios."""
    if not len(attack) or not np.all(attack[:, bf.E["label"]] == 1):
        raise ValueError("Pure attack rows required")
    if not np.all(background[:, bf.E["label"]] == 0) or not np.all(replacement[:, bf.E["label"]] == 0):
        raise ValueError("Benign rows required")
    n = len(attack)
    if len(background) < n*99 or len(replacement) < n:
        raise ValueError("Insufficient real rows; sampling with replacement is forbidden")
    background = background[rng.permutation(len(background))[:n*99]]
    replacement = replacement[rng.permutation(len(replacement))[:n]]
    for ratio in RATIOS:
        total = n*100//ratio
        b = background[:total-n]
        if Fraction(n, total) != Fraction(ratio, 100):
            raise AssertionError("Inexact dilution ratio")
        yield ratio, b, replacement


class EventReader:
    def __init__(self, prepared):
        prepared = Path(prepared)
        self.manifest = sqlite3.connect((prepared/"manifest.sqlite").resolve().as_uri()+"?mode=ro", uri=True)
        self.work = sqlite3.connect((prepared/"work.sqlite").resolve().as_uri()+"?mode=ro", uri=True)
        self.source = read(prepared/"manifest.json")["source"]
        self.raw = Path(self.source["path"])
        if signature(self.raw) != self.source:
            raise ValueError("Raw database signature changed")
        self.work.execute("ATTACH DATABASE ? AS raw", (self.raw.resolve().as_uri()+"?mode=ro",))

    def events(self, sid, attack=None):
        meta = self.manifest.execute("SELECT source_ip,window_start_ms,split,attack_flows FROM original_sequences WHERE sequence_id=?", (int(sid),)).fetchone()
        if meta is None or meta[2] != "train":
            raise ValueError("Only retained real training donors are allowed")
        columns = ",".join("e."+name for name in bf.EVENT_COLUMNS)
        query = "SELECT "+columns+" FROM events e INDEXED BY events_sequence"
        params = [int(sid)]
        if attack is not None:
            query += " JOIN raw.network_flows r ON r.rowid=e.raw_rowid"
        query += " WHERE e.sequence_id=?"
        if attack is not None:
            query += " AND e.label=1 AND r.attack=?"
            params.append(attack)
        query += " ORDER BY e.flow_start_milliseconds,e.raw_rowid"
        rows = self.work.execute(query, params).fetchall()
        if not rows:
            raise ValueError("No requested donor rows")
        return bf.local_events(np.asarray(rows, np.int64), meta[1]), meta

    def close(self):
        self.work.close()
        self.manifest.close()


class Representations:
    def __init__(self, models):
        self.models = models
        self.one = Path(models["scheme1_l2"]["config"]["dataset"])
        self.two = Path(models["scheme2"]["config"]["dataset"])
        self.schema1, self.scaler1 = read(self.one/"schema.json"), read(self.one/"scaler.json")
        self.schema2, self.scaler2 = read(self.two/"schema.json"), read(self.two/"scaler.json")
        with np.load(self.one/"embeddings.npz", allow_pickle=False) as a:
            self.tables = {key: a[key] for key in a.files}
        saved = torch.load(self.two/"frozen_bucket_encoder.pt", map_location="cpu", weights_only=True)
        self.bucket = BucketAE(**saved["model_args"])
        self.bucket.load_state_dict(saved["model_state"])
        self.bucket.eval().requires_grad_(False)
        self.snapshots = {str(folder/name): digest(folder/name) for folder, names in (
            (self.one, ("manifest.json", "schema.json", "scaler.json", "embeddings.npz")),
            (self.two, ("manifest.json", "schema.json", "scaler.json", "frozen_bucket_encoder.pt"))) for name in names}

    def volume(self, counts, mask):
        return ((np.log1p(counts.astype(np.float32))-self.scaler2["mean"][0])/self.scaler2["scale"][0])*mask

    def scheme1(self, events):
        state = raw_bucket(events)
        x = model_features(state, self.schema1, self.tables, self.scaler1)
        return transform_model_inputs(x, state["mask"], "bucket_l2"), state

    @torch.no_grad()
    def scheme2(self, events):
        if not len(events):
            state = dict(mask=np.zeros(10, np.uint8), flow_counts=np.zeros(10, np.int64),
                         attack_flow_counts=np.zeros(10, np.int64))
            width = self.bucket.encoder[-1].out_features+1
            return np.zeros((10,width), np.float32), state
        state, coo = v3_bucket(events)
        numeric = bf.normalize(state["numeric"], state["mask"], NUMERIC_SCHEMA, self.scaler2)
        hist = {field: torch.from_numpy(selected_distribution(coo[field], np.array([0]), self.schema2["vocabularies"][field], 1)) for field in FIELDS}
        z = self.bucket.encode(torch.from_numpy(numeric[None].astype(np.float32)), hist)[0].numpy()
        z *= state["mask"][:, None]
        x = np.column_stack((z, self.volume(state["flow_counts"], state["mask"]))).astype(np.float32)
        return x, state

    def vector_add(self, left, left_state, right, right_state):
        mask = np.maximum(left_state["mask"], right_state["mask"])
        z = left[:, :-1]+right[:, :-1]
        z /= np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-8)
        z *= mask[:, None]
        flows = left_state["flow_counts"]+right_state["flow_counts"]
        return np.column_stack((z, self.volume(flows, mask))).astype(np.float32)

    def assert_unchanged(self):
        if any(digest(Path(path)) != expected for path, expected in self.snapshots.items()):
            raise ValueError("Input representation artifacts changed")


def select_plan(reader, pair_limit, seed, original_plan):
    rng = np.random.default_rng(seed)
    attacks = reader.manifest.execute("SELECT DISTINCT a.attack FROM sequence_attacks a JOIN original_sequences s USING(sequence_id) WHERE s.split='train' ORDER BY a.attack").fetchall()
    # A maximum attack budget of five preserves low-flow types such as XSS.
    # Background pools with >=100 flows also support the one-flow budget.
    backgrounds = reader.manifest.execute("SELECT sequence_id,source_ip,flows FROM original_sequences WHERE split='train' AND attack_flows=0 AND flows>=100 ORDER BY sequence_id").fetchall()
    with np.load(original_plan, allow_pickle=False) as a:
        existing = set(zip(map(int, a["left_id"]), map(int, a["right_id"])))
    plan = []
    for (attack,) in attacks:
        rows = reader.manifest.execute("SELECT s.sequence_id,s.source_ip,a.flows FROM sequence_attacks a JOIN original_sequences s USING(sequence_id) WHERE s.split='train' AND a.attack=? ORDER BY s.sequence_id", (attack,)).fetchall()
        for sid, host, count in host_balanced(rows, pair_limit, rng):
            budget = min(5, count)
            eligible = [r for r in backgrounds if r[2] >= 100*budget and r[1] != host and (r[0], sid) not in existing]
            choices = host_balanced(eligible, 1, rng)
            if not choices:
                raise ValueError(f"No background with enough unique flows for {attack}")
            bg, bg_host, _ = choices[0]
            controls = [r for r in backgrounds if r[1] not in (host, bg_host) and r[0] != bg]
            control = host_balanced(controls, 1, rng)[0][0]
            plan.append(dict(pair_id=len(plan), attack=attack, attack_id=int(sid), attack_host=host,
                background_id=int(bg), background_host=bg_host, benign_replacement_id=int(control),
                attack_budget=int(budget), attack_flows_available=int(count)))
    return plan


def verify_original_encodings(reps, reader, plan):
    checked = []
    for sid in (plan[0]["attack_id"], plan[0]["background_id"]):
        events, _ = reader.events(sid)
        for name, encoder, transform in (("scheme1_l2", reps.scheme1, "bucket_l2"), ("scheme2", reps.scheme2, "none")):
            actual, _ = encoder(events)
            config = reps.models[name]["config"]
            with ModelInputs(config["dataset"], "train", transform) as store:
                positions = np.flatnonzero(store.arrays["sample_id"] == sid)
                if len(positions) != 1:
                    raise ValueError("Original donor does not map uniquely to training inputs")
                expected = store.take(positions)["X"][0]
            if not np.allclose(actual, expected, rtol=3e-5, atol=3e-6):
                raise ValueError(f"Raw-flow representation does not reproduce original {name}, donor {sid}, max={np.max(np.abs(actual-expected))}")
            checked.append(dict(model=name, donor_id=sid, maximum_difference=float(np.max(np.abs(actual-expected)))))
    return checked


def build_cases(reps, reader, plan, out, seed):
    values = {name: [] for name in ROUTES}
    masks, metadata, provenance = [], [], []
    for pair in plan:
        rng = np.random.default_rng(seed+1009*pair["pair_id"])
        attack, _ = reader.events(pair["attack_id"], pair["attack"])
        background, bg_meta = reader.events(pair["background_id"])
        replacement, control_meta = reader.events(pair["benign_replacement_id"])
        if bg_meta[3] or control_meta[3] or len(attack) != pair["attack_flows_available"]:
            raise ValueError("Donor labels or class counts differ from the plan")
        attack = attack[rng.choice(len(attack), pair["attack_budget"], replace=False)]
        attack_vector, attack_state = reps.scheme2(attack)
        provenance_rows = []
        for ratio, b, benign in controlled_subsets(attack, background, replacement, rng):
            b_vector, b_state = reps.scheme2(b)
            benign_vector, benign_state = reps.scheme2(benign)
            for positive, a, a_vector, a_state in ((True, attack, attack_vector, attack_state),
                                                   (False, benign, benign_vector, benign_state)):
                events = np.concatenate((a, b))
                x1, exact = reps.scheme1(events)
                x2, physical = reps.scheme2(events)
                added = reps.vector_add(a_vector, a_state, b_vector, b_state)
                if not np.array_equal(exact["flow_counts"], physical["flow_counts"]):
                    raise ValueError("Two representations disagree on raw-flow bucket counts")
                # Verify Scheme1 exact composition independently of concatenation.
                merged = merge_buckets(raw_bucket(a), raw_bucket(b)) if len(b) else raw_bucket(a)
                reference = transform_model_inputs(model_features(merged, reps.schema1, reps.tables, reps.scaler1), merged["mask"], "bucket_l2")
                if not np.allclose(x1, reference, rtol=3e-5, atol=3e-6):
                    raise ValueError("Physical concatenation differs from statistical composition")
                total = int(exact["flow_counts"].sum())
                count = int(exact["attack_flow_counts"].sum())
                if total != len(events) or count != (len(attack) if positive else 0):
                    raise ValueError("Wrong synthetic count or label")
                if positive and Fraction(count, total) != Fraction(ratio, 100):
                    raise ValueError("Wrong exact attack-flow ratio")
                active = exact["mask"] > 0
                overlap = (a_state["flow_counts"] > 0) & (b_state["flow_counts"] > 0)
                cosine = np.sum(x2[:, :-1]*added[:, :-1], axis=-1)
                local = exact["attack_flow_counts"]/np.maximum(exact["flow_counts"], 1)
                attacked = exact["attack_flow_counts"] > 0
                row = dict(case_id=len(metadata), **pair, positive=int(positive), target_attack_percent=ratio,
                    total_flows=total, attack_flows=count, background_flows=len(b),
                    active_buckets=int(active.sum()), attacked_buckets=int(attacked.sum()),
                    overlapping_buckets=int(overlap.sum()),
                    attack_flows_in_overlapping_buckets=int(exact["attack_flow_counts"][overlap].sum()),
                    mean_attacked_bucket_fraction=float(local[attacked].mean()) if positive else None,
                    physical_vs_added_cosine=float(cosine[active].mean()),
                    physical_vs_added_overlap_cosine=float(cosine[overlap].mean()) if overlap.any() else None)
                for route, x in zip(ROUTES, (x1, x2, added)):
                    if not np.isfinite(x).all() or np.any(x[~active] != 0):
                        raise ValueError("Nonfinite inputs or nonzero padding")
                    values[route].append(x)
                metadata.append(row)
                masks.append(exact["mask"].astype(np.float32))
            provenance_rows.append(dict(percent=ratio,
                background_rowids=b[:, bf.E["raw_rowid"]].tolist(),
                replacement_rowids=benign[:, bf.E["raw_rowid"]].tolist()))
        provenance.append(dict(**pair, attack_rowids=attack[:, bf.E["raw_rowid"]].tolist(), subsets=provenance_rows))
        if (pair["pair_id"]+1) % 20 == 0:
            print(f"[controlled pairs] {pair['pair_id']+1}/{len(plan)}", flush=True)
    values = {key: np.stack(v).astype(np.float32) for key, v in values.items()}
    values["mask"] = np.stack(masks)
    np.savez_compressed(out/"stress_inputs.npz", **values)
    write(out/"donor_provenance.json", provenance)
    write_csv(out/"cases.csv", metadata)
    return values, metadata


@torch.no_grad()
def score_cases(models, arrays, metadata, out):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits, predictions = {}, {}
    for route in ROUTES:
        name = "scheme1_l2" if route.startswith("scheme1") else "scheme2"
        info = models[name]
        checkpoint = torch.load(info["run"]/"best.pt", map_location="cpu", weights_only=True)
        model = TemporalAE(**checkpoint["model_args"]).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval().requires_grad_(False)
        params = read(info["report_path"].parent/"linear_probe.json")
        mean, scale, weights = (np.asarray(params[key]) for key in ("mean", "scale", "weights"))
        chunks = []
        for begin in range(0, len(metadata), 1024):
            x = torch.from_numpy(arrays[route][begin:begin+1024]).to(device)
            mask = torch.from_numpy(arrays["mask"][begin:begin+1024]).to(device)
            z = model.encode(x, mask).cpu().numpy()
            chunks.append(((z-mean)/scale)@weights+params["bias"])
        logits[route] = np.concatenate(chunks)
        if not np.isfinite(logits[route]).all():
            raise ValueError("Nonfinite frozen-classifier scores")
        predictions[route] = {}
        for point in THRESHOLDS:
            threshold = info["report"]["linear_probe"]["operating_points"][point]["threshold_logit"]
            predictions[route][point] = logits[route] >= threshold
            for i, row in enumerate(metadata):
                row[route+"_logit"] = float(logits[route][i])
                row[route+"_"+point] = int(predictions[route][point][i])
    write_csv(out/"cases.csv", metadata)
    np.savez_compressed(out/"scores.npz", **logits)
    return predictions


def summarize(metadata, predictions, out):
    attacks = np.array([r["attack"] for r in metadata])
    positive = np.array([r["positive"] for r in metadata], bool)
    ratio = np.array([r["target_attack_percent"] for r in metadata])
    overlap = np.array([r["overlapping_buckets"] for r in metadata])
    rows = []
    for group in ("all", *np.unique(attacks)):
        for target in RATIOS:
            selected = ratio == target
            if group != "all":
                selected &= attacks == group
            for condition in ("all", "overlap_only"):
                eligible = selected & ((overlap > 0) if condition == "overlap_only" else True)
                pos, neg = eligible & positive, eligible & ~positive
                if not eligible.any():
                    continue
                for route in ROUTES:
                    for point in THRESHOLDS:
                        pred = predictions[route][point]
                        rows.append(dict(attack=group, target_attack_percent=target, condition=condition,
                            route=route, threshold=point, positives=int(pos.sum()), negatives=int(neg.sum()),
                            positive_source_hosts=len({metadata[i]["attack_host"] for i in np.flatnonzero(pos)}),
                            detected=int(pred[pos].sum()), recall=float(pred[pos].mean()) if pos.any() else None,
                            benign_alarms=int(pred[neg].sum()), stress_benign_alarm_rate=float(pred[neg].mean()) if neg.any() else None,
                            cosine_mean=float(np.mean([metadata[i]["physical_vs_added_cosine"] for i in np.flatnonzero(pos)])) if pos.any() else None))
    write_csv(out/"slices.csv", rows)
    return rows


def baseline_transitions(metadata, predictions):
    """Distinguish losing an existing detection from never detecting the fragment."""
    positive = np.array([int(r["positive"]) == 1 for r in metadata])
    ratio = np.array([int(r["target_attack_percent"]) for r in metadata])
    pairs = np.array([int(r["pair_id"]) for r in metadata])
    lookup = {int(pairs[i]): i for i in np.flatnonzero(positive & (ratio == 100))}
    rows = []
    for target in RATIOS:
        ids = np.flatnonzero(positive & (ratio == target))
        original_ids = np.array([lookup[int(pairs[i])] for i in ids])
        for route in ROUTES:
            for point in THRESHOLDS:
                pred = predictions[route][point]
                old, new = pred[original_ids], pred[ids]
                rows.append(dict(target_attack_percent=target, route=route, threshold=point,
                    cases=len(ids), originally_detected=int(old.sum()), retained=int((old & new).sum()),
                    lost=int((old & ~new).sum()), gained=int((~old & new).sum()),
                    originally_missed_and_still_missed=int((~old & ~new).sum()),
                    retention=float(new[old].mean()) if old.any() else None))
    return rows


def report_markdown(out, rows, plan, audit):
    targets = sorted({r["target_attack_percent"] for r in rows})
    lines = ["# Controlled background-flow dilution: frozen train-derived stress test", "",
        "Existing real training donors were available during fitting. This is not an unseen-donor holdout or test score.",
        "No parameter fitting, validation/test flow reads, or threshold tuning occurred. Existing validation evaluation artifacts only pin and verify frozen references.",
        "Fixed attack rows are diluted with nested real benign subsets at exact flow ratios; durations/counters/timestamps are unchanged and no rows are duplicated.",
        *(["The 100% attack row is the identical selected attack fragment without background, with an equal-count benign control."] if 100 in targets else []),
        "Matched benign controls replace the same number of attack flows using another benign source. They have equal total counts, not matched byte totals.",
        "Increasing background changes volume and possibly temporal overlap; this is not an isolation of flow ratio from all other statistics.", "",
        "| Route | Meaning |", "|---|---|",
        "| scheme1_l2_physical | Aggregate actual flow union, field features/scaling, final bucket L2 |",
        "| scheme2_physical | Aggregate the same actual union, then apply the frozen bucket encoder |",
        "| scheme2_vector_add | Encode donors separately, add/unit-normalize, append total-flow channel; augmentation-space proxy |", "",
        "The two physical routes evaluate the same traffic. The vector-add route diagnoses the augmentation proxy, not real mixed-traffic deployment performance.", "",
        f"Donor pairs: {len(plan)}; attack types: {len({p['attack'] for p in plan})}. Attack budget per pair: min(5, available class flows).", "",
        "## Coverage", "", "| Attack type | Pairs | Attack hosts | Attack rows per pair |", "|---|---:|---:|---|"]
    for attack in sorted({p["attack"] for p in plan}):
        group = [p for p in plan if p["attack"] == attack]
        lines.append(f"| {attack} | {len(group)} | {len({p['attack_host'] for p in group})} | {sorted({p['attack_budget'] for p in group})} |")
    for point in THRESHOLDS:
        lines += ["", f"## All selected pairs: {point}", "",
            "Rates are descriptive selected-pair averages, not natural prevalence or population FPR.", "",
            "| Attack flow % | Route | n attacks | Recall | n benign controls | Benign alarm rate | Mean physical/add cosine |",
            "|---|---|---:|---:|---:|---:|---:|"]
        for r in rows:
            if r["attack"] == "all" and r["condition"] == "all" and r["threshold"] == point:
                lines.append(f"| {r['target_attack_percent']} | {r['route']} | {r['positives']} | {percent(r['recall'])} | {r['negatives']} | {percent(r['stress_benign_alarm_rate'])} | {r['cosine_mean']:.4f} |")
    lines += ["", "## Attack-specific physical-route comparison: train-benign 99th-percentile thresholds", "",
        "| Attack | Flow % | n | L2 physical recall | S2 physical recall | S2 add-proxy recall |", "|---|---:|---:|---:|---:|---:|"]
    for attack in sorted({p["attack"] for p in plan}):
        for target in targets:
            selected = {r["route"]: r for r in rows if r["attack"] == attack and r["target_attack_percent"] == target and r["condition"] == "all" and r["threshold"] == THRESHOLDS[0]}
            lines.append(f"| {attack} | {target} | {selected[ROUTES[0]]['positives']} | "+" | ".join(percent(selected[route]["recall"]) for route in ROUTES)+" |")
    lines += ["", "## Audit and interpretation", "",
        "- Donor sequence membership is checked as retained train at every read. Background sources differ from attack/control sources; original training donor pairs are excluded.",
        "- New pairs/subsets can reuse flows seen during fitting. No independent-donor or augmentation-benefit claim follows.",
        "- Ratio denotes flow count, not bytes, packets, attack severity or local attacked-bucket dilution. Local overlap/fractions are retained in cases.csv and overlap-only slices.",
        "- Small budgets may remove multi-flow attack structure. This tests weak selected flow fragments, not complete campaigns or all attack intensities.",
        "- Thresholds are the two pre-existing real-training benign quantiles, fixed separately for each model. They need not have equal stress-set alarm rates.",
        "- Every case uses a known exact attack type, not only its sequence primary label. Counts, labels, finite values, zero padding and physical/statistical composition agreement are checked.",
        "- Original full-donor encodings reproduce both frozen input pipelines before stress construction; checks are in report.json.",
        "- Existing models/scalers/vocabularies/category tables remain frozen; source signatures and artifact hashes are checked before/after.",
        "- Interpret recall jointly with matched benign alarms and per-type coverage. High add-proxy recall alone does not establish realistic background-attack detection.",
        "- Outputs include reproducible raw row IDs, input tensors, individual scores/decisions and all subgroup counts."]
    (out/"CONTROLLED_DILUTION.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l2-run", type=Path)
    parser.add_argument("--pairs-per-attack", type=int, default=20)
    parser.add_argument("--seed", type=int, default=4267)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.pairs_per_attack < 1:
        parser.error("Positive pair limit required")
    parent = ROOT/"CIC_new/outputs/diagnostics"
    parent.mkdir(parents=True, exist_ok=True)
    if args.out is None:
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        out = Path(tempfile.mkdtemp(prefix=stamp+"_controlled_dilution_", dir=parent))
    else:
        out = args.out.resolve()
        out.mkdir(parents=True, exist_ok=False)
    write(out/"status.json", dict(status="running", complete=False))
    print(f"Stress diagnostic directory: {out}", flush=True)
    reader = None
    try:
        torch.set_num_threads(2)
        models = load_models(args.l2_run or latest_l2_run())
        reps = Representations(models)
        prepared = Path(read(reps.one/"manifest.json")["prepared_path"])
        reader = EventReader(prepared)
        work = prepared/"work.sqlite"
        work_signature = signature(work)
        plan = select_plan(reader, args.pairs_per_attack, args.seed, reps.one/"donor_plan.npz")
        if not plan:
            raise ValueError("No controlled donor pairs")
        write(out/"plan.json", plan)
        print(f"Selected {len(plan)} train-derived pairs across {len({p['attack'] for p in plan})} attack types", flush=True)
        original_checks = verify_original_encodings(reps, reader, plan)
        arrays, metadata = build_cases(reps, reader, plan, out, args.seed)
        predictions = score_cases(models, arrays, metadata, out)
        rows = summarize(metadata, predictions, out)
        transitions = baseline_transitions(metadata, predictions)
        write_csv(out/"baseline_transitions.csv", transitions)
        reps.assert_unchanged()
        if signature(reader.raw) != reader.source or signature(work) != work_signature:
            raise ValueError("Source database/event cache signature changed")
        for model in models.values():
            if any(digest(model["run"]/name) != expected for name, expected in model["report"]["source_sha256"].items()):
                raise ValueError("Training artifacts changed")
        audit = dict(original_encoding_checks=original_checks, representation_hashes=reps.snapshots,
            source_signature=reader.source, event_cache_signature=work_signature,
            source_unchanged=True, training_artifacts_unchanged=True,
            donor_scope="real retained train; previously available during fitting; new pairs and subsets only",
            test_opened=False, validation_flow_data_opened=False, parameters_fitted=False,
            inherited_validation_reports_read=True, saved_validation_predictions_read=True)
        write(out/"report.json", dict(complete=True, audit=audit, pairs=len(plan), cases=len(metadata),
            seed=args.seed, ratios=RATIOS, routes=ROUTES, summaries=rows, baseline_transitions=transitions,
            source_models={name: str(m["run"]) for name, m in models.items()},
            input_sha256=digest(out/"stress_inputs.npz"), scores_sha256=digest(out/"scores.npz")))
        report_markdown(out, rows, plan, audit)
    except BaseException as error:
        write(out/"status.json", dict(status="failed", complete=False, error=str(error)))
        raise
    finally:
        if reader is not None:
            reader.close()
    write(out/"status.json", dict(status="complete", complete=True, parameters_fitted=False, test_opened=False))
    print(f"COMPLETE: {out/'CONTROLLED_DILUTION.md'}", flush=True)


if __name__ == "__main__":
    main()
