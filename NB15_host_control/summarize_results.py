"""Audit completed runs and score their identical test-window intersection."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "team_snapshot"))
from evaluate import evaluate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=HERE / "artifacts/seed0_w20s10")
    args = ap.parse_args()
    run = args.run
    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    audits = {arm: json.loads((run / arm / "split_audit.json").read_text(encoding="utf-8")) for arm in ("time", "host")}
    assert all(not hosts for hosts in audits["host"]["pairwise_source_host_overlap"].values())
    for split in ("val", "test"):
        assert not set(audits["host"]["scaler_fit_hosts"]) & set(audits["host"]["window_hosts"][split])
    scoreframes = {arm: pd.read_csv(run / arm / "test_scores.csv") for arm in ("time", "host")}
    common = scoreframes["time"].merge(scoreframes["host"], on="window_id", suffixes=("_time", "_host"), validate="one_to_one")
    assert len(common) and (common.label_time == common.label_host).all()
    assert (common.host_time == common.host_host).all()
    matched = {}
    for arm in ("time", "host"):
        matched[arm] = evaluate(common[f"score_{arm}"], common.label_time, common.attack_time,
                                np.load(run / arm / "benign_validation_scores.npy"))
    (run / "matched_test_metrics.json").write_text(json.dumps(matched, indent=2), encoding="utf-8")
    common.to_csv(run / "matched_test_scores.csv", index=False)
    lines = ["# NB15 source-host control: seed 0", "",
             "Actual completed GPU runs; same team CNN-AE and features, different source-host split and corresponding scaler-fit population.", "",
             "| Split / evaluation | Windows | Attacks | ROC-AUC | AP | Recall at validation 1% FPR | Actual FPR |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in ("time", "host"):
        for split in ("val", "test"):
            m = results[arm][split]
            lines.append(f"| {arm} / {split} | {m['n_windows']:,} | {m['n_malicious']:,} | {m['roc_auc']:.6f} | {m['pr_auc']:.6f} | {m['dr@fpr0.01']:.2%} | {m['actual_fpr@fpr0.01']:.2%} |")
    for arm, m in matched.items():
        lines.append(f"| {arm} / SAME test intersection | {m['n_windows']:,} | {m['n_malicious']:,} | {m['roc_auc']:.6f} | {m['pr_auc']:.6f} | {m['dr@fpr0.01']:.2%} | {m['actual_fpr@fpr0.01']:.2%} |")
    lines += ["", "Matched evaluation source-host coverage (same windows, no additional training):", ""]
    for (host, label), count in common.groupby(["host_time", "label_time"]).size().items():
        lines.append(f"- {host}: {count:,} windows; label {label}.")
    lines += ["", "The matched subset contains only one benign 149.* window, so its perfect AUC does not test the full problematic 149.* subgroup."]
    lines += ["", "## Source-host isolation audit", "",
              f"Host arm, source-IP counts: train {len(audits['host']['window_hosts']['train'])}; validation {len(audits['host']['window_hosts']['val'])}; test {len(audits['host']['window_hosts']['test'])}.",
              "All three pairwise source-host intersections are empty; scaler-fit hosts exclude validation/test hosts.",
              "Destination endpoints may overlap. Full-data unlabeled category vocabulary and original relative-time feature construction are retained for faithful reproduction, not claimed fully inductive preprocessing.", ""]
    for split in ("train", "val", "test"):
        lines.append(f"- Host-arm {split}: {', '.join(audits['host']['window_hosts'][split])}")
    lines += ["", "## Benign-family subgroup checks (independent full test sets)", "",
              "Each subgroup compares that benign family against all attacks from its arm's test set. AP/base rates therefore differ from the overall test set.", "",
              "| Arm | Benign family | Benign windows | ROC-AUC | Recall | Actual FPR |",
              "|---|---|---:|---:|---:|---:|"]
    for arm in ("time", "host"):
        for fam, m in results[arm]["test"]["benign_family_vs_all_attacks"].items():
            lines.append(f"| {arm} | {fam}.* | {m['n_windows'] - m['n_malicious']:,} | {m['roc_auc']:.6f} | {m['dr@fpr0.01']:.2%} | {m['actual_fpr@fpr0.01']:.2%} |")
    lines += ["", "## Main finding", "",
              "The host arm retains very high overall AUC and 100% attack recall, but all 100 benign windows from four held-out 149.171.126.* sources are false positives. Those 100 windows represent only 0.22% of test benign windows and include overlapping sequences; they are not 100 independent host observations. The subgroup AUC is approximately 0.479, versus approximately 0.999959 for 59.* benign windows against the same attacks.",
              "This exposes a source-host/role-associated subgroup failure concealed by the overall metric. It does NOT isolate memorization of numerical IP identities: 149.* also exists in benign training, source roles and capture times differ, and the original chronological test does not contain the same rare-source composition. In particular, this run does not reproduce the earlier training-on-59-only scenario.",
              "The time baseline also has a rare benign 10.40.85.* subgroup with 95.83% false positives (24 windows). Small normal subpopulations warrant separate reporting even when aggregate AUC is excellent."]
    lines += ["", "## Coverage and interpretation", ""]
    for arm in ("time", "host"):
        absent = audits[arm]["test_absent_attack_types"]
        lines.append(f"- {arm}: attack types present in any generated window but absent from test: {', '.join(absent) if absent else 'none'}.")
    lines += ["- Window-level absence reporting cannot recover attacks lost before/inside windowing; minority labels can disappear under majority attack-type assignment. Worms is not represented by any generated window in this run.",
              "- Existing team ablation tables are validation-selection numbers, not these new independent-test results.",
              "- One seed, no feature tuning or test-driven parameter selection. Host holdout changes sample counts, time coverage and attack-source distribution. The matched test intersection fixes evaluation examples, not training-population/time-coverage confounding.",
              "- This tests unseen source IPs within the same NB15 capture environment. It does not establish generalization to a different benign role/subnet, laboratory or dataset, nor prove causal IP memorization.",
              "", "See README.md for the exact configuration, provenance, and rerun commands.", ""]
    report = "\n".join(lines)
    (run / "RESULTS.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
