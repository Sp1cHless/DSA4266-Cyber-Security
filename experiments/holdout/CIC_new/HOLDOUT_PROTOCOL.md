# Phase II Training-Holdout Protocol

This guide follows Section VI of the final report. The fixed 100s/10s pipeline is
retained from `feature/cic-holdout`; use the [workflow commands](../README.md).

## Arms

| Arm | Excluded primary labels | Related training types | Removed real training attacks | Remaining P |
|---|---|---|---:|---:|
| Control | None | All available types | 0 | 8093 |
| A | XSS, SlowHTTPTest, HOIC, SSH brute force | Web BF/SQL, other DoS, LOIC-HTTP/UDP and FTP brute force remain | 246 | 7847 |
| B-1 | All seven DoS/DDoS flooding labels | No flooding sibling | 313 | 7780 |
| B-2 | All three Web labels | No Web sibling | 1032 | 7061 |

Exact strings for `--holdout-train-attacks`:

- A: `Brute_Force_-XSS,DoS_attacks-SlowHTTPTest,DDOS_attack-HOIC,SSH-Bruteforce`
- B-1: `DoS_attacks-GoldenEye,DoS_attacks-Hulk,DoS_attacks-Slowloris,DoS_attacks-SlowHTTPTest,DDOS_attack-HOIC,DDoS_attacks-LOIC-HTTP,DDOS_attack-LOIC-UDP`
- B-2: `Brute_Force_-Web,Brute_Force_-XSS,SQL_Injection`

The report groups FTP and SSH under brute force. Bot and Infilteration stay in
training in all three arms.

## Construction and selection

1. Allocate time blocks, coverage reservations and boundary exclusions exactly
   as for control, then remove training sequences whose primary label is held out.
2. Refit vocabulary, scaler, bucket encoder and donor plan on retained training
   sequences. Removed sequences cannot re-enter through synthetic donors.
3. Retain identical validation/test evaluation populations. Primary-label removal
   does not remove every secondary held-out-type flow; inspect the residual audit.
4. Train control and holdout detectors with seeds 4266, 4267 and 4268. Pair arms
   by seed; the input-encoder/donor plan within each arm stays fixed across seeds.
5. Select checkpoints and global 1%/0.1% FPR thresholds using validation, which
   still includes excluded types. Apply frozen thresholds to test and report
   achieved test FPR separately from the validation budget.

`--holdout-train-attacks` must be supplied during a fresh build. It cannot be
added to an existing `--prepared` directory. Resume requires identical settings;
complete outputs are preserved and interrupted stages require fresh locations.

## Audit and reporting

Check `workflow.json`, `dataset/validation.json`, `train_holdout_attacks`, reduced
training counts, evaluation IDs and donor ownership. The original implementation's
per-type `*_sequences_primary` holdout audit uses membership in the full held-out
set when multiple types coexist. It can overcount an individual Web type; do not
sum those fields as true per-label populations. The final report's subtype n and
recall are grouped by each sequence's actual `primary_attack`. Original source is
preserved; this audit limitation is documented rather than changing a completed
experiment's implementation or results.

Use `unseen_subtype_report.py` for one frozen pair and `seed_report.py` for paired
three-seed summaries. Reported values are in the
[research results](../../../docs/research/08_training_exclusions/README.md).
The retained historical three-subtype/four-model summary and later single-seed
geometry diagnostics are separate from final-report Tables VI-IX.

## Evidence boundaries

Small subtype populations, host/campaign overlap and historical test inspection
limit generalisation. Excluded types still influence validation selection, so
this is transfer after training exclusion. Raw per-seed prediction arrays are
needed for independent rescoring; the published CSVs preserve recorded report
values and are not a new experiment.
