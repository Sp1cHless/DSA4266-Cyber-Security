# Research Index

Organized on 1 November 2026 by the report's research logic. Start with the research question and finding, then inspect the relevant table, figure or implementation. The [Word report used for this organization](../GROUP%20REPORT.docx) supplies the Section 3-6 structure and figures. The [document guide](../README.md) also lists the separately created discussion revision. [Compact project findings](../RESEARCH_FINDINGS.md) retain earlier research context.

| Research branch | Report | Contents |
|---|---|---|
| [01 Data exploration](01_data_exploration/README.md) | 3 | Observation unit, subtype support, sparsity, temporal scale, natural background and split feasibility |
| [02 Representation](02_representation/README.md) | 4, 5.1 | Scheme 1, L2, Scheme 2 and representation selection |
| [03 Model selection](03_model_selection/README.md) | 5.2 | Four families, five seeds and circular-rotation retraining |
| [04 Attack profiles](04_attack_profiles/README.md) | 5.3 | Projector distributions and native-space neighborhood comparisons |
| [05 Host information](05_host_information/README.md) | 5.3 | Source information, adversarial tradeoffs and the motivation for targeted studies |
| [06 Web and temporal context](06_web_and_temporal_context/README.md) | 6.1 | Sparse local evidence, model complementarity, common-anchor history and rebuilt temporal geometries |
| [07 Targeted studies](07_targeted_studies/README.md) | 6.2, 6.3 | Input ambiguity, attention, feature probes, distribution extension, mixing and bucket supervision |
| [08 Training exclusions](08_training_exclusions/README.md) | 6.4 | Related-subtype versus broad-family removal, three-seed summaries and separate geometry diagnostics |
| [09 Evidence review](09_evidence_review/README.md) | 5, 6 | Fact checks, figure decisions and evidence limits |

## Folder contract

- `README.md`: one study's finding, evaluation scope, report references and implementation links.
- `figures/`: only figures used in the current report, with PNG and editable SVG exports.
- `results/`: compact completed summaries, current report tables and source provenance.
- `local/`: original working bundles, insertion drafts, original screenshots and superseded charts; ignored by Git.
- [ARTIFACTS.csv](ARTIFACTS.csv): published artifact inventory with report references, original paths and SHA256 hashes.
- [EXPERIMENTS.csv](EXPERIMENTS.csv): study-to-evidence routing with local availability and hashes for saved source records.

The [historical research tree](overview/RESEARCH_TREE.md) preserves the 29 October overview and its PDF/SVG exports. It predates the completed host study; use the table above for the current report structure.

## Reproduction and local evidence

Reusable construction code stays in `CIC_new/`, detector implementations in `train/`, and diagnostic scripts in `analysis/`. This avoids changing imports and established run commands for a presentation-only reorganization. Each study links its relevant entry points.

The [complete code guide](../code/README.md) classifies all implementation and
configuration files, resolves static module dependencies, distinguishes baseline
models from extensions and records the team holdout version boundary.

Training data, caches, weights, complete run directories, prediction arrays, detailed machine-specific evidence maps and report-rendering output stay local. They are intentionally absent from a clean clone. Small published CSVs are conclusions or report transcriptions, not training datasets. Historical evidence paths can be resolved locally through `_local_archive/project_organization_20261101/relocation_manifest.json`.

## Evaluation boundaries

Keep full-source EDA, reused-training probes, oracle stress diagnostics, inspected-validation comparisons and frozen historical-test summaries distinct. Table H1 is checked against the group's committed three-seed summary; its raw per-seed prediction arrays are not available in this checkout. The reorganization introduces no new experiment or numerical result.
