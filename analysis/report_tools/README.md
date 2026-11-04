# Historical Report Tools

These scripts preserve the original source of completed report-production work.
They are not part of model training or evaluation. Report drafts, exported media
and rendering output are archived and excluded from the active repository.

| Folder | Source purpose |
|---|---|
| [part5_revision_20261103_v1](part5_revision_20261103_v1/README.md) | First replacement-section plots and preview |
| [part5_part7_revision_20261103_v2](part5_part7_revision_20261103_v2/README.md) | Scoped Part 5/7 evidence plots and preview |
| [part5_part7_revision_20261103_v3](part5_part7_revision_20261103_v3/README.md) | Revised labels and expanded preview |
| [host_information](host_information/README.md) | Host-probe figure labels |

Source bytes are unchanged. The scripts retain their historical input paths and
some expect drafts/results beside the script. To reproduce an old preview,
restore its original tree from the local archive and run it in that original
location with NumPy, Matplotlib and ReportLab installed. These are historical
source snapshots, not standalone rendering commands for a fresh clone.

Current representation export and viewer source remains in `train/`:
`run_part6_visualizations.py`, `plot_part6_visualizations.py`,
`prepare_part6_projector_website.py`, `compose_part6_projector_screenshots.py`,
`compose_part6_orientation_v2.py` and `part6_viewer_template.html`.
