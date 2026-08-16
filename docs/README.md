# docs/

Documentation index. If a doc below and the code disagree, the code is
correct and the doc is stale -- open an issue/PR rather than trusting the
doc silently.

| file | read this for |
|---|---|
| [architecture.md](architecture.md) | Package layout, and exactly what changed vs. the original NGP-Net author code (file-by-file provenance table). |
| [datasets.md](datasets.md) | PNG dataset's measured properties, the manifest's `.nii`/`.nii.gz` defect, and the training/test split overlap. |
| [reproduction.md](reproduction.md) | Environment setup, dataset validation, smoke test, training, evaluation -- the commands, in order. |
| [research-status.md](research-status.md) | What's implemented vs. planned, per pipeline stage, and why (e.g. why Uncertainty is deliberately unformulated). |
| [upstream/](upstream/) | The vendored NGP-Net authors' own README and figures, kept for attribution -- not a specification for this repo. |

See also: [`../notebooks/README.md`](../notebooks/README.md) for the
visual walkthrough notebook, and [`../data/README.md`](../data/README.md)
for dataset setup.
