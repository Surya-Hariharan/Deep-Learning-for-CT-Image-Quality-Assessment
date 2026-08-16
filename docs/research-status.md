# Research status

| component | status |
|---|---|
| NGP-Net integration (packaging, config, dataset loader, training/eval loops) | **implemented** |
| PNG dataset access -- manifest, splits, loader, `.nii`/`.nii.gz` resolution | **implemented** |
| NGP-Net baseline **trained** | **not started** -- no checkpoint exists; see `docs/reproduction.md` |
| Image Quality Assessment (Q) | **planned** -- `quality/` is a boundary package only |
| Prediction Uncertainty (U) | **planned** -- deliberately unformulated; NGP-Net outputs a warped image + segmentation logits + deformation field, not a classification, so the usual entropy-based uncertainty formulation does not apply directly. A formulation should follow measuring the trained baseline, not precede it. |
| Quality-Uncertainty Fusion | **planned** -- this is the project's actual contribution; designed after Q and U exist. |
| Accept/Flag/Reject decision rule | **planned** -- depends on Fusion. |
| Structured Longitudinal Report | **planned** -- depends on the decision rule. |

Nothing under `quality/`, `uncertainty/`, `fusion/`, or `reporting/`
contains a stub, placeholder, or fake implementation. Each package's
`__init__.py` docstring states its status; import them for their (empty)
namespace, not for behavior.
