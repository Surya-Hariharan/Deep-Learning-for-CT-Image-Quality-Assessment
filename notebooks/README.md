# notebooks/

| file | what it shows |
|---|---|
| `01_ngpnet_walkthrough.ipynb` | Builds NGP-Net, loads one real three-timepoint window from the PNG dataset (falls back to a synthetic window if the dataset isn't present), runs a forward pass, and visualizes the inputs, predicted image/mask, and deformation field. Currently uses a randomly initialized model -- see the notebook's own status note and `docs/research-status.md`. |

## Running

```bash
pip install -e ".[notebook]"
jupyter notebook notebooks/01_ngpnet_walkthrough.ipynb
```

Or non-interactively, to re-execute and refresh its saved outputs:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_ngpnet_walkthrough.ipynb
```

## `_build_*.py` generator scripts

Each notebook has a matching `_build_<name>.py` script that constructs it
via `nbformat` and is the notebook's actual source of truth -- `.ipynb`
JSON is not hand-edited. To change a notebook: edit its `_build_*.py`,
then regenerate and re-execute:

```bash
python notebooks/_build_walkthrough.py
jupyter nbconvert --to notebook --execute --inplace notebooks/01_ngpnet_walkthrough.ipynb
```

This keeps the notebook's structure reviewable as a plain-Python diff
instead of opaque notebook JSON, and guarantees the checked-in `.ipynb`
was actually executed (its outputs are real, not hand-written).
