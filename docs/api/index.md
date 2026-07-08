# API reference

The API reference is **auto-generated from docstrings** in the source code. When you update a function's docstring, the reference updates automatically on the next docs build.

## Package layout

```
deep4downscaling/
├── trans.py          # Data preprocessing and transforms
├── viz.py            # Visualization helpers
├── metrics.py        # Historical evaluation metrics
├── metrics_ccs.py    # Climate change signal utilities
└── deep/
    ├── train.py      # Training loops
    ├── pred.py       # Prediction and inference
    ├── utils.py      # Dataset helpers
    ├── tracker.py    # Training progress logging
    ├── xai.py        # Explainability methods
    ├── models/       # Neural network architectures
    └── loss/         # Loss functions
```

## Public API policy

The following are considered **stable public API**:

| Module | Key exports |
| --- | --- |
| `deep4downscaling.trans` | All preprocessing functions |
| `deep4downscaling.metrics` | All metric functions |
| `deep4downscaling.metrics_ccs` | CCS reducers and `compute_ccs` |
| `deep4downscaling.viz` | Map plotting functions |
| `deep4downscaling.deep.models` | `DeepESD`, `DeepESDtas`, `DeepESDpr`, `NoisyDeepESD`, `DeepESD_Discriminator`, `UnetTas`, `UnetPr`, `ViT`, `NoisyViT` |
| `deep4downscaling.deep.loss` | All exported loss classes |
| `deep4downscaling.deep.train` | Training loop functions |
| `deep4downscaling.deep.pred` | Prediction functions |
| `deep4downscaling.deep.xai` | XAI functions |

Internal helpers (names starting with `_`) are not part of the public API and may change without notice.

## Modules

- [trans](trans.md) — preprocessing
- [viz](viz.md) — visualization
- [metrics](metrics.md) — evaluation metrics
- [metrics_ccs](metrics_ccs.md) — climate change signals
- [deep.train](deep/train.md) — training
- [deep.pred](deep/pred.md) — prediction
- [deep.models](deep/models.md) — neural networks
- [deep.loss](deep/loss.md) — losses
- [deep.xai](deep/xai.md) — explainability
- [deep.utils](deep/utils.md) — utilities
- [deep.tracker](deep/tracker.md) — training tracker

## Docstring style

All public functions use **NumPy-style docstrings** with `Parameters`, `Returns`, and optional `Notes` sections. When contributing new code, follow the style in `deep4downscaling.trans`.
