# Loss functions

This folder contains custom loss functions used by deep learning models in `deep4downscaling`.

## Included losses

- Standard regression losses (`MaeLoss`, `MseLoss`).
- Probabilistic negative log-likelihood losses (`NLLGaussianLoss`, `NLLBerGammaLoss`).
- Asymmetric loss (`Asym`) for weighted penalties.
- CRPS-based losses (`CRPSLoss`, `CRPSSpectralLoss`) for probabilistic forecasts.

These losses support both deterministic and stochastic downscaling configurations.
