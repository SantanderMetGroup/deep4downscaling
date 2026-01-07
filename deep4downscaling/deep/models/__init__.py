"""
This package contains implementations of different deep learning models for statistical downscaling.
"""

from .deepesd import DeepESDtas, DeepESDpr, NoisyDeepESD, DeepESD_Discriminator, DeepESDMultiHead
from .unets import UnetTas, UnetPr
from .vit import ViT, NoisyViT

