"""
This package contains implementations of different deep learning models for statistical downscaling.
"""

from .deepesd import DeepESD, DeepESDtas, DeepESDpr, NoisyDeepESD, DeepESD_Discriminator
from .unets import UnetTas, UnetPr
from .vit import ViT, NoisyViT
from .deepesdv2 import DeepESDv2, NoisyDeepESDv2

