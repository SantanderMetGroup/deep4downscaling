import torch
import torch.nn as nn

class UnitConv(nn.Module):
    
    """
    Implement the following set of layers:
    2D convolution => Batch Normalization (opt.) => ReLU (x2)

    Parameters
    ----------
    in_channels : int
        Input channels to the block

    out_channels : int
        Output channels of the block

    kernel_size : int
        Kernel size of all convolutions applied within the
        block

    padding: str
        Padding (same or valid) to apply before each convolutional
        layer
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 padding: int, batch_norm: bool):
        super().__init__()
 
        if batch_norm:
            self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU())
        else:

            self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
                    nn.ReLU())

    def forward(self, x):
        return self.conv(x)

class UpLayer(nn.Module):
    
    """
    Implement one of the following set of layers:
    (2D transposed conv) or (up sampling => 2D convolution)

    Parameters
    ----------
    in_channels : int
        Input channels to the block

    out_channels : int
        Output channels of the block

    trans_conv: bool
        Whether to apply the transposed convolution (True)
        or the up-sampling + 2D convolution
    """

    def __init__(self, in_channels: int, out_channels: int, trans_conv: bool):
        super().__init__()
 
        if trans_conv:
            self.layer_op = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                               kernel_size=2, stride=2)
        else:
            self.layer_op = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )

    def forward(self, x):
        return self.layer_op(x)

