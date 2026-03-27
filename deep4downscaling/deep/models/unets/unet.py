# SPDX-License-Identifier: MIT

"""
This module contains the U-Net architecture for statistical downscaling.

Authors:
    Jose González-Abad
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..blocks import UnitConv, UpLayer

class UnetTas(torch.nn.Module):

    def __init__(self, x_shape, y_shape, stochastic,
                 input_padding, kernel_size, padding,
                 batch_norm, trans_conv):

        super(UnetTas, self).__init__()

        if (len(x_shape) != 4) or (len(y_shape) != 2):
            error_msg =\
            'X and Y data must have a dimension of length 4'
            'and 2, correspondingly'

            raise ValueError(error_msg)

        self.x_shape = x_shape
        self.y_shape = y_shape
        self.stochastic = stochastic
        self.input_padding = input_padding
        self.kernel_size = int(kernel_size)
        self.padding = padding
        self.batch_norm = batch_norm
        self.trans_conv = trans_conv

        # Encoder
        self.down_conv_1 = UnitConv(in_channels=self.x_shape[1], out_channels=64,
                                    kernel_size=self.kernel_size, padding=self.padding,
                                    batch_norm=self.batch_norm)
        self.maxpool_1 = nn.MaxPool2d((2, 2))

        self.down_conv_2 = UnitConv(in_channels=64, out_channels=128,
                                    kernel_size=self.kernel_size, padding=self.padding,
                                    batch_norm=self.batch_norm)
        self.maxpool_2 = nn.MaxPool2d((2, 2))

        self.down_conv_3 = UnitConv(in_channels=128, out_channels=256,
                                    kernel_size=self.kernel_size, padding=self.padding,
                                    batch_norm=self.batch_norm)
        self.maxpool_3 = nn.MaxPool2d((2, 2))

        self.down_conv_4 = UnitConv(in_channels=256, out_channels=512,
                                    kernel_size=self.kernel_size, padding=self.padding,
                                    batch_norm=self.batch_norm)
        self.maxpool_4 = nn.MaxPool2d((2, 2))

        # Decoder
        self.trans_conv_1 = UpLayer(in_channels=512, out_channels=256,
                                    trans_conv=self.trans_conv)
        self.up_conv_1 = UnitConv(in_channels=512, out_channels=256,
                                  kernel_size=self.kernel_size, padding=self.padding,
                                  batch_norm=self.batch_norm)

        self.trans_conv_2 = UpLayer(in_channels=256, out_channels=128,
                                    trans_conv=self.trans_conv)
        self.up_conv_2 = UnitConv(in_channels=256, out_channels=128,
                                  kernel_size=self.kernel_size, padding=self.padding,
                                  batch_norm=self.batch_norm)

        self.trans_conv_3 = UpLayer(in_channels=128, out_channels=64,
                                    trans_conv=self.trans_conv)
        self.up_conv_3 = UnitConv(in_channels=128, out_channels=64,
                                  kernel_size=self.kernel_size, padding=self.padding,
                                  batch_norm=self.batch_norm)

        # Final segment
        self.trans_conv_4 = UpLayer(in_channels=64, out_channels=64,
                                    trans_conv=self.trans_conv)
        self.up_conv_4 = UnitConv(in_channels=64, out_channels=64,
                                  kernel_size=self.kernel_size, padding=self.padding,
                                  batch_norm=self.batch_norm)

        self.trans_conv_5 = UpLayer(in_channels=64, out_channels=64,
                                    trans_conv=self.trans_conv)
        self.up_conv_5 = UnitConv(in_channels=64, out_channels=64,
                                    kernel_size=self.kernel_size, padding=self.padding,
                                    batch_norm=self.batch_norm)

        if self.stochastic:
            self.out_mean = nn.Conv2d(in_channels=64, out_channels=1,
                                      kernel_size=1)

            self.out_log_var = nn.Conv2d(in_channels=64, out_channels=1,
                                         kernel_size=1)
        else:
            self.out = nn.Conv2d(in_channels=64, out_channels=1,
                                 kernel_size=1)

    def forward(self, x):

        x = F.pad(x, self.input_padding)

        # Encoder
        x1 = self.down_conv_1(x)
        x1_maxpool = self.maxpool_1(x1)

        x2 = self.down_conv_2(x1_maxpool)
        x2_maxpool = self.maxpool_2(x2)

        x3 = self.down_conv_3(x2_maxpool)
        x3_maxpool = self.maxpool_3(x3)

        x4 = self.down_conv_4(x3_maxpool)

        # Decoder
        x5 = self.trans_conv_1(x4)
        x5 = torch.cat([x5, x3], dim=1)
        x5 = self.up_conv_1(x5)

        x6 = self.trans_conv_2(x5)
        x6 = torch.cat([x6, x2], dim=1)
        x6 = self.up_conv_2(x6)

        x7 = self.trans_conv_3(x6)
        x7 = torch.cat([x7, x1], dim=1)
        x7 = self.up_conv_3(x7)

        # Final segment
        x8 = self.trans_conv_4(x7)
        x8 = self.up_conv_4(x8)

        x9 = self.trans_conv_5(x8)
        x9 = self.up_conv_5(x9)

        # Final layers
        if self.stochastic:
            mean = self.out_mean(x9) 
            log_var = self.out_log_var(x9)

            mean = torch.flatten(mean, start_dim=1)
            log_var = torch.flatten(log_var, start_dim=1)

            out = torch.cat([mean, log_var], dim=1)
        else:
            out = self.out(x9)
            out = torch.flatten(out, start_dim=1)

        return out

class UnetPr(torch.nn.Module):

    def __init__(self, x_shape, y_shape, stochastic,
                 input_padding, kernel_size, padding,
                 batch_norm, trans_conv):

        super(UnetPr, self).__init__()

        if (len(x_shape) != 4) or (len(y_shape) != 2):
            error_msg =\
            'X and Y data must have a dimension of length 4'
            'and 2, correspondingly'

            raise ValueError(error_msg)

        self.x_shape = x_shape
        self.y_shape = y_shape
        self.stochastic = stochastic
        self.input_padding = input_padding
        self.kernel_size = int(kernel_size)
        self.padding = padding
        self.batch_norm = batch_norm
        self.trans_conv = trans_conv

        # Encoder
        self.down_conv_1 = UnitConv(in_channels=self.x_shape[1], out_channels=64,
                                    kernel_size=self.kernel_size, padding=self.padding,
                                    batch_norm=self.batch_norm)
        self.maxpool_1 = nn.MaxPool2d((2, 2))

        self.down_conv_2 = UnitConv(in_channels=64, out_channels=128,
                                    kernel_size=self.kernel_size, padding=self.padding,
                                    batch_norm=self.batch_norm)
        self.maxpool_2 = nn.MaxPool2d((2, 2))

        self.down_conv_3 = UnitConv(in_channels=128, out_channels=256,
                                    kernel_size=self.kernel_size, padding=self.padding,
                                    batch_norm=self.batch_norm)
        self.maxpool_3 = nn.MaxPool2d((2, 2))

        self.down_conv_4 = UnitConv(in_channels=256, out_channels=512,
                                    kernel_size=self.kernel_size, padding=self.padding,
                                    batch_norm=self.batch_norm)
        self.maxpool_4 = nn.MaxPool2d((2, 2))

        # Decoder
        self.trans_conv_1 = UpLayer(in_channels=512, out_channels=256,
                                    trans_conv=self.trans_conv)
        self.up_conv_1 = UnitConv(in_channels=512, out_channels=256,
                                  kernel_size=self.kernel_size, padding=self.padding,
                                  batch_norm=self.batch_norm)

        self.trans_conv_2 = UpLayer(in_channels=256, out_channels=128,
                                    trans_conv=self.trans_conv)
        self.up_conv_2 = UnitConv(in_channels=256, out_channels=128,
                                  kernel_size=self.kernel_size, padding=self.padding,
                                  batch_norm=self.batch_norm)

        self.trans_conv_3 = UpLayer(in_channels=128, out_channels=64,
                                    trans_conv=self.trans_conv)
        self.up_conv_3 = UnitConv(in_channels=128, out_channels=64,
                                  kernel_size=self.kernel_size, padding=self.padding,
                                  batch_norm=self.batch_norm)

        # Final segment
        self.trans_conv_4 = UpLayer(in_channels=64, out_channels=64,
                                    trans_conv=self.trans_conv)
        self.up_conv_4 = UnitConv(in_channels=64, out_channels=64,
                                  kernel_size=self.kernel_size, padding=self.padding,
                                  batch_norm=self.batch_norm)

        self.trans_conv_5 = UpLayer(in_channels=64, out_channels=64,
                                    trans_conv=self.trans_conv)
        self.up_conv_5 = UnitConv(in_channels=64, out_channels=64,
                                    kernel_size=self.kernel_size, padding=self.padding,
                                    batch_norm=self.batch_norm)

        if self.stochastic:
            self.p = nn.Conv2d(in_channels=64, out_channels=1,
                               kernel_size=1)

            self.log_shape = nn.Conv2d(in_channels=64, out_channels=1,
                                       kernel_size=1)

            self.log_scale = nn.Conv2d(in_channels=64, out_channels=1,
                                       kernel_size=1)

        else:
            self.out = nn.Conv2d(in_channels=64, out_channels=1,
                                 kernel_size=1)
    def forward(self, x):

        x = F.pad(x, self.input_padding)

        # Encoder
        x1 = self.down_conv_1(x)
        x1_maxpool = self.maxpool_1(x1)

        x2 = self.down_conv_2(x1_maxpool)
        x2_maxpool = self.maxpool_2(x2)

        x3 = self.down_conv_3(x2_maxpool)
        x3_maxpool = self.maxpool_3(x3)

        x4 = self.down_conv_4(x3_maxpool)

        # Decoder
        x5 = self.trans_conv_1(x4)
        x5 = torch.cat([x5, x3], dim=1)
        x5 = self.up_conv_1(x5)

        x6 = self.trans_conv_2(x5)
        x6 = torch.cat([x6, x2], dim=1)
        x6 = self.up_conv_2(x6)

        x7 = self.trans_conv_3(x6)
        x7 = torch.cat([x7, x1], dim=1)
        x7 = self.up_conv_3(x7)

        # Final segment
        x8 = self.trans_conv_4(x7)
        x8 = self.up_conv_4(x8)

        x9 = self.trans_conv_5(x8)
        x9 = self.up_conv_5(x9)

        # Final layers
        if self.stochastic:
            p = self.p(x9)
            p = torch.sigmoid(p)
            p = torch.flatten(p, start_dim=1)

            log_shape = self.log_shape(x9)
            log_shape = torch.flatten(log_shape, start_dim=1)

            log_scale = self.log_scale(x9)
            log_scale = torch.flatten(log_scale, start_dim=1)

            out = torch.cat((p, log_shape, log_scale), dim = 1)
        else:
            out = self.out(x9)
            out = torch.flatten(out, start_dim=1)

        return out

