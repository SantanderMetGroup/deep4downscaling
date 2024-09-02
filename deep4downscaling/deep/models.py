import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

class DeepESDtas(torch.nn.Module):

    """
    DeepESD model as proposed in Baño-Medina et al. 2024 for temperature
    downscasling. This implementation allows for a deterministic (MSE-based)
    and stochastic (NLL-based) definition.

    Baño-Medina, J., Manzanas, R., Cimadevilla, E., Fernández, J., González-Abad,
    J., Cofiño, A. S., and Gutiérrez, J. M.: Downscaling multi-model climate projection
    ensembles with deep learning (DeepESD): contribution to CORDEX EUR-44, Geosci. Model
    Dev., 15, 6747–6758, https://doi.org/10.5194/gmd-15-6747-2022, 2022.

    Parameters
    ----------
    x_shape : tuple
        Shape of the data used as predictor. This must have dimension 4
        (time, channels/variables, lon, lat).

    y_shape : tuple
        Shape of the data used as predictand. This must have dimension 2
        (time, gridpoint)

    filters_last_conv : int
        Number of filters/kernels of the last convolutional layer

    stochastic: bool
        If set to True, the model is composed of two final dense layers computing
        the mean and log fo the variance. Otherwise, the models is composed of one
        final layer computing the values.
    """


    def __init__(self, x_shape: tuple, y_shape: tuple,
                 filters_last_conv: int, stochastic: bool):

        super(DeepESDtas, self).__init__()

        if (len(x_shape) != 4) or (len(y_shape) != 2):
            error_msg =\
            'X and Y data must have a dimension of length 4'
            'and 2, correspondingly'

            raise ValueError(error_msg)

        self.x_shape = x_shape
        self.y_shape = y_shape
        self.filters_last_conv = filters_last_conv
        self.stochastic = stochastic

        self.conv_1 = torch.nn.Conv2d(in_channels=self.x_shape[1],
                                      out_channels=50,
                                      kernel_size=3,
                                      padding=1)

        self.conv_2 = torch.nn.Conv2d(in_channels=50,
                                      out_channels=25,
                                      kernel_size=3,
                                      padding=1)

        self.conv_3 = torch.nn.Conv2d(in_channels=25,
                                      out_channels=self.filters_last_conv,
                                      kernel_size=3,
                                      padding=1)

        if self.stochastic:
            self.out_mean = torch.nn.Linear(in_features=\
                                            self.x_shape[2] * self.x_shape[3] * self.filters_last_conv,
                                            out_features=self.y_shape[1])

            self.out_log_var = torch.nn.Linear(in_features=\
                                               self.x_shape[2] * self.x_shape[3] * self.filters_last_conv,
                                               out_features=self.y_shape[1])

        else:
            self.out = torch.nn.Linear(in_features=\
                                       self.x_shape[2] * self.x_shape[3] * self.filters_last_conv,
                                       out_features=self.y_shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv_1(x)
        x = torch.relu(x)

        x = self.conv_2(x)
        x = torch.relu(x)

        x = self.conv_3(x)
        x = torch.relu(x)

        x = torch.flatten(x, start_dim=1)

        if self.stochastic:
            mean = self.out_mean(x)
            log_var = self.out_log_var(x)
            out = torch.cat((mean, log_var), dim=1)
        else:
            out = self.out(x)

        return out

class DeepESDpr(torch.nn.Module):

    """
    DeepESD model as proposed in Baño-Medina et al. 2024 for precipitation
    downscaling. This implementation allows for a deterministic (MSE-based)
    and stochastic (NLL-based) definition.

    Baño-Medina, J., Manzanas, R., Cimadevilla, E., Fernández, J., González-Abad,
    J., Cofiño, A. S., and Gutiérrez, J. M.: Downscaling multi-model climate projection
    ensembles with deep learning (DeepESD): contribution to CORDEX EUR-44, Geosci. Model
    Dev., 15, 6747–6758, https://doi.org/10.5194/gmd-15-6747-2022, 2022.

    Parameters
    ----------
    x_shape : tuple
        Shape of the data used as predictor. This must have dimension 4
        (time, channels/variables, lon, lat).

    y_shape : tuple
        Shape of the data used as predictand. This must have dimension 2
        (time, gridpoint)

    filters_last_conv : int
        Number of filters/kernels of the last convolutional layer

    stochastic: bool
        If set to True, the model is composed of three final dense layers computing
        the p, shape and scale of the Bernoulli-gamma distribution. Otherwise,
        the models is composed of one final layer computing the values.
    """


    def __init__(self, x_shape: tuple, y_shape: tuple,
                 filters_last_conv: int, stochastic: bool):

        super(DeepESDpr, self).__init__()

        if (len(x_shape) != 4) or (len(y_shape) != 2):
            error_msg =\
            'X and Y data must have a dimension of length 4'
            'and 2, correspondingly'

            raise ValueError(error_msg)

        self.x_shape = x_shape
        self.y_shape = y_shape
        self.filters_last_conv = filters_last_conv
        self.stochastic = stochastic

        self.conv_1 = torch.nn.Conv2d(in_channels=self.x_shape[1],
                                      out_channels=50,
                                      kernel_size=3,
                                      padding=1)

        self.conv_2 = torch.nn.Conv2d(in_channels=50,
                                      out_channels=25,
                                      kernel_size=3,
                                      padding=1)

        self.conv_3 = torch.nn.Conv2d(in_channels=25,
                                      out_channels=self.filters_last_conv,
                                      kernel_size=3,
                                      padding=1)

        if self.stochastic:
            self.p = torch.nn.Linear(in_features=\
                                            self.x_shape[2] * self.x_shape[3] * self.filters_last_conv,
                                            out_features=self.y_shape[1])

            self.log_shape = torch.nn.Linear(in_features=\
                                             self.x_shape[2] * self.x_shape[3] * self.filters_last_conv,
                                             out_features=self.y_shape[1])

            self.log_scale = torch.nn.Linear(in_features=\
                                             self.x_shape[2] * self.x_shape[3] * self.filters_last_conv,
                                             out_features=self.y_shape[1])

        else:
            self.out = torch.nn.Linear(in_features=\
                                       self.x_shape[2] * self.x_shape[3] * self.filters_last_conv,
                                       out_features=self.y_shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv_1(x)
        x = torch.relu(x)

        x = self.conv_2(x)
        x = torch.relu(x)

        x = self.conv_3(x)
        x = torch.relu(x)

        x = torch.flatten(x, start_dim=1)

        if self.stochastic:
            p = self.p(x)
            p = torch.sigmoid(p)

            log_shape = self.log_shape(x)
            log_scale = self.log_scale(x)

            out = torch.cat((p, log_shape, log_scale), dim = 1)
        else:
            out = self.out(x)

        return out

class DeepESDElevationTas(torch.nn.Module):

    """
    DeepESD model integrating elevation data for temperature downscaling.
    Instead of just predicting the target variable, it also predicts the
    elevation of the predictand domain with an additional dense layer,
    thus being forced to learn some representation of the elevation.

    This elevation model is only defined for stochastic=False, otherwise the
    increase of the number of parameters would be too high.

    Parameters
    ----------
    x_shape : tuple
        Shape of the data used as predictor. This must have dimension 4
        (time, channels/variables, lon, lat).

    y_shape : tuple
        Shape of the data used as predictand. This must have dimension 2
        (time, gridpoint)

    filters_last_conv : int
        Number of filters/kernels of the last convolutional layer

    stochastic: bool
        If set to True, the model is composed of two final dense layers computing
        the mean and log fo the variance. Otherwise, the models is composed of one
        final layer computing the values.
    """


    def __init__(self, x_shape: tuple, y_shape: tuple,
                 filters_last_conv: int, stochastic: bool):

        super(DeepESDElevationTas, self).__init__()

        if (len(x_shape) != 4) or (len(y_shape) != 2):
            error_msg =\
            'X and Y data must have a dimension of length 4'
            'and 2, correspondingly'

            raise ValueError(error_msg)

        self.x_shape = x_shape
        self.y_shape = y_shape
        self.filters_last_conv = filters_last_conv

        self.conv_1 = torch.nn.Conv2d(in_channels=self.x_shape[1],
                                      out_channels=50,
                                      kernel_size=3,
                                      padding=1)

        self.conv_2 = torch.nn.Conv2d(in_channels=50,
                                      out_channels=25,
                                      kernel_size=3,
                                      padding=1)

        self.conv_3 = torch.nn.Conv2d(in_channels=25,
                                      out_channels=self.filters_last_conv,
                                      kernel_size=3,
                                      padding=1)

        if self.stochastic:
            self.out_mean = torch.nn.Linear(in_features=\
                                            self.x_shape[2] * self.x_shape[3] * self.filters_last_conv,
                                            out_features=self.y_shape[1])

            self.out_log_var = torch.nn.Linear(in_features=\
                                               self.x_shape[2] * self.x_shape[3] * self.filters_last_conv,
                                               out_features=self.y_shape[1])

        else:
            self.out = torch.nn.Linear(in_features=\
                                       self.x_shape[2] * self.x_shape[3] * self.filters_last_conv,
                                       out_features=self.y_shape[1])

        self.elev_out = torch.nn.Linear(in_features=\
                                        self.x_shape[2] * self.x_shape[3] * self.filters_last_conv,
                                        out_features=self.y_shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv_1(x)
        x = torch.relu(x)

        x = self.conv_2(x)
        x = torch.relu(x)

        x = self.conv_3(x)
        x = torch.relu(x)

        x = torch.flatten(x, start_dim=1)

        if self.stochastic:
            mean = self.out_mean(x)
            log_var = self.out_log_var(x)
            out = torch.cat((mean, log_var), dim=1)
        else:
            out = self.out(x)

        out_elev = self.elev_out(x)

        return out, out_elev

class DeepESDElevationPr(torch.nn.Module):

    """
    DeepESD model integrating elevation data for precipitation downscaling.
    Instead of just predicting the target variable, it also predicts the
    elevation of the predictand domain with an additional dense layer,
    thus being forced to learn some representation of the elevation.

    Parameters
    ----------
    x_shape : tuple
        Shape of the data used as predictor. This must have dimension 4
        (time, channels/variables, lon, lat).

    y_shape : tuple
        Shape of the data used as predictand. This must have dimension 2
        (time, gridpoint)

    filters_last_conv : int
        Number of filters/kernels of the last convolutional layer

    stochastic: bool
        If set to True, the model is composed of three final dense layers computing
        the p, shape and scale of the Bernoulli-gamma distribution. Otherwise,
        the models is composed of one final layer computing the values.
    """


    def __init__(self, x_shape: tuple, y_shape: tuple,
                 filters_last_conv: int, stochastic: bool):

        super(DeepESDElevationPr, self).__init__()

        if (len(x_shape) != 4) or (len(y_shape) != 2):
            error_msg =\
            'X and Y data must have a dimension of length 4'
            'and 2, correspondingly'

            raise ValueError(error_msg)

        self.x_shape = x_shape
        self.y_shape = y_shape
        self.filters_last_conv = filters_last_conv
        self.stochastic = stochastic

        self.conv_1 = torch.nn.Conv2d(in_channels=self.x_shape[1],
                                      out_channels=50,
                                      kernel_size=3,
                                      padding=1)

        self.conv_2 = torch.nn.Conv2d(in_channels=50,
                                      out_channels=25,
                                      kernel_size=3,
                                      padding=1)

        self.conv_3 = torch.nn.Conv2d(in_channels=25,
                                      out_channels=self.filters_last_conv,
                                      kernel_size=3,
                                      padding=1)

        if self.stochastic:
            self.p = torch.nn.Linear(in_features=\
                                     self.x_shape[2] * self.x_shape[3] * self.filters_last_conv,
                                     out_features=self.y_shape[1])

            self.log_shape = torch.nn.Linear(in_features=\
                                             self.x_shape[2] * self.x_shape[3] * self.filters_last_conv,
                                             out_features=self.y_shape[1])

            self.log_scale = torch.nn.Linear(in_features=\
                                             self.x_shape[2] * self.x_shape[3] * self.filters_last_conv,
                                             out_features=self.y_shape[1])

        else:
            self.out = torch.nn.Linear(in_features=\
                                       self.x_shape[2] * self.x_shape[3] * self.filters_last_conv,
                                       out_features=self.y_shape[1])

        self.elev_out = torch.nn.Linear(in_features=\
                                        self.x_shape[2] * self.x_shape[3] * self.filters_last_conv,
                                        out_features=self.y_shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv_1(x)
        x = torch.relu(x)

        x = self.conv_2(x)
        x = torch.relu(x)

        x = self.conv_3(x)
        x = torch.relu(x)

        x = torch.flatten(x, start_dim=1)

        if self.stochastic:
            p = self.p(x)
            p = torch.sigmoid(p)

            log_shape = self.log_shape(x)
            log_scale = self.log_scale(x)

            out = torch.cat((p, log_shape, log_scale), dim = 1)
        else:
            out = self.out(x)

        out_elev = self.elev_out(x)

        return out, out_elev

class DeepESDMultivariable(torch.nn.Module):

    """
    DeepESD model adapted to jointly model num_target_vars. This is done
    by replicating the last dense layer for each of these variables. Notice
    that this model is only implemented for stochastic=False, otherwise the
    number of parameters spanned by the last dense layer would be significantly
    large.

    Parameters
    ----------
    x_shape : tuple
        Shape of the data used as predictor. This must have dimension 4
        (time, channels/variables, lon, lat).

    y_shape : tuple
        Shape of the data used as predictand. This must have dimension 2
        (time, gridpoint)

    filters_last_conv : int
        Number of filters/kernels of the last convolutional layer

    num_target_vars: int
        Number of target variables to model, thus number of replcias of the
        last dense layer.
    """

    def __init__(self, x_shape: tuple, y_shape: tuple,
                 filters_last_conv: int, num_target_vars: int):

        super(DeepESDMultivariable, self).__init__()

        if len(x_shape) != 4:
            error_msg =\
            'X data must have a dimension of length 4'
            raise ValueError(error_msg)

        self.x_shape = x_shape
        self.y_shape = y_shape
        self.filters_last_conv = filters_last_conv
        self.num_target_vars = num_target_vars

        self.conv_1 = torch.nn.Conv2d(in_channels=self.x_shape[1],
                                      out_channels=50,
                                      kernel_size=3,
                                      padding=1)

        self.conv_2 = torch.nn.Conv2d(in_channels=50,
                                      out_channels=25,
                                      kernel_size=3,
                                      padding=1)

        self.conv_3 = torch.nn.Conv2d(in_channels=25,
                                      out_channels=self.filters_last_conv,
                                      kernel_size=3,
                                      padding=1)

        self.target_final_layers = []
        for _ in range(self.num_target_vars):
            self.target_final_layers.append(torch.nn.Linear(in_features=\
                                            self.x_shape[2] * self.x_shape[3] * self.filters_last_conv,
                                            out_features=self.y_shape[1]))

        self.target_final_layers = nn.ModuleList(self.target_final_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv_1(x)
        x = torch.relu(x)

        x = self.conv_2(x)
        x = torch.relu(x)

        x = self.conv_3(x)
        x = torch.relu(x)

        x = torch.flatten(x, start_dim=1)

        final_preds = []
        for final_layer in self.target_final_layers:
            final_preds.append(final_layer(x))

        return *final_preds,