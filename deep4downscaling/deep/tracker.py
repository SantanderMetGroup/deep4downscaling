"""
This module contains the TrainingTracker class for tracking and visualizing
the progress of deep learning model training. It saves loss curves and
prediction samples at configurable intervals to the local filesystem.

This module does not work with models trained using loss functions that do not
return the variable to be predicted (e.g., BernoulliGammaLoss).

Author: Jose González-Abad
"""

import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter

_SPECTRAL_COMPONENT_DEFS = [
    ("Field accuracy",    "train_field_accuracy",    "valid_field_accuracy",    "#1f77b4"),
    ("Field spread",      "train_field_spread",       "valid_field_spread",      "#d62728"),
    ("Spectral accuracy", "train_spectral_accuracy",  "valid_spectral_accuracy", "#2ca02c"),
    ("Spectral spread",   "train_spectral_spread",    "valid_spectral_spread",   "#ff7f0e"),
]


class TrainingTracker:

    """
    Tracker for monitoring training progress of deep learning models. It saves
    loss curves and prediction samples at configurable epoch intervals under
    ``experiment_dir``.

    Parameters
    ----------
    experiment_dir : str
        Base folder for saving tracking artifacts.

    experiment_name : str, optional
        Name for the experiment. If not provided, a timestamp-based ID is
        generated (e.g., exp_20260508_171000).

    log_every : int, optional
        Frequency (in epochs) to save artifacts. By default set to 5.

    num_samples : int, optional
        Number of random samples to track predictions for. By default set
        to 4.

    spatial_mask : np.ndarray, optional
        2D array with 1 for valid spatial positions and 0 for positions to
        fill with NaN. Used to reconstruct the original spatial layout when
        the model output is a 1D vector with NaN positions removed.

    flip_ud : bool, optional
        If True, flip the prediction maps top-to-bottom before plottinga.
        By default False.

    flip_lr : bool, optional
        If True, flip the prediction maps left-to-right before plotting.
        By default False.

    lambda_spectral : float, optional
        Weighting factor for the spectral loss components. When provided,
        spectral columns in component_data.csv are multiplied by this value
        before plotting so all components are on the same scale. A
        ``loss_components.png`` figure styled after the intercomparison
        plotting convention is saved alongside the standard
        ``component_curves.png``.
    """

    def __init__(self, experiment_dir: str, experiment_name: str=None,
                 log_every: int=5, num_samples: int=4,
                 spatial_mask: np.ndarray=None,
                 flip_ud: bool=False, flip_lr: bool=False,
                 lambda_spectral: float=None) -> None:

        self.experiment_dir = os.path.expanduser(experiment_dir)
        self.log_every = log_every
        self.num_samples = num_samples
        self.spatial_mask = spatial_mask
        self.flip_ud = flip_ud
        self.flip_lr = flip_lr
        self.lambda_spectral = lambda_spectral

        # Generate experiment name if not provided
        if experiment_name is None:
            self.experiment_name = f'exp_{time.strftime("%Y%m%d_%H%M%S")}'
        else:
            self.experiment_name = experiment_name

        # Setup local directories
        self.output_dir = os.path.join(self.experiment_dir, self.experiment_name)
        self.loss_dir = os.path.join(self.output_dir, 'loss')
        self.predictions_dir = os.path.join(self.output_dir, 'predictions')

        os.makedirs(self.loss_dir, exist_ok=True)
        os.makedirs(self.predictions_dir, exist_ok=True)

        # Track sample indices (set on first log_epoch call)
        self._sample_indices = None

        print(f'TrainingTracker initialized | Output: {self.output_dir}')

    def _select_sample_indices(self, dataloader: torch.utils.data.DataLoader) -> None:

        """
        Randomly select sample indices from the dataset underlying the
        provided DataLoader.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            DataLoader to sample indices from.
        """

        dataset_size = len(dataloader.dataset)
        n = min(self.num_samples, dataset_size)
        self._sample_indices = np.random.choice(dataset_size, size=n,
                                                replace=False)

    def _save_loss_curves(self, train_loss: list, valid_loss: list=None) -> None:

        """
        Save loss curves as a PNG plot and the raw data as a CSV file.

        Parameters
        ----------
        train_loss : list
            List of training loss values per epoch.

        valid_loss : list, optional
            List of validation loss values per epoch.
        """

        epochs = np.arange(1, len(train_loss) + 1)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, train_loss, label='Training Loss', color='tab:blue')

        if valid_loss is not None and len(valid_loss) > 0:
            ax.plot(epochs, valid_loss, label='Validation Loss',
                    color='tab:orange')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Curves')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(round(x))}'))
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(self.loss_dir, 'loss_curves.png'), dpi=150)
        plt.close(fig)

        # Save CSV
        csv_path = os.path.join(self.loss_dir, 'loss_data.csv')
        with open(csv_path, 'w') as f:
            if valid_loss is not None and len(valid_loss) > 0:
                f.write('epoch,train_loss,valid_loss\n')
                for i, (tl, vl) in enumerate(zip(train_loss, valid_loss)):
                    f.write(f'{i+1},{tl},{vl}\n')
            else:
                f.write('epoch,train_loss\n')
                for i, tl in enumerate(train_loss):
                    f.write(f'{i+1},{tl}\n')

    def _save_component_curves(self, train_components: dict,
                               valid_components: dict=None) -> None:

        """
        Save per-component loss curves as a PNG and CSV. Each component gets
        its own color; train and validation are distinguished by solid vs.
        dashed lines.

        Parameters
        ----------
        train_components : dict
            Mapping from component name to list of training values per epoch.

        valid_components : dict, optional
            Mapping from component name to list of validation values per epoch.
        """

        if not train_components:
            return

        component_names = list(train_components.keys())
        n_epochs = len(next(iter(train_components.values())))
        epochs = np.arange(1, n_epochs + 1)

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        fig, ax = plt.subplots(figsize=(10, 6))

        for idx, name in enumerate(component_names):
            color = colors[idx % len(colors)]
            ax.plot(epochs, train_components[name], label=f'{name} (train)',
                    color=color, linestyle='-', marker='o', markersize=4)
            if valid_components is not None and name in valid_components and \
                    len(valid_components[name]) > 0:
                ax.plot(epochs, valid_components[name],
                        label=f'{name} (valid)', color=color, linestyle='--',
                        marker='o', markersize=4)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss component value')
        ax.set_title('Loss Component Curves')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(round(x))}'))
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(self.loss_dir, 'component_curves.png'), dpi=150)
        plt.close(fig)

        # Save CSV
        has_valid = valid_components is not None and len(valid_components) > 0
        csv_path = os.path.join(self.loss_dir, "component_data.csv")
        with open(csv_path, 'w') as f:
            train_cols = ','.join(f'train_{n}' for n in component_names)
            if has_valid:
                valid_cols = ','.join(f'valid_{n}' for n in component_names)
                f.write(f'epoch,{train_cols},{valid_cols}\n')
            else:
                f.write(f'epoch,{train_cols}\n')
            for i in range(n_epochs):
                train_vals = ','.join(
                    str(train_components[n][i]) for n in component_names)
                if has_valid:
                    valid_vals = ','.join(
                        str(valid_components[n][i]) for n in component_names)
                    f.write(f'{i+1},{train_vals},{valid_vals}\n')
                else:
                    f.write(f'{i+1},{train_vals}\n')

        if self.lambda_spectral is not None:
            self._save_spectral_loss_curves()

    def _save_spectral_loss_curves(self) -> None:

        """
        Read component_data.csv, scale spectral columns by lambda_spectral,
        and save a loss_components.png figure with the same style as the
        intercomparison plot_run function.
        """

        import pandas as pd

        csv_path = os.path.join(self.loss_dir, "component_data.csv")
        df = pd.read_csv(csv_path)
        lam = self.lambda_spectral

        spectral_cols = [c for c in df.columns if "spectral" in c]
        for col in spectral_cols:
            df[col] = df[col] * lam

        fig, ax = plt.subplots(figsize=(10, 6))

        for label, train_col, valid_col, color in _SPECTRAL_COMPONENT_DEFS:
            scaled_label = (f"{label} × λ={lam}"
                            if "Spectral" in label else label)
            if train_col in df.columns:
                ax.plot(df["epoch"], df[train_col],
                        label=f"{scaled_label} — train",
                        color=color, linewidth=1.8)
            if valid_col in df.columns:
                ax.plot(df["epoch"], df[valid_col],
                        label=f"{scaled_label} — valid",
                        color=color, linewidth=1.8, linestyle="--")

        ax.set_xlabel("Epoch", fontweight="bold")
        ax.set_ylabel("Loss", fontweight="bold")
        ax.set_title(
            f"Loss components — {self.experiment_name}"
            f"  (spectral × λ={lam})",
            fontsize=13, fontweight="bold",
        )
        ax.legend(fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(self.loss_dir, "loss_components.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _reshape_to_2d(self, arr: np.ndarray) -> np.ndarray:

        """
        Reshape a 1D array into the closest rectangular 2D shape for
        spatial visualization.

        Parameters
        ----------
        arr : np.ndarray
            1D array to reshape.

        Returns
        -------
        np.ndarray
            2D array with approximate square shape.
        """

        n = arr.shape[0]
        ncols = int(np.ceil(np.sqrt(n)))
        nrows = int(np.ceil(n / ncols))

        padded = np.full(nrows * ncols, np.nan)
        padded[:n] = arr
        return padded.reshape(nrows, ncols, order='C')

    def _reconstruct_from_mask(self, arr: np.ndarray) -> np.ndarray:

        """
        Reconstruct a 2D spatial field from a 1D array of valid-only values
        using the spatial mask provided at initialization.

        Parameters
        ----------
        arr : np.ndarray
            1D array containing values only for valid spatial positions.

        Returns
        -------
        np.ndarray
            2D array with the original spatial layout, NaN at masked
            positions.
        """

        result = np.full(self.spatial_mask.shape, np.nan)
        result[self.spatial_mask == 1] = arr
        return result

    def _save_prediction_samples(self, epoch: int, model: torch.nn.Module,
                                 dataloader: torch.utils.data.DataLoader,
                                 device: str,
                                 mixed_precision: bool=False) -> None:

        """
        Generate and save prediction sample plots for the selected random
        samples. Data is always visualized as 2D spatial fields using imshow.

        Parameters
        ----------
        epoch : int
            Current epoch number (0-indexed).

        model : torch.nn.Module
            The model used to compute predictions.

        dataloader : torch.utils.data.DataLoader
            DataLoader from which the samples are drawn.

        device : str
            Device used for inference (cuda or cpu).

        mixed_precision : bool, optional
            Whether to use automatic mixed precision for inference.
        """

        model.eval()
        dataset = dataloader.dataset

        n_samples = len(self._sample_indices)
        fig, axes = plt.subplots(n_samples, 2, figsize=(10, 4 * n_samples))

        if n_samples == 1:
            axes = axes[np.newaxis, :]

        for i, idx in enumerate(self._sample_indices):
            x, y = dataset[idx]
            x_input = x.unsqueeze(0).to(device)

            with torch.no_grad():
                if mixed_precision:
                    with torch.amp.autocast(device_type=device):
                        pred = model(x_input)
                else:
                    pred = model(x_input)

            # Handle models that return tuples
            if isinstance(pred, tuple):
                pred = pred[0]

            y_np = y.cpu().numpy()
            pred_np = pred.squeeze(0).cpu().numpy()

            # Prepare arrays for 2D visualization
            if y_np.ndim == 1:
                if self.spatial_mask is not None:
                    y_plot = self._reconstruct_from_mask(y_np)
                    pred_plot = self._reconstruct_from_mask(pred_np.flatten())
                else:
                    y_plot = self._reshape_to_2d(y_np)
                    pred_plot = self._reshape_to_2d(pred_np.flatten())
            elif y_np.ndim == 2:
                y_plot = y_np
                pred_plot = pred_np
            elif y_np.ndim == 3:
                y_plot = y_np[0]
                pred_plot = pred_np[0]

            # Apply orientation corrections if requested
            if self.flip_ud:
                y_plot = np.flipud(y_plot)
                pred_plot = np.flipud(pred_plot)
            if self.flip_lr:
                y_plot = np.fliplr(y_plot)
                pred_plot = np.fliplr(pred_plot)

            # Use shared color limits for target and prediction
            vmin = min(np.nanmin(y_plot), np.nanmin(pred_plot))
            vmax = max(np.nanmax(y_plot), np.nanmax(pred_plot))

            im0 = axes[i, 0].imshow(y_plot, aspect='auto', cmap='turbo',
                                    vmin=vmin, vmax=vmax)
            plt.colorbar(im0, ax=axes[i, 0], fraction=0.046)

            im1 = axes[i, 1].imshow(pred_plot, aspect='auto', cmap='turbo',
                                    vmin=vmin, vmax=vmax)
            plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)

            axes[i, 0].set_title(f'Target (sample {idx})')
            axes[i, 1].set_title(f'Prediction (sample {idx})')

        fig.suptitle(f'Epoch {epoch + 1}', fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(os.path.join(self.predictions_dir,
                                 f'epoch_{epoch+1:04d}.png'), dpi=150)
        plt.close(fig)

    def log_epoch(self, epoch: int, train_loss: list,
                  valid_loss: list=None,
                  train_loss_components: dict=None,
                  valid_loss_components: dict=None,
                  model: torch.nn.Module=None,
                  train_data: torch.utils.data.DataLoader=None,
                  valid_data: torch.utils.data.DataLoader=None,
                  device: str='cpu',
                  mixed_precision: bool=False) -> None:

        """
        Log training information for the current epoch. This method saves loss
        curves and, if a model and data are provided, prediction samples.

        Parameters
        ----------
        epoch : int
            Current epoch number (0-indexed).

        train_loss : list
            List of training loss values up to and including the current epoch.

        valid_loss : list, optional
            List of validation loss values up to and including the current
            epoch.

        train_loss_components : dict, optional
            Mapping from component name to list of training values per epoch.
            When provided, a component_curves.png plot is saved alongside the
            main loss_curves.png.

        valid_loss_components : dict, optional
            Mapping from component name to list of validation values per epoch.

        model : torch.nn.Module, optional
            Model to use for generating prediction samples. If not provided,
            prediction samples are not saved.

        train_data : torch.utils.data.DataLoader, optional
            DataLoader with the training data.

        valid_data : torch.utils.data.DataLoader, optional
            DataLoader with the validation data. If provided, samples are
            drawn from here; otherwise from train_data.

        device : str, optional
            Device used for inference. By default 'cpu'.

        mixed_precision : bool, optional
            Whether to use automatic mixed precision for inference.
        """

        # Save loss curves
        self._save_loss_curves(train_loss, valid_loss)

        # Save component curves if available
        if train_loss_components:
            self._save_component_curves(train_loss_components,
                                        valid_loss_components)

        # Save prediction samples
        if model is not None:
            dataloader = valid_data if valid_data is not None else train_data

            if dataloader is not None:
                if self._sample_indices is None:
                    self._select_sample_indices(dataloader)

                self._save_prediction_samples(epoch, model, dataloader,
                                              device, mixed_precision)

    def finalize(self) -> None:

        """
        Finalize the tracker (prints output location; artifacts are already on disk).
        """

        print(f'TrainingTracker finalized | Artifacts saved to: '
              f'{self.output_dir}')
