"""
This module contains the functions for training deep learning models.

Authors:
    Jose González-Abad
    Alfonso Hernanz
"""

import os
import copy
import numpy as np
import time
import math
import torch

def standard_training_loop(model: torch.nn.Module, model_name: str, model_path: str,
                           loss_function: torch.nn.Module, optimizer: torch.optim,
                           num_epochs: int, device: str,
                           train_data: torch.utils.data.dataloader.DataLoader,
                           valid_data: torch.utils.data.dataloader.DataLoader=None,
                           scheduler: torch.optim=None,
                           patience_early_stopping: int=None,
                           mixed_precision: bool=False,
                           clip_gradients_norm: float=None,
                           save_checkpoint_every: int=None,
                           resume_checkpoint: str=None,
                           tracker: 'TrainingTracker'=None) -> dict:
    
    """
    Standard training loop for a DL model in a supervised setting. Besides the
    training, it is possible to perform a validation step and control the saving 
    of the model through an early stopping strategy. To activate the latter, pass
    a value to the argument patience_early_stopping, otherwise the training will 
    continue for the num_epochs specified, saving the model at the end of each
    epoch.

    Parameters
    ----------
    model : torch.nn.Module
        Pytorch model to train

    model_name : str
        Name of the model when saved as
        a .pt file

    model_path : str
        Path of the folder where the model
        will be saved

    loss_function : torch.nn.Module
        Loss function to use when training/evaluating
        the model

    optimizer : torch.optim
        Optimizer to use when training the model

    num_epochs : int
        Number of epochs

    device : str
        Device used to run the training (cuda or cpu)

    train_data : torch.utils.data.dataloader.DataLoader
        DataLoader with the training data

    valid_data : torch.utils.data.dataloader.DataLoader, optional
        DataLoader with the validation data

    scheduler : torch.optim=None, optional
        Scheduler to use for the optimization

    patience_early_stopping : int, optional
        Number of steps allowed for the model to run before
        any improvement in the loss function occurs. If this
        number is surpassd without improvement the training is
        stopped.

    mixed_precision : bool, optional
        If training on GPUs, mixed_precision allows for automatic
        mixed precision training to reduce computation and memory
        footprint. By default this parameter is set to False.

    clip_gradients_norm : float, optional
        Maximum norm of the gradients. If provided, the gradients are clipped
        to avoid exploding gradients.

    save_checkpoint_every : int, optional
        Frequency (in epochs) to save a full training checkpoint containing the
        model weights, optimizer state, scheduler state, and loss history. The
        checkpoint is saved at {model_path}/{model_name}_checkpoint.pt and only
        the most recent one is kept. To resume training with only the model
        weights (e.g., with a different optimizer), load them manually before
        calling this function using torch.load(checkpoint_path)['model'].

    resume_checkpoint : str, optional
        Path to a checkpoint file to resume training from. Restores the full
        training state (model, optimizer, scheduler, scaler, epoch and loss
        history).

    tracker : TrainingTracker, optional
        Training tracker instance for logging during training. If not
        provided, no tracking is performed.

    Returns
    -------
    dict
        Dictionary with list(s) representing the loss function
        across epochs.
    """

    model = model.to(device)

    # The scaler scales the loss to avoid the underflow
    # of gradients
    if mixed_precision:
        scaler = torch.amp.GradScaler()

    # Set the early stopping parameters
    if patience_early_stopping is not None:
        best_val_loss = math.inf
        early_stopping_step = 0

    # Register the losses per epoch
    epoch_train_loss = []
    epoch_valid_loss = []

    start_epoch = 0

    # Resume from checkpoint if provided
    if resume_checkpoint is not None:
        print(f'Resuming training from checkpoint: {resume_checkpoint}')
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None and checkpoint.get('scheduler') is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        if mixed_precision and checkpoint.get('scaler') is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        epoch_train_loss = checkpoint.get('train_loss', [])
        epoch_valid_loss = checkpoint.get('valid_loss', [])
        if patience_early_stopping is not None:
            best_val_loss = checkpoint.get('best_val_loss', math.inf)
        print(f'Resumed from epoch {start_epoch}')

    # Iterate over epochs
    for epoch in range(start_epoch, num_epochs):
        
        epoch_start = time.time()
        epoch_train_loss.append(0)

        # Iterate over batches
        model.train()
        for x, y in train_data:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            if mixed_precision: 
                with torch.amp.autocast(device_type=device):
                    output = model(x)
                    loss = loss_function(target=y, output=output)
                scaler.scale(loss).backward()
                if clip_gradients_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_gradients_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(x)
                loss = loss_function(target=y, output=output)
                loss.backward()               
                if clip_gradients_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_gradients_norm)
                optimizer.step()

            epoch_train_loss[-1] += loss.item()

        # Compute mean loss across the epoch
        epoch_train_loss[-1] = epoch_train_loss[-1] / len(train_data)

        # If valid data is provided, perform a pass through it
        if valid_data is not None:

            epoch_valid_loss.append(0)

            model.eval()
            for x, y in valid_data:

                x = x.to(device)
                y = y.to(device)    

                if mixed_precision: 
                    with torch.amp.autocast(device_type=device):
                        output = model(x)
                        loss = loss_function(target=y, output=output)
                else:
                        output = model(x)
                        loss = loss_function(target=y, output=output)                    

                epoch_valid_loss[-1] += loss.item()

            # Compute mean loss across the epoch
            epoch_valid_loss[-1] = epoch_valid_loss[-1] / len(valid_data)

        epoch_end = time.time()
        epoch_time = np.round(epoch_end - epoch_start, 2)

        # Build log message
        log_msg = f'Epoch {epoch+1} ({epoch_time} secs) | Training Loss {np.round(epoch_train_loss[-1], 4)}'
        if valid_data is not None: 
            log_msg = log_msg + f' Valid Loss {np.round(epoch_valid_loss[-1], 4)}'
        
        # Step the scheduler if provided
        if scheduler is not None:
            # If scheduler is ReduceLROnPlateau, it needs the validation loss
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if valid_data is not None:
                    scheduler.step(epoch_valid_loss[-1])
                else:
                    scheduler.step(epoch_train_loss[-1])
            else:
                scheduler.step()
            
            # Add current learning rate to log message
            current_lr = scheduler.get_last_lr()[0] if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else optimizer.param_groups[0]['lr']
            log_msg = log_msg + f' | LR {current_lr:.2e}'

        # Early stopping logic
        if patience_early_stopping is not None:
            # Save the model if the validation loss improves
            if epoch_valid_loss[-1] < best_val_loss:
                best_val_loss = epoch_valid_loss[-1]
                early_stopping_step = 0
                log_msg = log_msg + ' (Model saved)'
                torch.save(model.state_dict(),
                           os.path.expanduser(f'{model_path}/{model_name}.pt'))
            else:
                early_stopping_step +=1

            # If no improvement over the specified steps, stop the training
            if early_stopping_step >= patience_early_stopping:
                print(log_msg)
                print('***Training finished***')
                break

        else: # If early stopping is not configured save the model at each epoch
            log_msg = log_msg + ' (Model saved)'
            torch.save(model.state_dict(),
                       os.path.expanduser(f'{model_path}/{model_name}.pt'))

        # Save training checkpoint
        if save_checkpoint_every is not None and (epoch + 1) % save_checkpoint_every == 0:
            ckpt_path = os.path.expanduser(f'{model_path}/{model_name}_checkpoint.pt')
            torch.save({'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict() if scheduler is not None else None,
                        'scaler': scaler.state_dict() if mixed_precision else None,
                        'best_val_loss': best_val_loss if patience_early_stopping is not None else None,
                        'train_loss': epoch_train_loss,
                        'valid_loss': epoch_valid_loss}, ckpt_path)
            log_msg = log_msg + ' (Checkpoint saved)'

        # Log to tracker
        if tracker is not None and (epoch + 1) % tracker.log_every == 0:
            tracker.log_epoch(epoch=epoch, train_loss=epoch_train_loss,
                              valid_loss=epoch_valid_loss if valid_data is not None else None,
                              model=model, train_data=train_data,
                              valid_data=valid_data, device=device,
                              mixed_precision=mixed_precision)
        
        # Print log
        print(log_msg)

    # Finalize tracker
    if tracker is not None:
        tracker.finalize()

    # Return loss functions
    if valid_data is not None:
        return epoch_train_loss, epoch_valid_loss
    else:
        return epoch_train_loss, None

def adversarial_training_loop(model: torch.nn.Module, model_name: str, model_path: str,
                               loss_function: torch.nn.Module, optimizer: torch.optim,
                               num_epochs: int, device: str,
                               train_data: torch.utils.data.dataloader.DataLoader,
                               valid_data: torch.utils.data.dataloader.DataLoader=None,
                               scheduler: torch.optim=None,
                               patience_early_stopping: int=None,
                               mixed_precision: bool=False,
                               epsilon: float=0.01,
                               save_checkpoint_every: int=None,
                               resume_checkpoint: str=None,
                               tracker: 'TrainingTracker'=None) -> dict:
    
    """
    Adversarial training loop for Deep Ensembles using FGSM (Fast Gradient Sign Method).
    This implementation follows Lakshminarayanan et al. (2017) where adversarial training
    is used to smooth predictive distributions. For each training batch, an adversarial
    example is generated using FGSM and the model is trained to minimize the sum of losses
    on both the original and adversarial examples.
    
    The adversarial perturbation is computed as:
        x_adv = x + epsilon * sign(∇_x L(θ, x, y))
    
    The optimization objective is:
        Minimize L(θ, x, y) + L(θ, x_adv, y) w.r.t. θ
    
    where L is the loss function, θ are the model parameters, and epsilon controls
    the perturbation magnitude (default 1% of the input range).
    
    Parameters
    ----------
    model : torch.nn.Module
        Pytorch model to train
    
    model_name : str
        Name of the model when saved as
        a .pt file
    
    model_path : str
        Path of the folder where the model
        will be saved
    
    loss_function : torch.nn.Module
        Loss function to use when training/evaluating
        the model
    
    optimizer : torch.optim
        Optimizer to use when training the model
    
    num_epochs : int
        Number of epochs
    
    device : str
        Device used to run the training (cuda or cpu)
    
    train_data : torch.utils.data.dataloader.DataLoader
        DataLoader with the training data
    
    valid_data : torch.utils.data.dataloader.DataLoader, optional
        DataLoader with the validation data
    
    scheduler : torch.optim, optional
        Scheduler to use for the optimization
    
    patience_early_stopping : int, optional
        Number of steps allowed for the model to run before
        any improvement in the loss function occurs. If this
        number is surpassed without improvement the training is
        stopped.
    
    mixed_precision : bool, optional
        If training on GPUs, mixed_precision allows for automatic
        mixed precision training to reduce computation and memory
        footprint. By default this parameter is set to False.
    
    epsilon : float, optional
        Magnitude of adversarial perturbation. Following the Deep Ensembles
        paper, the default is 1% of the input range (0.01). This value assumes
        standardized inputs.

    save_checkpoint_every : int, optional
        Frequency (in epochs) to save a full training checkpoint containing the
        model weights, optimizer state, scheduler state, and loss history. The
        checkpoint is saved at {model_path}/{model_name}_checkpoint.pt and only
        the most recent one is kept. To resume training with only the model
        weights (e.g., with a different optimizer), load them manually before
        calling this function using torch.load(checkpoint_path)['model'].

    resume_checkpoint : str, optional
        Path to a checkpoint file to resume training from. Restores the full
        training state (model, optimizer, scheduler, scaler, epoch and loss
        history).

    tracker : TrainingTracker, optional
        Training tracker instance for logging during training. If not
        provided, no tracking is performed.
    
    Returns
    -------
    dict
        Dictionary with list(s) representing the loss function
        across epochs.
    
    References
    ----------
    Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable
    predictive uncertainty estimation using deep ensembles. In Advances in neural
    information processing systems (pp. 6402-6413).
    """
    
    model = model.to(device)
    
    # The scaler scales the loss to avoid the underflow
    # of gradients
    if mixed_precision:
        scaler = torch.amp.GradScaler()
    
    # Set the early stopping parameters
    if patience_early_stopping is not None:
        best_val_loss = math.inf
        early_stopping_step = 0
    
    # Register the losses per epoch
    epoch_train_loss = []
    epoch_valid_loss = []

    start_epoch = 0

    # Resume from checkpoint if provided
    if resume_checkpoint is not None:
        print(f'Resuming training from checkpoint: {resume_checkpoint}')
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None and checkpoint.get('scheduler') is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        if mixed_precision and checkpoint.get('scaler') is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        epoch_train_loss = checkpoint.get('train_loss', [])
        epoch_valid_loss = checkpoint.get('valid_loss', [])
        if patience_early_stopping is not None:
            best_val_loss = checkpoint.get('best_val_loss', math.inf)
        print(f'Resumed from epoch {start_epoch}')
    
    # Iterate over epochs
    for epoch in range(start_epoch, num_epochs):
        
        epoch_start = time.time()
        epoch_train_loss.append(0)
        
        # Iterate over batches
        model.train()
        for x, y in train_data:
            
            x = x.to(device)
            y = y.to(device)
            
            # Enable gradients w.r.t. input for adversarial example generation
            x.requires_grad = True
            
            optimizer.zero_grad()
            
            if mixed_precision:
                with torch.amp.autocast(device_type=device):
                    # Forward pass on original data to get gradients for FGSM
                    output = model(x)
                    loss = loss_function(target=y, output=output)
                
                # Compute gradients w.r.t. input for FGSM
                scaler.scale(loss).backward()
                
                # Generate adversarial example using FGSM
                with torch.no_grad():
                    # Get sign of gradient
                    grad_sign = x.grad.sign()
                    # Create adversarial example
                    x_adv = x + epsilon * grad_sign
                    x_adv = x_adv.detach()
                    x_adv.requires_grad = False
                
                # Zero gradients before computing total loss (sum of both losses)
                optimizer.zero_grad()
                
                # Forward pass on both original and adversarial examples
                with torch.amp.autocast(device_type=device):
                    output = model(x)
                    loss = loss_function(target=y, output=output)
                    
                    output_adv = model(x_adv)
                    loss_adv = loss_function(target=y, output=output_adv)
                    
                    # Total loss: sum of original and adversarial losses
                    # Following Algorithm 1: Minimize ℓ(θ_m, x_n_m, y_n_m) + ℓ(θ_m, x'_n_m, y_n_m)
                    total_loss = loss + loss_adv
                
                # Backward pass and optimization step on total loss
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
            else:
                # Forward pass on original data to get gradients for FGSM
                output = model(x)
                loss = loss_function(target=y, output=output)
                
                # Compute gradients w.r.t. input for FGSM
                loss.backward()
                
                # Generate adversarial example using FGSM
                with torch.no_grad():
                    # Get sign of gradient
                    grad_sign = x.grad.sign()
                    # Create adversarial example
                    x_adv = x + epsilon * grad_sign
                    x_adv = x_adv.detach()
                    x_adv.requires_grad = False
                
                # Zero gradients before computing total loss (sum of both losses)
                optimizer.zero_grad()
                
                # Forward pass on both original and adversarial examples
                output = model(x)
                loss = loss_function(target=y, output=output)
                
                output_adv = model(x_adv)
                loss_adv = loss_function(target=y, output=output_adv)
                
                # Total loss: sum of original and adversarial losses
                total_loss = loss + loss_adv
                
                # Backward pass and optimization step on total loss
                total_loss.backward()
                optimizer.step()
            
            # Accumulate total loss for tracking
            epoch_train_loss[-1] += total_loss.item()
        
        # Compute mean loss across the epoch
        epoch_train_loss[-1] = epoch_train_loss[-1] / len(train_data)
        
        # If valid data is provided, perform a pass through it
        if valid_data is not None:
            
            epoch_valid_loss.append(0)
            
            model.eval()
            for x, y in valid_data:
                
                x = x.to(device)
                y = y.to(device)
                
                if mixed_precision:
                    with torch.amp.autocast(device_type=device):
                        output = model(x)
                        loss = loss_function(target=y, output=output)
                else:
                    output = model(x)
                    loss = loss_function(target=y, output=output)
                
                epoch_valid_loss[-1] += loss.item()
            
            # Compute mean loss across the epoch
            epoch_valid_loss[-1] = epoch_valid_loss[-1] / len(valid_data)
        
        epoch_end = time.time()
        epoch_time = np.round(epoch_end - epoch_start, 2)
        
        # Build log message
        log_msg = f'Epoch {epoch+1} ({epoch_time} secs) | Training Loss {np.round(epoch_train_loss[-1], 4)}'
        if valid_data is not None:
            log_msg = log_msg + f' Valid Loss {np.round(epoch_valid_loss[-1], 4)}'
        
        # Step the scheduler if provided
        if scheduler is not None:
            # If scheduler is ReduceLROnPlateau, it needs the validation loss
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if valid_data is not None:
                    scheduler.step(epoch_valid_loss[-1])
                else:
                    scheduler.step(epoch_train_loss[-1])
            else:
                scheduler.step()
            
            # Add current learning rate to log message
            current_lr = scheduler.get_last_lr()[0] if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else optimizer.param_groups[0]['lr']
            log_msg = log_msg + f' | LR {current_lr:.2e}'
        
        # Early stopping logic
        if patience_early_stopping is not None:
            # Save the model if the validation loss improves
            if epoch_valid_loss[-1] < best_val_loss:
                best_val_loss = epoch_valid_loss[-1]
                early_stopping_step = 0
                log_msg = log_msg + ' (Model saved)'
                torch.save(model.state_dict(),
                           os.path.expanduser(f'{model_path}/{model_name}.pt'))
            else:
                early_stopping_step +=1
            
            # If no improvement over the specified steps, stop the training
            if early_stopping_step >= patience_early_stopping:
                print(log_msg)
                print('***Training finished***')
                break
        
        else: # If early stopping is not configured save the model at each epoch
            log_msg = log_msg + ' (Model saved)'
            torch.save(model.state_dict(),
                       os.path.expanduser(f'{model_path}/{model_name}.pt'))

        # Save training checkpoint
        if save_checkpoint_every is not None and (epoch + 1) % save_checkpoint_every == 0:
            ckpt_path = os.path.expanduser(f'{model_path}/{model_name}_checkpoint.pt')
            torch.save({'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict() if scheduler is not None else None,
                        'scaler': scaler.state_dict() if mixed_precision else None,
                        'best_val_loss': best_val_loss if patience_early_stopping is not None else None,
                        'train_loss': epoch_train_loss,
                        'valid_loss': epoch_valid_loss}, ckpt_path)
            log_msg = log_msg + ' (Checkpoint saved)'

        # Log to tracker
        if tracker is not None and (epoch + 1) % tracker.log_every == 0:
            tracker.log_epoch(epoch=epoch, train_loss=epoch_train_loss,
                              valid_loss=epoch_valid_loss if valid_data is not None else None,
                              model=model, train_data=train_data,
                              valid_data=valid_data, device=device,
                              mixed_precision=mixed_precision)
        
        # Print log
        print(log_msg)

    # Finalize tracker
    if tracker is not None:
        tracker.finalize()
    
    # Return loss functions
    if valid_data is not None:
        return epoch_train_loss, epoch_valid_loss
    else:
        return epoch_train_loss, None


def standard_cgan_training_loop(generator: torch.nn.Module,
                                discriminator: torch.nn.Module,
                                gen_name: str,
                                disc_name: str,
                                model_path: str,
                                loss_function: torch.nn.Module,
                                optimizer_G: torch.optim.Optimizer,
                                optimizer_D: torch.optim.Optimizer,
                                num_epochs: int,
                                device: str,
                                train_data: torch.utils.data.DataLoader,
                                valid_data: torch.utils.data.DataLoader=None,
                                lambda_adv: float=1.0,
                                lambda_recon: float=1.0,
                                freq_train_gen: int=1,
                                freq_train_disc: int=1,
                                scheduler: torch.optim.lr_scheduler._LRScheduler=None,
                                mixed_precision: bool=False,
                                save_checkpoint_every: int=None,
                                resume_checkpoint: str=None,
                                tracker: 'TrainingTracker'=None) -> dict:
    """
    Adversarial training loop for standard cGANs (BCE-based adversarial loss function) with
    extended control and checkpoint options.

    Parameters
    ----------
    generator : torch.nn.Module
        Generator model.

    discriminator : torch.nn.Module
        Discriminator model.

    gen_name : str
        Name for saving the generator model file.

    disc_name : str
        Name for saving the discriminator model file.

    model_path : str
        Path where models will be saved.

    loss_function : torch.nn.Module
        Reconstruction loss function (e.g., MSE).

    optimizer_G : torch.optim.Optimizer
        Optimizer for the generator.

    optimizer_D : torch.optim.Optimizer
        Optimizer for the discriminator.

    num_epochs : int
        Number of training epochs.

    device : str
        Device used for training ('cuda' or 'cpu').

    train_data : torch.utils.data.DataLoader
        DataLoader providing (X, Y_real) pairs for training.

    valid_data : torch.utils.data.DataLoader, optional
        DataLoader providing validation data.

    lambda_adv : float, optional
        Weight for the adversarial loss term in generator training.

    lambda_recon : float, optional
        Weight for the reconstruction loss term in generator training.

    freq_train_gen : int, optional
        Frequency to train generator.

    freq_train_disc : int, optional
        Frequency to train discriminator.

    scheduler : torch.optim.lr_scheduler, optional
        Scheduler to adjust the learning rate.

    mixed_precision : bool, optional
        If True, enables automatic mixed precision training.

    save_checkpoint_every : int, optional
        Frequency (in epochs) to save a full training checkpoint containing
        the generator and discriminator weights, optimizer states, scheduler
        state, and loss history. The checkpoint is saved at
        {model_path}/{gen_name}_checkpoint.pt and only the most recent one
        is kept. To resume training with only the model weights (e.g., with
        a different optimizer), load them manually before calling this
        function using torch.load(checkpoint_path)['generator'] and
        torch.load(checkpoint_path)['discriminator'].

    resume_checkpoint : str, optional
        Path to a checkpoint file to resume training from. Restores the full
        training state (generator, discriminator, optimizers, scheduler,
        scalers, epoch and loss history).

    tracker : TrainingTracker, optional
        Training tracker instance for logging during training. If not
        provided, no tracking is performed.
    """

    os.makedirs(model_path, exist_ok=True)

    generator.to(device)
    discriminator.to(device)

    # Loss functions
    adversarial_criterion = torch.nn.BCELoss()
    recon_criterion = loss_function

    # Mixed precision setup
    if mixed_precision:
        scaler_G = torch.amp.GradScaler()
        scaler_D = torch.amp.GradScaler()

    start_epoch = 0
    best_val_loss = math.inf

    # Epoch tracking
    epoch_G_loss, epoch_D_loss, epoch_val_loss = [], [], []
    epoch_adv_loss, epoch_recon_loss = [], []
    epoch_loss_D_real, epoch_loss_D_fake = [], []

    # Resume from checkpoint if provided
    if resume_checkpoint is not None:
        print(f'Resuming training from checkpoint: {resume_checkpoint}')
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        generator.load_state_dict(checkpoint['generator'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        if scheduler is not None and checkpoint.get('scheduler') is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        if mixed_precision:
            if checkpoint.get('scaler_G') is not None:
                scaler_G.load_state_dict(checkpoint['scaler_G'])
            if checkpoint.get('scaler_D') is not None:
                scaler_D.load_state_dict(checkpoint['scaler_D'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', math.inf)
        train_losses = checkpoint.get('train_losses', {})
        epoch_G_loss = train_losses.get('train_G', [])
        epoch_D_loss = train_losses.get('train_D', [])
        epoch_adv_loss = train_losses.get('adv_loss', [])
        epoch_recon_loss = train_losses.get('recon_loss', [])
        epoch_loss_D_real = train_losses.get('loss_D_real', [])
        epoch_loss_D_fake = train_losses.get('loss_D_fake', [])
        epoch_val_loss = train_losses.get('valid', [])
        print(f'Resumed from epoch {start_epoch}')

    # Main training loop
    for epoch in range(start_epoch, num_epochs):

        epoch_start = time.time()
        generator.train()
        discriminator.train()

        running_G_loss, running_D_loss = [], []
        running_adv_loss, running_recon_loss = [], []
        running_loss_D_real, running_loss_D_fake = [], []

        for i, (X, Y_real) in enumerate(train_data):
            X, Y_real = X.to(device), Y_real.to(device)
            batch_size = X.size(0)

            real_labels = torch.ones((batch_size, 1), device=device)
            fake_labels = torch.zeros((batch_size, 1), device=device)

            # Train Discriminator
            if epoch % freq_train_disc == 0:
                optimizer_D.zero_grad()

                if mixed_precision:
                    with torch.amp.autocast(device_type=device):
                        pred_real = discriminator(X, Y_real)
                        loss_D_real = adversarial_criterion(pred_real, real_labels)
                        Y_fake = generator(X).detach()
                        pred_fake = discriminator(X, Y_fake)
                        loss_D_fake = adversarial_criterion(pred_fake, fake_labels)
                        loss_D = 0.5 * (loss_D_real + loss_D_fake)

                    scaler_D.scale(loss_D).backward()
                    scaler_D.step(optimizer_D)
                    scaler_D.update()
                else:
                    pred_real = discriminator(X, Y_real)
                    loss_D_real = adversarial_criterion(pred_real, real_labels)
                    Y_fake = generator(X).detach()
                    pred_fake = discriminator(X, Y_fake)
                    loss_D_fake = adversarial_criterion(pred_fake, fake_labels)
                    loss_D = 0.5 * (loss_D_real + loss_D_fake)
                    loss_D.backward()
                    optimizer_D.step()

                running_D_loss.append(loss_D.item())
                running_loss_D_real.append(loss_D_real.item())
                running_loss_D_fake.append(loss_D_fake.item())

            # Train Generator
            if epoch % freq_train_gen == 0:
                optimizer_G.zero_grad()

                if mixed_precision:
                    with torch.amp.autocast(device_type=device):
                        Y_fake = generator(X)
                        pred_fake = discriminator(X, Y_fake)
                        loss_adv = adversarial_criterion(pred_fake, real_labels)
                        loss_recon = recon_criterion(Y_real, Y_fake)
                        loss_G = lambda_recon * loss_recon + lambda_adv * loss_adv

                    scaler_G.scale(loss_G).backward()
                    scaler_G.step(optimizer_G)
                    scaler_G.update()
                else:
                    Y_fake = generator(X)
                    pred_fake = discriminator(X, Y_fake)
                    loss_adv = adversarial_criterion(pred_fake, real_labels)
                    loss_recon = recon_criterion(Y_real, Y_fake)
                    loss_G = lambda_recon * loss_recon + lambda_adv * loss_adv
                    loss_G.backward()
                    optimizer_G.step()

                running_G_loss.append(loss_G.item())
                running_adv_loss.append(loss_adv.item())
                running_recon_loss.append(loss_recon.item())

        # Compute mean epoch losses
        epoch_G_loss.append(float(np.mean(running_G_loss)))
        epoch_D_loss.append(float(np.mean(running_D_loss)))
        epoch_adv_loss.append(float(np.mean(running_adv_loss)))
        epoch_recon_loss.append(float(np.mean(running_recon_loss)))
        epoch_loss_D_real.append(float(np.mean(running_loss_D_real)))
        epoch_loss_D_fake.append(float(np.mean(running_loss_D_fake)))

        # Validation (reconstruction only)
        if valid_data is not None:
            generator.eval()
            val_loss = 0.0
            with torch.no_grad():
                for Xv, Yv in valid_data:
                    Xv, Yv = Xv.to(device), Yv.to(device)
                    if mixed_precision:
                        with torch.amp.autocast(device_type=device):
                            Y_pred = generator(Xv)
                            val_loss += recon_criterion(Yv, Y_pred).item()
                    else:
                        Y_pred = generator(Xv)
                        val_loss += recon_criterion(Yv, Y_pred).item()
            val_loss /= len(valid_data)
            epoch_val_loss.append(val_loss)
        else:
            val_loss = None

        epoch_end = time.time()
        epoch_time = np.round(epoch_end - epoch_start, 2)

        # Scheduler update
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss if val_loss is not None else epoch_G_loss[-1])
            else:
                scheduler.step()
            current_lr = (scheduler.get_last_lr()[0]
                          if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
                          else optimizer_G.param_groups[0]['lr'])
        else:
            current_lr = optimizer_G.param_groups[0]['lr']


        log_msg = (f"Epoch {epoch+1} ({epoch_time}s) | "
                   f"Loss_D {np.round(epoch_D_loss[-1], 4)} | "
                   f"Loss_G {np.round(epoch_G_loss[-1], 4)}")

        if val_loss is not None:
            log_msg += f" | Val_Loss {np.round(val_loss, 4)}"

        log_msg += (f" | LR {current_lr:.2e}"
                    f" | avd_loss {np.round(epoch_adv_loss[-1], 4)}"
                    f" | recon_loss {np.round(epoch_recon_loss[-1], 4)}")

        # Model saving
        log_msg += " (Model saved)"
        torch.save(generator.state_dict(), os.path.join(model_path, f"{gen_name}.pt"))
        torch.save(discriminator.state_dict(), os.path.join(model_path, f"{disc_name}.pt"))

        # Save training checkpoint
        if save_checkpoint_every is not None and (epoch + 1) % save_checkpoint_every == 0:
            ckpt_path = os.path.join(model_path, f"{gen_name}_checkpoint.pt")
            torch.save({'epoch': epoch,
                        'generator': generator.state_dict(),
                        'discriminator': discriminator.state_dict(),
                        'optimizer_G': optimizer_G.state_dict(),
                        'optimizer_D': optimizer_D.state_dict(),
                        'scheduler': scheduler.state_dict() if scheduler is not None else None,
                        'scaler_G': scaler_G.state_dict() if mixed_precision else None,
                        'scaler_D': scaler_D.state_dict() if mixed_precision else None,
                        'best_val_loss': best_val_loss,
                        'train_losses': {'train_G': epoch_G_loss,
                                         'train_D': epoch_D_loss,
                                         'adv_loss': epoch_adv_loss,
                                         'recon_loss': epoch_recon_loss,
                                         'loss_D_real': epoch_loss_D_real,
                                         'loss_D_fake': epoch_loss_D_fake,
                                         'valid': epoch_val_loss}}, ckpt_path)
            log_msg += ' (Checkpoint saved)'

        # Log to tracker
        if tracker is not None and (epoch + 1) % tracker.log_every == 0:
            tracker.log_epoch(epoch=epoch, train_loss=epoch_G_loss,
                              valid_loss=epoch_val_loss if valid_data is not None else None,
                              model=generator, train_data=train_data,
                              valid_data=valid_data, device=device,
                              mixed_precision=mixed_precision)

        print(log_msg)

    # Finalize tracker
    if tracker is not None:
        tracker.finalize()

    # Return losses
    return {"train_G": epoch_G_loss,
            "train_D": epoch_D_loss,
            "avd_loss": epoch_adv_loss,
            "recon_loss": epoch_recon_loss,
            "loss_D_real": epoch_loss_D_real,
            "loss_D_fake": epoch_loss_D_fake,
            "valid": epoch_val_loss if valid_data is not None else None}