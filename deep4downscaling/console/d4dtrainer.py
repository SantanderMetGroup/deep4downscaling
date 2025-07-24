## Load libraries
import os
import sys
import yaml
import string
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import torch
import importlib
from torch.utils.data import DataLoader, random_split

## Deep4downscaling
import deep4downscaling as d4d
from deep4downscaling.datasets import d4d_dataloader

##################################################################################################################################
##################################################################################################################################

def d4dtrainer(
    data: dict,
    dataloader: dict,
    loss_params: dict,
    model_params: dict,
    training: dict,
    run_ID: int = None,
    output_path: str = "./"
):



    ######## INIT
    # Assign random run_ID if not provided.
    if run_ID is None:
      run_ID = ''.join(random.choices(string.ascii_letters + string.digits, k=5))

    # Create dirs to store the outputs
    id_path = os.path.abspath(f"{output_path}/{run_ID}")
    model_path = f"{id_path}/models/"
    os.makedirs(model_path, exist_ok=True)

    print(f"""
    -----------------------------------------------------------------------------------------------------------
    WELCOME TO D4D TRAINING MODULE! 📈🤖📊
  
    Model: {model_params["name"]}
    Run ID: {run_ID}
    Model will be saved here: {model_path}
    -----------------------------------------------------------------------------------------------------------
    """)  
    
    # Empty dictionary for metadata purposes
    metadata = {}
    
    


    ######## DATASET
    ## Create training and validation PYTORCH datasets/objects
    train_dataset = d4d_dataloader(path_predictors = data["training"]["predictors"]["paths"], 
                                  path_predictands = data["training"]["predictands"]["paths"], 
                                  variables_predictors = data["training"]["predictors"]["variables"],
                                  variables_predictands = data["training"]["predictands"]["variables"],
                                  standardize_predictors = data["training"]["predictors"]["standardize"],
                                  years_predictors = data["training"]["predictors"]["years"],
                                  years_predictands = data["training"]["predictands"]["years"])
    m, s = train_dataset.get_stats()
    metadata["mean"] = np.array(m.squeeze()).tolist()
    metadata["std"] = np.array(s.squeeze()).tolist()

    valid_dataset = d4d_dataloader(path_predictors = data["validation"]["predictors"]["paths"], 
                                  path_predictands = data["validation"]["predictands"]["paths"], 
                                  variables_predictors = data["validation"]["predictors"]["variables"],
                                  variables_predictands = data["validation"]["predictands"]["variables"],
                                  standardize_predictors = data["training"]["predictors"]["standardize"],
                                  years_predictors = data["validation"]["predictors"]["years"],
                                  years_predictands = data["validation"]["predictands"]["years"])




    ######## DATA LOADER
    ## Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size = dataloader["batch_size"], shuffle = dataloader["shuffle"], num_workers = dataloader["num_workers"])
    valid_dataloader = DataLoader(valid_dataset, batch_size = dataloader["batch_size"], shuffle = dataloader["shuffle"], num_workers = dataloader["num_workers"])





    ######## LOSS
    loss_name = loss_params["name"]  # e.g., "NLLBerGammaLoss"
    module = importlib.import_module("deep4downscaling.deep.loss") # Dynamically import from module
    loss_func = getattr(module, loss_name)
    # Parameters
    kwargs = {
           k: v for k, v in loss_params.items()
           if k not in ["name"]
         }
    loss_function = loss_func(**kwargs)
    # Update metadata dictionary
    metadata["loss"] = loss_name





    ######## MODEL
    template_x, template_y = train_dataset.__getitem__(idx=0)
    model_name = model_params["name"]  # e.g., "DeepESDpr"
    module = importlib.import_module("deep4downscaling.deep.models") # Dynamically import from module
    model_func = getattr(module, model_name)
    # Parameters
    kwargs = {
           k: v for k, v in model_params.items()
           if k not in ["name"]
         }
    print(template_x.unsqueeze(0).shape)
    print(template_y.unsqueeze(0).shape)
    model = model_func(x_shape=template_x.unsqueeze(0).shape , y_shape=template_y.unsqueeze(0).shape , **kwargs)

    # Update metadata dictionary
    metadata["architecture"] = model_name
    metadata["model_parameters"] = {
        "x_shape": np.array(template_x.unsqueeze(0).shape ).tolist(),
        "y_shape": np.array(template_y.unsqueeze(0).shape ).tolist(),
        **kwargs  
    }
    del template_x, template_y
    # print(model)

    print(f"""
    MODEL has the following characteristics:
      Architecture: {model_name}
      Loss: {loss_name}
      Model Name: {run_ID}
      Model Path: {output_path}
      Number of parameters: {sum(p.numel() for p in model.parameters())}
      Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}
    """)  





    ####### METADATA
    # When ready, write the metadata to a file
    metadata["var_target"] = data["training"]["predictands"]["variables"]
    with open(f"{id_path}/metadata.yaml", "w") as f:
      yaml.safe_dump(metadata, f, default_flow_style=False)

    



    ######## TRAINING
    ## Hardware
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"""
    TRAINING with the following characteristics:
      Device: {device}
      Learning rate: {training["learning_rate"]}
      Number of epochs: {training["num_epochs"]}
      Patience early stopping: {training["patience_early_stopping"]}
      Batch size: {dataloader["batch_size"]}
    """)  
    ## Train...
    train_loss, val_loss = d4d.deep.train.standard_training_loop(
                              model=model, 
                              model_name=run_ID, 
                              model_path=model_path,
                              device=device,
                              num_epochs=training["num_epochs"],
                              loss_function=loss_function, 
                              optimizer=torch.optim.Adam(model.parameters(), lr=training["learning_rate"]),
                              train_data=train_dataloader, 
                              valid_data=valid_dataloader,
                              patience_early_stopping=training["patience_early_stopping"])
    print("----------------------------------------------------")
    print("✅  🤞 🎯 Training finished successfully! 🎯  🤞 ✅")