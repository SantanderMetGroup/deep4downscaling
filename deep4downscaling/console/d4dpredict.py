## Load libraries
import os
import sys
import yaml
import zarr
import string
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import torch
import importlib

## Deep4downscaling
import deep4downscaling as d4d
from deep4downscaling.trans import compute_valid_mask


##################################################################################################################################
##################################################################################################################################

def read_metadata_from_yaml(yaml_path: str) -> dict:
    """
    Reads a YAML file and returns its content as a dictionary.

    Parameters
    ----------
    yaml_path : str
        Path to the YAML file.

    Returns
    -------
    dict
        Contents of the YAML file.
    """
    with open(f"{yaml_path}", "r") as f:
        metadata = yaml.safe_load(f)
    return metadata


def d4dpredict(
    input_data: dict,
    model_path: str,
    metadata_yaml: str, 
    template_path: str,
    output: str,
    ensemble_size: int = 1,
    batch_size: int = 1,
    ref_data: str = None, # for bias correction
    kwargs: dict = {}, # for bias correction
):

    if ref_data is None:
        ref_data = input_data


    print(f"""
    -----------------------------------------------------------------------------------------------------------
    WELCOME TO D4D PREDICTION MODULE! 📈🤖📊

    Model: {model_path}
    Metadata: {metadata_yaml}
    Prediction(s) will be saved here: {output}
    -----------------------------------------------------------------------------------------------------------
    """) 





    ######## PARSING METADATA
    metadata = read_metadata_from_yaml(metadata_yaml)
    # print(f"""
    # From METADATA YAML FILE obtained the following information:
    # {metadata}
    # """) 


    ######## LOADER --- OR LOAD DATA FROM A ZARR??!!
    ds = zarr.open(input_data["path"], mode='r')
    # z_ref = zarr.open(ref_data, mode='r')
    samples_idx = ds.shape[0]
    dates_test = [ date for date in ds.attrs.get("dates") if int(date[:4]) in input_data["years"] ]

    if input_data["variables"] is None:
        vars = ds.attrs.get("variables")


    ######## LOAD MODEL
    # Load the model weights into the DeepESD architecture
    model_architecture = metadata["architecture"]
    module = importlib.import_module("deep4downscaling.deep.models") # Dynamically import from module
    model_func = getattr(module, model_architecture)
    model = model_func(**metadata["model_parameters"])
    model.load_state_dict(torch.load(model_path))


    # ######## KWARGS: Template, spatial_dims,..
    kwargs["var_target"] = metadata["var_target"]

    template = xr.open_dataset(template_path)[kwargs["var_target"]]
    kwargs["mask"] = compute_valid_mask(template)
    
    kwargs["spatial_dims"] = ["lat", "lon"]
    if "x" in kwargs["mask"].dims and "y" in kwargs["mask"].dims:
      kwargs["spatial_dims"] = ["y", "x"]


    ######## HARDWARE
    # Hardware
    device = ('cuda' if torch.cuda.is_available() else 'cpu')


    ######## RUNNER
    # Prediction function
    if metadata["loss"] == "NLLBerGammaLoss":
        pred_func_name = "compute_preds_ber_gamma"
    elif metadata["loss"] == "NLLGaussianLoss":
        pred_func_name = "compute_preds_gaussian"
    else:
        pred_func_name = "compute_preds_standard"

    module = importlib.import_module("deep4downscaling.deep.pred") # Dynamically import from module
    pred_func = getattr(module, pred_func_name)

    print(f"""
    Using the following PREDICTION RUNNER: {pred_func_name}
    Device: {device}
    Batch size: {batch_size}
    Ensemble size: {ensemble_size}
    """)  


    ######## STATISTICS
    num_vars = ds.shape[1]
    m = np.array(metadata["mean"]).reshape(num_vars, 1, 1)
    s = np.array(metadata["std"]).reshape(num_vars, 1, 1)

    ######## PREDICT
    pred = []
    i = 0
    for sample_idx in range(samples_idx):
        date = ds.attrs.get("dates")[sample_idx]
        if date in dates_test:
            print(f"Inference for sample: {date} ---- ({i}/{len(dates_test)})")
            
            # Input data
            x = ds[sample_idx].astype(np.float32)   
            x = (x - m) / s
            x = x.astype(np.float32)

            # From numpy array to xarray, using attributes stored in the zarr
            x_input = xr.Dataset(
                data_vars = {
                    var_name: (("lat", "lon"), x[i, :, :])
                    for i, var_name in enumerate(vars)
                },
                coords = {
                    "lat": np.array(ds.attrs.get("lat")),
                    "lon": np.array(ds.attrs.get("lon"))
                }
            )
            x_input = x_input.expand_dims(time=[date])

            # Compute predictions
            pred.append(
                pred_func(x_data=x_input, 
                        model=model,
                        device=device, 
                        batch_size=1,
                        ensemble_size=ensemble_size, # It is possible to sample multiple times from the conditional distribution
                        **kwargs)
            )

            i = i + 1
        
    ## Concatenate samples along dimension "time"
    pred = xr.concat(pred, dim = "time")
    print(pred)
    
    ## Save prediction
    os.makedirs(os.path.dirname(output), exist_ok=True)
    pred.to_netcdf(output) # Save prediction
    print("✅  🤞 🎯 Prediction finished successfully! 🎯  🤞 ✅")
    print(f"✅  🤞 🎯 Prediction saved at: {output}  🎯  🤞 ✅")