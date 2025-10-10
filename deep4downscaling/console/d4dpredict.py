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


    ######## LOADER --- OR LOAD DATA FROM A ZARR??!!
    ds = zarr.open(input_data["path"], mode='r')

    num_samples = ds.shape[0]

    dates_test = [ date for date in ds.attrs.get("dates") if int(date[:4]) in input_data["years"] ]
    num_test_samples = len(dates_test)
    test_sample_indices = [ i for i, date in enumerate(ds.attrs.get("dates")) if int(date[:4]) in input_data["years"] ]

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
    
    # Add threshold parameter for specific loss functions
    if metadata["loss"] == "NLLBerGammaLoss":
        # For BerGamma, the threshold should be in metadata
        if "bergamma_threshold" in metadata:
            kwargs["threshold"] = metadata["bergamma_threshold"]
    elif metadata["loss"] == "DualOutputLoss":
        # For DualOutput, use the stored threshold
        if "dual_output_threshold" in metadata:
            kwargs["threshold"] = metadata["dual_output_threshold"]
        if "classification_threshold" in metadata:
            kwargs["classification_threshold"] = metadata["classification_threshold"]

    if "sample" in metadata:
        kwargs["sample"] = metadata["sample"]

    ######## HARDWARE
    # Hardware
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    ######## RUNNER
    # Prediction function
    if metadata["loss"] == "NLLBerGammaLoss":
        pred_func_name = "compute_preds_ber_gamma"
    elif metadata["loss"] == "NLLGaussianLoss":
        pred_func_name = "compute_preds_gaussian"
    elif metadata["loss"] == "DualOutputLoss":
        pred_func_name = "compute_preds_dual_output"
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
    total_batches = (num_test_samples + batch_size - 1) // batch_size

    for batch_num, sample_idx in enumerate(range(test_sample_indices[0], test_sample_indices[-1] +1, batch_size), 1):
        print(f'Processing batch {batch_num}/{total_batches}')
        limit = min(sample_idx + batch_size, test_sample_indices[-1] + 1)
        dates_batch = ds.attrs.get("dates")[sample_idx:(limit)]
            
        # Input data
        x = ds[sample_idx:(limit)].astype(np.float32)   
        x = (x - m) / s
        x = x.astype(np.float32)

        # From numpy array to xarray, using attributes stored in the zarr
        x_input = xr.Dataset(
            data_vars = {
                var_name: (("time", "lat", "lon"), x[:, i, :, :])
                for i, var_name in enumerate(vars)
            },
            coords = {
                "time": dates_batch,
                "lat": np.array(ds.attrs.get("lat")),
                "lon": np.array(ds.attrs.get("lon"))
            }
        )

        # Compute predictions
        pred.append(
            pred_func(x_data=x_input, 
                      model=model,
                      device=device, 
                      batch_size=len(dates_batch),
                      **kwargs)
        )

    print("All batches processed.")
        
    # Concatenate samples along dimension "time"
    pred = xr.concat(pred, dim = "time")
    
    ## Save prediction
    os.makedirs(os.path.dirname(output), exist_ok=True)
    pred.to_netcdf(output) # Save prediction
    print("✅  🤞 🎯 Prediction finished successfully! 🎯  🤞 ✅")
    print(f"✅  🤞 🎯 Prediction saved at: {output}  🎯  🤞 ✅")
