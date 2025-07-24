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
from deep4downscaling.datasets import d4d_dataset
from deep4downscaling.datasets import d4d_dataloader
##################################################################################################################################
##################################################################################################################################

def d4dcreate(
    date_init: str,
    date_end: str,
    freq: str,
    data: dict,
    transform: dict,
    output_path: str = "./",
    overwrite: bool = False
):

    print(f"""
    -----------------------------------------------------------------------------------------------------------
    WELCOME TO D4D CREATE DATASET! 📈🤖📊
  
    Date Init: {date_init}
    Date End: {date_end}
    Temporal freq.: {freq}
    Dataset will be saved here: {output_path}
    Overwrite: {overwrite}
    -----------------------------------------------------------------------------------------------------------
    """)  
    
    # Create dirs to store the outputs
    os.makedirs(output_path, exist_ok=True)

    ######## SAVE DATASET TO DISK AS ZARR
    if not os.path.exists(output_path) or overwrite:
      os.makedirs(output_path, exist_ok=True)
      # Training dataset
      d = d4d_dataset(date_init, date_end, freq, data, transform) # Call Init
      d.to_disk(zarr_path = output_path) # Preprocess dataset and save it as a torch dataset for rapid loading in d4d_dataloader


    print("----------------------------------------------------")
    print("✅  🤞 🎯 Dataset(s) created successfully! 🎯  🤞 ✅")



