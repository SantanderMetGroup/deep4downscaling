import os
import glob
import importlib
import numpy as np
import pandas as pd
import xarray as xr 
import calendar
import torch
import zarr
import numcodecs

from torch import from_numpy
from torch.utils.data import Dataset

from deep4downscaling.trans import xarray_to_numpy
from deep4downscaling.trans import compute_valid_mask


########################################################################################################
class d4d_dataset(Dataset):
  def __init__(self, date_init, date_end, freq, data, transform):
    
    # General info
    self.data = data
    self.transform = transform
    
    self.compute_mask = data["compute_mask"] 

    # Number of variables
    self.variables = data["vars"] 
    self.num_vars = len(self.variables)

    # Get sources of netcdf files
    raw_sources = self.data["paths"]
    self.source_files = []
    for source in raw_sources:
        if "*" in source:
            matched_files = sorted(glob.glob(source))
            if not matched_files:
                print(f"⚠️ Warning: No files matched pattern: {source}")
            self.source_files.extend(matched_files)
        elif os.path.isfile(source):
            self.source_files.append(source)
        else:
            print(f"⚠️ Warning: File not found: {source}")
    self.num_sources = len(self.source_files)
    print(f"✅ Found {self.num_sources} source NetCDF files.")

    # Number of samples
    self.date_init = pd.to_datetime(date_init)
    self.date_end = pd.to_datetime(date_end)
    self.freq = freq
    self.dates = pd.date_range(start=self.date_init, end=self.date_end, freq=self.freq).to_numpy()
    dates_as_str = np.array([np.datetime_as_string(d, unit='s') for d in self.dates])

    available_dates = self.get_available_dates_in_sources(self.source_files)
    self.dates = np.array([d for d in self.dates if d in available_dates])
    self.num_samples = len(self.dates)

    # Spatial dims original
    temp = xr.open_dataset(self.source_files[0])
    self.coords = temp.coords
    vars_target = [v for v in temp.data_vars if v in self.variables]
    temp = temp[vars_target]
    self.spatial_dims = ["lat", "lon"]
    if "x" in temp.dims and "y" in temp.dims:
        self.spatial_dims = ["y", "x"]  
    
    self.num_spatial_dims = [len( temp[self.spatial_dims[0]] ), len( temp[self.spatial_dims[1]] )]

    # Apply transforms to a sample to get spatial dims transformed later
    self.transformed = False
    for t_params in self.transform:
        if t_params.get("transformation_func"):
            self.transformed = True
            module = importlib.import_module(t_params["module"])
            func_name = t_params["transformation_func"]
            func = getattr(module, func_name)
            temp = func(temp, **{k: v for k, v in t_params.items() if k not in ["transformation_func", "module"]})
        
    # Spatial dims transformed
    if self.transformed:
        self.num_spatial_dims_transformed = [len(temp.gridpoint)]
        self.spatial_dims_transformed = ["gridpoint"]
    else:
        self.num_spatial_dims_transformed = self.num_spatial_dims

    del temp


  def get_available_dates_in_sources(self, paths):
    available_dates = []
    for p in paths:
        try:
            with xr.open_dataset(p) as ds:
                times = ds["time"].values
                # Convert to ISO strings regardless of datetime type
                times_str = [str(t) for t in times]
                available_dates.append(times_str)
        except Exception as e:
            print(f"Warning: Could not read {p}: {e}")
    return np.array( np.concat(available_dates), dtype='datetime64[ns]')


  def compute_mean_std_per_channel(self, zarr_path):
    z = zarr.open(zarr_path, mode='r')
    C = z.shape[1]
    count = 0
    mean = np.zeros(C)
    M2 = np.zeros(C)
    spatial_size = np.prod(z.shape[2:])

    for i in range(z.shape[0]):
        x = z[i].astype(np.float64)  # (C, H, W)
        x = x.reshape(C, -1)  # (C, H*W)
        count += x.shape[1]
        delta = x - mean[:, None]
        mean += np.nansum(delta, axis=1) / count
        M2 += np.nansum((x - mean[:, None]) * delta, axis=1)

    std = np.sqrt(M2 / count)
    return mean.astype(np.float32), std.astype(np.float32)

  def compute_min_max_per_channel(self, zarr_path):
    z = zarr.open(zarr_path, mode='r')
    C = z.shape[1]  # number of channels (variables)
    min_vals = np.full(C, np.inf, dtype=np.float32)
    max_vals = np.full(C, -np.inf, dtype=np.float32)

    for i in range(z.shape[0]):  # over time steps
        x = z[i].astype(np.float32)  # shape: (C, H, W)
        x = x.reshape(C, -1)  # shape: (C, H*W)

        min_vals = np.minimum(min_vals, np.nanmin(x, axis=1))
        max_vals = np.maximum(max_vals, np.nanmax(x, axis=1))

    return min_vals, max_vals


  def to_disk(self, zarr_path, compute_mask = False):

    
    # Initialize zarr store once (on first data block)
    zarr_store = zarr.open(
        zarr_path,
        mode='w',
        shape=(self.num_samples, self.num_vars, *self.num_spatial_dims_transformed),
        chunks=(1, self.num_vars, *self.num_spatial_dims_transformed),
        dtype="float32",
        compressor=numcodecs.Blosc(cname='zstd', clevel=5),
        zarr_format=2,
        fill_value=np.nan
    )

    # Attributes
    zarr_store.attrs['dates'] = [str(date) for date in self.dates]
    zarr_store.attrs['num_samples'] = self.num_samples
    zarr_store.attrs['temporal_freq'] = self.freq
    zarr_store.attrs['variables'] = self.variables
    zarr_store.attrs['spatial_dims'] = self.spatial_dims
    zarr_store.attrs['num_spatial_dims'] = self.num_spatial_dims
    zarr_store.attrs['num_spatial_dims_transformed'] = self.num_spatial_dims_transformed
    zarr_store.attrs['name_dims'] = ["time", "variable"] + self.spatial_dims
    if self.transformed:
        zarr_store.attrs['name_dims_transformed'] = ["time", "variable"] + self.spatial_dims_transformed
    else: 
        zarr_store.attrs['name_dims_transformed'] = zarr_store.attrs['name_dims']

    for t_params in self.transform:
        if t_params.get("transformation_func"):
            func_name = t_params["transformation_func"]
            zarr_store.attrs[func_name] = {k: v for k, v in t_params.items() if k not in ["transformation_func", "module"]}
            print(func_name)
            print(zarr_store.attrs[func_name])

    for coord_name in self.coords:
        if coord_name != "time":
            zarr_store.attrs[coord_name] = self.coords[coord_name].values.tolist()

    # Data
    sources = self.source_files
    for source in sources:
        x = xr.open_dataset(source)      
        vars_source = x.data_vars

        for var in vars_source:
            if var in self.variables:
                print(f"✅ Variable {var} from {source} matches target variables.")

                # Load data
                x_ = x[[var]]

                # Apply transforms
                for t_params in self.transform:
                    if t_params.get("transformation_func"):
                        module = importlib.import_module(t_params["module"])
                        func_name = t_params["transformation_func"]
                        func = getattr(module, func_name)
                        zarr_store.attrs[func_name] = {k: v for k, v in t_params.items() if k not in ["transformation_func", "module"]}
                        x_ = func(x_, **{k: v for k, v in t_params.items() if k not in ["transformation_func", "module"]})

                # General info
                idx_var = self.variables.index(var)
                idx_samples = [np.where(self.dates == t)[0][0] for t in x_.time.values.astype('datetime64[ns]')]
                
                # From xarray to numpy
                x_tensor = xarray_to_numpy(x_).astype(np.float32)
                del x_

                # Write data block
                for i, t_idx in enumerate(idx_samples):
                    zarr_store[t_idx, idx_var, ...] = x_tensor[i]
            else:
                print(f"⚠️ Skipping variable {var} in {source} not in target variable list.")
        
        x.close()        
        del x

    # Compute mean/std, min/max after writing
    m, s = self.compute_mean_std_per_channel(zarr_path)
    mn, mx = self.compute_min_max_per_channel(zarr_path)

    print("------ SOME STATISTICS ------")
    print(f"Mean: {m}")
    print(f"Std: {s}")
    print(f"Min: {mn}")
    print(f"Max: {mx}")
    print("-----------------------------")

    # Store statistics as attributes in .zarr
    zarr_store.attrs['mean'] = m.tolist()
    zarr_store.attrs['std'] = s.tolist()

    # Store statistics as attributes in .zarr
    zarr_store.attrs['min'] = mn.tolist()
    zarr_store.attrs['max'] = mx.tolist()


    return f"Saved to disk...: {zarr_path}"
  




########################################################################################################
class d4d_dataloader(Dataset):
  def __init__(self, 
               path_predictors, path_predictands, 
               variables_predictors, variables_predictands,
               years_predictors, years_predictands,
               standardize_predictors: str = None):


    # Files
    self.x = [zarr.open(p, mode='r') for p in path_predictors]
    self.y = [zarr.open(p, mode='r') for p in path_predictands]


    # Variables
    self.vars_predictors = variables_predictors
    self.vars_predictands = variables_predictands


    # For standardization
    self.standardize_predictors = standardize_predictors      
    if self.standardize_predictors is None:
        zarr_file = self.x[0]
    else:
        zarr_file = self.standardize_predictors
    self.mean_x, self.std_x = self.get_statistics(zarr_file)


    # Number of samples
    self.idx_x, self.dates_x = zip(*[
        (
            [i for i, date in enumerate(X.attrs.get("dates")) if int(date[:4]) in years_predictors],
            [date for i, date in enumerate(X.attrs.get("dates")) if int(date[:4]) in years_predictors]
        )
        for X in self.x
    ])
    self.idx_x = [i for sublist in self.idx_x for i in sublist]
    self.dates_x = [d for sublist in self.dates_x for d in sublist]

    # print(self.dates_x)
    # print(len(self.dates_x))

    self.idx_y, self.dates_y = zip(*[
        (
            [i for i, date in enumerate(Y.attrs.get("dates")) if int(date[:4]) in years_predictands],
            [date for i, date in enumerate(Y.attrs.get("dates")) if int(date[:4]) in years_predictands]
        )
        for Y in self.y
    ])
    self.idx_y = [i for sublist in self.idx_y for i in sublist]
    self.dates_y = [d for sublist in self.dates_y for d in sublist]
    

    if len(self.dates_x) != len(self.dates_y):
        raise ValueError("X and Y datasets have different number of samples.")

    # if self.dates_x != self.dates_y:
    #     print("X and Y datasets have different number of samples. Selecting a common temporal period.")

    #     dates_x_full = self.dates_x
    #     dates_y_full = self.dates_y

    #     print(dates_x_full[0])
    #     print(len(dates_x_full))
    #     print(dates_y_full[0])
    #     print(len(dates_y_full))

    #     # Convert to sets to get intersection
    #     common_dates = set(self.dates_x).intersection(set(self.dates_y))
    #     print("Common dates")
    #     print(common_dates)

    #     self.dates_x = [d for d in self.dates_x if d in common_dates]
    #     self.idx_x = [self.idx_x[i] for i, d in enumerate(dates_x_full) if d in common_dates]

    #     self.dates_y = [d for d in self.dates_y if d in common_dates]
    #     self.idx_y = [self.idx_y[i] for i, d in enumerate(dates_y_full) if d in common_dates]

    # print(f"X: {len(self.dates_x)}")
    # print(f"Y: {len(self.dates_y)}")
    self.num_samples = len(self.dates_x)
    print(f"Number of samples: {self.num_samples}")

  def get_stats(self):
    return self.mean_x, self.std_x

  def get_statistics(self, zarr_file):
    zarr_file = zarr.open(zarr_file, mode='r')
    mean = np.array(zarr_file.attrs.get('mean')).astype(np.float32)
    std = np.array(zarr_file.attrs.get('std')).astype(np.float32)
    # Extend dimensions to match input data dimensions
    axis = 0  # the axis you want to standardize over (e.g., channels)
    shape = [1] * (zarr_file.ndim - 1) # Remove time/batch dimension
    shape[axis] = -1  # insert C here
    # print(shape)
    mean = from_numpy(mean).view(*shape)
    std = from_numpy(std).view(*shape)
    return mean, std


  def __len__(self):
    return self.num_samples

  def __getitem__(self, idx):

    date_x = self.dates_x[idx]
    date_y = self.dates_y[idx]
    idx_real_x = self.idx_x[idx]
    idx_real_y = self.idx_y[idx]

    idx_zarr_x = [i for i, X in enumerate(self.x) if date_x in X.attrs.get("dates") ]
    idx_zarr_x = idx_zarr_x[0]
    idx_zarr_y = [i for i, Y in enumerate(self.y) if date_y in Y.attrs.get("dates") ]
    idx_zarr_y = idx_zarr_y[0]

    # print(f"Pytorch:{idx} --- {date_x} --- X IDX in zarr:{idx_real_x} --- X IDX zarr: {idx_zarr_x} --- Y IDX in zarr:{idx_real_y} --- Y IDX zarr: {idx_zarr_y}")

    source_x = self.x[idx_zarr_x][idx_real_x].astype(np.float32)
    source_y = self.y[idx_zarr_y][idx_real_y].astype(np.float32)

    x = from_numpy(source_x)
    y = from_numpy(source_y)

    # Apply standardization 
    x = (x - self.mean_x) / self.std_x

    # Eliminate singleton variable dimension for univariate cases.
    y = y.squeeze()

    return x, y





