## Load libraries
import sys
import zarr

import deep4downscaling as d4d

##################################################################################################################################
##################################################################################################################################

def d4dinspect(
    zarr_path: str,
):

    # Load the Zarr store
    zarr_store = zarr.open(zarr_path, mode='r')
    
    # General Info
    date_init = zarr_store.attrs.get('dates', [None])[0]
    date_end = zarr_store.attrs.get('dates', [None])[-1]
    # freq = zarr_store.attrs.get('freq', [None])[0]
    num_samples = zarr_store.attrs.get('num_samples', 'Unknown')
    
    num_spatial_dims = zarr_store.attrs.get('num_spatial_dims', [])
    name_dims = zarr_store.attrs.get('name_dims', [])

    num_spatial_dims_transformed = zarr_store.attrs.get('num_spatial_dims_transformed', [])
    name_dims_transformed = zarr_store.attrs.get('name_dims_transformed', [])

    variables = zarr_store.attrs.get('variables', [])
    num_variables = len(variables)
    means = zarr_store.attrs.get('mean', [])
    stds = zarr_store.attrs.get('std', [])
    mins = zarr_store.attrs.get('min', [])
    maxs = zarr_store.attrs.get('max', [])

    # PRINT ------------------------------------------------------
    print("-" * 100)
    print("General Information 📈🤖📊")
    print("-" * 100)
    print(f"{'Date Init:'} {date_init}")
    print(f"{'Date End:'} {date_end}")
    print(f"Number of samples: {num_samples}")

    print()

    print(f"Dimensions: {num_samples}x{num_variables}x{num_spatial_dims} ----> {num_samples}x{num_variables}x{num_spatial_dims_transformed}")
    print(f"Name of dimensions: {name_dims} ----> {name_dims_transformed}")

    print()
    print()

    print("-" * 100)
    print("Variables Summary 📊📉📈")
    print("-" * 100)
    print(f"{'Variable':15} | {'Mean':>10} | {'Std':>10} | {'Min':>10} | {'Max':>10}")
    print("-" * 100)
    
    for i, var in enumerate(variables):
        m = f"{means[i]:.4f}" 
        s = f"{stds[i]:.4f}" 
        mn = f"{mins[i]:.4f}" 
        mx = f"{maxs[i]:.4f}" 
        print(f"{var:15} | {m:>10} | {s:>10} | {mn:>10} | {mx:>10}")
    
    print("-" * 100)
    


