import sys
import yaml
from deep4downscaling.console.d4dinspect import d4dinspect 

def main():
    zarr_path = sys.argv[1]
    d4dinspect(zarr_path)
