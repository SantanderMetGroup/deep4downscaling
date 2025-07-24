import sys
import yaml
from deep4downscaling.console.d4dcreate import d4dcreate

def main():
    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Call your create function with unpacked config
    d4dcreate(**config)
