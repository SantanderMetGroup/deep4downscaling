import sys
import yaml
from deep4downscaling.console.d4dtrainer import d4dtrainer

def main():
    if len(sys.argv) != 2:
        print("Usage: d4d-train path/to/config.yaml")
        sys.exit(1)  # Exit with error code

    config_file = sys.argv[1]
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    d4dtrainer(**config)