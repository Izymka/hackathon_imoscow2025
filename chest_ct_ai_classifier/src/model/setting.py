import argparse
import yaml


class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

    def __repr__(self):
        return f"Config({self.__dict__})"


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    # Load config from YAML
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    return Config(config_dict)