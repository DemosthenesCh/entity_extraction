import yaml


def get_data_config(config_path: str = 'data_config.yaml') -> dict[str, str]:
    with open(config_path) as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)
    return data_config
