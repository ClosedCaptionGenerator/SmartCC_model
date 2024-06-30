import json

def get_config(config_file='config/config.json'):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config
