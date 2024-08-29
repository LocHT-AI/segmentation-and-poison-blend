import yaml
import json
 

def load_env():
    with open('env.json') as stream:
        env_cfg = json.load(stream)
        return env_cfg['env']

env = load_env()

def load_config():
    file_name = f'config/config.yml'

    with open(file_name, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

cfg = load_config()

 

# CONFIGS

MODEL_SEG =cfg["model"]["model-seg"]
MODEL_POSE =cfg["model"]["model-pose"]
 