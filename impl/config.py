import collections
import logging.config
import os

import yaml

logger = logging.getLogger(__name__)
CONFIG = {}


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f.read())


def merge_dict(default: dict, user: dict):
    for k, v in user.items():
        if k not in default or not isinstance(default[k], dict):
            default[k] = v
        else:
            default[k] = merge_dict(default[k], v)
    return default


def init_config():
    """Load configurations."""
    default_config_base = "conf"
    config_name = "config.yaml"
    log_config_name = "log.yaml"

    # General configurations.
    global CONFIG
    default_config_path = os.path.join(default_config_base, config_name)
    if os.path.isfile(default_config_path):
        default_config = load_yaml(default_config_path)
    else:
        raise ValueError("Cannot find default configuration file.")
    custom_config = load_yaml(config_name) if os.path.isfile(config_name) else {}
    CONFIG = merge_dict(default_config, custom_config)

    # Create directories.
    CONFIG["workspace"] = os.path.join(os.path.dirname(os.path.dirname(__file__)), CONFIG["workspace"])
    os.makedirs(CONFIG["workspace"], exist_ok=True)
    CONFIG["log"] = os.path.join(CONFIG["workspace"], CONFIG["log"])
    os.makedirs(CONFIG["log"], exist_ok=True)
    CONFIG["result"] = os.path.join(CONFIG["workspace"], CONFIG["result"])
    os.makedirs(CONFIG["result"], exist_ok=True)

    # Log configurations.
    def update_log_dir(dictionary):
        for k, v in dictionary.items():
            if isinstance(v, collections.abc.Mapping):
                update_log_dir(v)
            elif k == "filename":
                dictionary[k] = os.path.join(CONFIG["log"], dictionary[k])

    default_log_config_path = os.path.join(default_config_base, log_config_name)
    default_log_config = load_yaml(default_log_config_path) if os.path.isfile(default_log_config_path) else {}
    custom_log_config = load_yaml(log_config_name) if os.path.isfile(log_config_name) else {}
    log_config = merge_dict(default_log_config, custom_log_config)
    if log_config:
        update_log_dir(log_config)
        logging.config.dictConfig(log_config)
    else:
        logger.warning("Cannot find log configuration file.")
