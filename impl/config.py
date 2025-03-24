import json
import logging.config
import os
import sys
import time
import uuid
from collections.abc import Mapping

import yaml

logger = logging.getLogger(__name__)
CONFIG = {}


class LazyPath:
    """Lazy directory creation."""

    def __init__(self, root, relative_path):
        self.root = root
        self.relative_path = relative_path
        self.full_path = os.path.join(root, relative_path)
        self._created = False

    def __str__(self):
        if not self._created:
            os.makedirs(self.full_path, exist_ok=True)
            self._created = True
        return self.full_path

    def __repr__(self):
        return repr(self.full_path)


def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f.read())


def _merge_dict(default: dict, user: dict):
    for k, v in user.items():
        if k not in default or not isinstance(default[k], dict):
            default[k] = v
        else:
            default[k] = _merge_dict(default[k], v)
    return default


def _update_log_dir(dictionary):
    for k, v in dictionary.items():
        if isinstance(v, Mapping):
            _update_log_dir(v)
        elif k == "filename":
            dictionary[k] = os.path.join(CONFIG["workspace"]["log"], dictionary[k])


def init_config():
    """Load configurations."""
    default_config_base = "conf"
    config_name = "config.yaml"
    log_config_name = "log.yaml"

    # General configurations.
    global CONFIG
    default_config_path = os.path.join(default_config_base, config_name)
    if os.path.isfile(default_config_path):
        default_config = _load_yaml(default_config_path)
    else:
        raise ValueError("Cannot find default configuration file.")
    custom_config = _load_yaml(config_name) if os.path.isfile(config_name) else {}
    CONFIG = _merge_dict(default_config, custom_config)

    # Set up workspace root and other paths with lazy directory creation.
    CONFIG["workspace"]["root"] = os.path.join(os.path.dirname(os.path.dirname(__file__)), CONFIG["workspace"]["root"])
    os.makedirs(CONFIG["workspace"]["root"], exist_ok=True)
    for k, v in CONFIG["workspace"].items():
        if k == "root" or v is None: continue
        CONFIG["workspace"][k] = LazyPath(CONFIG["workspace"]["root"], v)

    # Log configurations.
    default_log_config_path = os.path.join(default_config_base, log_config_name)
    default_log_config = _load_yaml(default_log_config_path) if os.path.isfile(default_log_config_path) else {}
    custom_log_config = _load_yaml(log_config_name) if os.path.isfile(log_config_name) else {}
    log_config = _merge_dict(default_log_config, custom_log_config)
    if log_config:
        _update_log_dir(log_config)
        logging.config.dictConfig(log_config)
    else:
        logger.warning("Cannot find log configuration file.")

    # Add InterFuser to path
    for path in [
        "carla/PythonAPI",
        "carla/PythonAPI/carla",
        "carla/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg",
        "leaderboard",
        "leaderboard/team_code",
        "scenario_runner",
    ]:
        sys.path.append(os.path.join(CONFIG["interfuser"]["repo"], path))


def init_project_folder(name: str, resume=False):
    """Create necessary folders for executions or resume from an existing project folder.

    :param name: Execution name. If `resume` is `True`, this is the folder name to resume from.
    :param resume: Whether to resume.
    """
    global CONFIG
    if resume:
        root_path = os.path.join(CONFIG["workspace"]["result"], name)
        with open(os.path.join(root_path, "config.json"), "r") as f:
            CONFIG = json.loads(f)
    else:
        root_path = os.path.join(CONFIG["workspace"]["result"],
                                 f"{str(int(round(time.time() * 1000)))}-{name.lower()}-{uuid.uuid4().hex[:6]}")
        os.makedirs(root_path, exist_ok=True)
    CONFIG["workspace"]["current_root"] = root_path
    for key, folder in (
            ("sim_result", "results"),
            ("checkpoint", "checkpoints"),
            ("solution", "solutions"),
    ):
        folder_path = os.path.join(root_path, folder)
        os.makedirs(folder_path, exist_ok=True)
        CONFIG["workspace"][key] = folder_path
