import json
import logging.config
import os
import sys
import time
import uuid
from collections.abc import Mapping
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)
CONFIG = {}  # Always use this variable by using `from impl import config as cfg; cfg.CONFIG`


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
            dictionary[k] = CONFIG["workspace"]["log"] / dictionary[k]


class DirectoryAccessor:
    def __init__(self, workspace_config: dict):
        self._config = workspace_config
        self._config["root"].mkdir(exist_ok=True, parents=True)

    def __setitem__(self, key, value):
        if key == "root":
            raise KeyError(f"Setting 'root' is not allowed.")
        self._config[key] = Path(value)

    def __getitem__(self, key):
        if key not in self._config:
            raise KeyError(f"'{key}' not in workspace configuration.")
        if self._config[key] is None: return None
        if key == "root": return self._config[key]

        path = self._config["root"] / self._config[key]
        path.mkdir(exist_ok=True, parents=True)
        return path

    def __contains__(self, key):
        return key in self._config

    def get(self, key, default=None):
        return self._config.get(key, default)


def init_config():
    """Load configurations."""
    config_base = Path("conf")
    config_name = Path("config.yaml")
    log_config_name = Path("log.yaml")

    # General configurations.
    global CONFIG
    default_config_path = config_base / config_name
    if default_config_path.is_file():
        default_config = yaml.safe_load(default_config_path.read_text())
    else:
        raise ValueError("Cannot find default configuration file.")
    custom_config = yaml.safe_load(config_name.read_text()) if config_name.is_file() else {}
    CONFIG = _merge_dict(default_config, custom_config)

    # Set up workspace root and create other directories on-demand.
    CONFIG["workspace"]["root"] = Path(__file__).parent.parent / CONFIG["workspace"]["root"]
    CONFIG["workspace"] = DirectoryAccessor({k: Path(v) for k, v in CONFIG["workspace"].items()})

    # Log configurations.
    default_log_config_path = config_base / log_config_name
    default_log_config = yaml.safe_load(default_log_config_path.read_text()) \
        if default_log_config_path.is_file() else {}
    custom_log_config = yaml.safe_load(log_config_name.read_text()) if log_config_name.is_file() else {}
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


def init_project_directory(name: str, resume=False):
    """Create necessary directories for executions or resume from an existing project directory.

    :param name: Execution name. If `resume` is `True`, this is the directory name to resume from.
    :param resume: Whether to resume.
    """
    global CONFIG
    if resume:
        CONFIG["workspace"]["current"] = CONFIG["workspace"].get("result") / name
        config = json.loads((CONFIG["workspace"]["current"] / "config.json").read_text())
        config["workspace"] = CONFIG["workspace"]
        CONFIG = config
    else:
        CONFIG["workspace"]["current"] = (
                CONFIG["workspace"].get("result") /
                f"{str(int(round(time.time() * 1000)))}-{name.lower()}-{uuid.uuid4().hex[:6]}"
        )

    for key, directory in (
            ("sim_result", "results"),
            ("checkpoint", "checkpoints"),
            ("solution", "solutions"),
            ("visual", "visualizations"),
    ):
        CONFIG["workspace"][key] = CONFIG["workspace"].get("current") / directory


def dump_config_to_json():
    """Dump the configuration dictionary to a JSON file."""
    serializable_config = CONFIG.copy()
    workspace_dict = {}
    root_path = serializable_config["workspace"]["root"]
    config_path = CONFIG["workspace"]["current"] / "config.json"
    for k, v in serializable_config["workspace"]._config.items():
        if k == "root":
            workspace_dict[k] = str(v)
        elif v is None:
            workspace_dict[k] = None
        else:
            workspace_dict[k] = v.relative_to(root_path)
    serializable_config["workspace"] = workspace_dict
    config_path.write_text(json.dumps(serializable_config, indent=4))
