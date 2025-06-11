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


def merge_dict(default: dict, user: dict):
    """Recursively merge user dictionary into the default dictionary.

    :param default: The default configuration dictionary.
    :param user: The user-provided configuration dictionary.
    :return: Merged configuration dictionary.
    """
    for k, v in user.items():
        if k not in default or not isinstance(default[k], dict):
            default[k] = v
        else:
            default[k] = merge_dict(default[k], v)
    return default


def _update_log_dir(dictionary):
    """Update log directory paths in the logging configuration.

    :param dictionary: Dictionary of logging configurations.
    """
    for k, v in dictionary.items():
        if isinstance(v, Mapping):
            _update_log_dir(v)
        elif k == "filename":
            dictionary[k] = CONFIG["workspace"]["log"] / dictionary[k]


class _DirectoryAccessor:
    """A utility class to access and manage workspace directories."""

    def __init__(self, workspace_config: dict):
        """Initialize with the provided workspace configuration.

        :param workspace_config: Dictionary of workspace paths.
        """
        self._config = workspace_config
        self._config["root"].mkdir(exist_ok=True, parents=True)

    def __setitem__(self, key, value):
        """Set a directory path for a given key.

        :param key: Directory key (excluding `root`).
        :param value: Directory path.
        """
        if key == "root":
            raise KeyError(f"Setting 'root' is not allowed.")
        self._config[key] = Path(value)

    def __getitem__(self, key):
        """Get the full path for a directory.

        :param key: Directory key.
        :return: Full path for the given key.
        """
        if key not in self._config:
            raise KeyError(f"'{key}' not in workspace configuration.")
        if self._config[key] is None: return None
        if key == "root": return self._config[key]

        path = self._config["root"] / self._config[key]
        path.mkdir(exist_ok=True, parents=True)
        return path

    def __contains__(self, key):
        """Check if the workspace configuration contains the key.

        :param key: Key to check.
        :return: :data:`True` if key exists, :data:`False` otherwise.
        """
        return key in self._config

    def get(self, key, default=None):
        """Get the value for a given key, with optional default.

        :param key: Key to look up.
        :param default: Default value if `key` is not present.
        :return: Value associated with the key.
        """
        return self._config.get(key, default)


def init_config():
    """Initialize and load default and custom configurations, set up paths and logging."""
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
    CONFIG = merge_dict(default_config, custom_config)

    # Set up workspace root and create other directories on-demand.
    CONFIG["workspace"]["root"] = Path(__file__).parent.parent / CONFIG["workspace"]["root"]
    CONFIG["workspace"] = _DirectoryAccessor({k: Path(v) for k, v in CONFIG["workspace"].items()})

    # Log configurations.
    default_log_config_path = config_base / log_config_name
    default_log_config = yaml.safe_load(default_log_config_path.read_text()) \
        if default_log_config_path.is_file() else {}
    custom_log_config = yaml.safe_load(log_config_name.read_text()) if log_config_name.is_file() else {}
    log_config = merge_dict(default_log_config, custom_log_config)
    if log_config:
        _update_log_dir(log_config)
        logging.config.dictConfig(log_config)
    else:
        logger.warning("Cannot find log configuration file.")

    # Add InterFuser to the path
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
    """Create the necessary directories for executions or resume from an existing project directory.

    :param name: Execution name. If `resume` is :data:`True`, this is the directory name to resume from.
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
    """Dump the current configuration to a JSON file in the current project directory."""
    serializable_config = CONFIG.copy()
    workspace_dict = {}
    for k, v in serializable_config["workspace"]._config.items():
        if v is None:
            workspace_dict[k] = None
        else:
            workspace_dict[k] = str(v)
    serializable_config["workspace"] = workspace_dict
    (CONFIG["workspace"]["current"] / "config.json").write_text(json.dumps(serializable_config, indent=4))
