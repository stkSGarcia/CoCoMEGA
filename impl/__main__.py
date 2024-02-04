import argparse
import collections.abc
import logging.config
import os

import argformat
import yaml

from impl.algorithm.ccea import CCEA
from impl.algorithm.nsga2 import NSGA2

logger = logging.getLogger("impl")
CONFIG = None


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f.read())


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
    CONFIG = {**default_config, **custom_config}

    # Create out and log directories.
    module_dir = os.path.dirname(os.path.dirname(__file__))
    out_dir = os.path.join(module_dir, CONFIG["out"])
    os.makedirs(out_dir, exist_ok=True)
    log_dir = os.path.join(module_dir, CONFIG["log"])
    os.makedirs(log_dir, exist_ok=True)

    # Log configurations.
    def update_log_dir(dictionary):
        for k, v in dictionary.items():
            if isinstance(v, collections.abc.Mapping):
                update_log_dir(v)
            elif k == "filename":
                dictionary[k] = os.path.join(log_dir, dictionary[k])

    default_log_config_path = os.path.join(default_config_base, log_config_name)
    default_log_config = load_yaml(default_log_config_path) if os.path.isfile(default_log_config_path) else {}
    custom_log_config = load_yaml(log_config_name) if os.path.isfile(log_config_name) else {}
    log_config = {**default_log_config, **custom_log_config}
    if log_config:
        update_log_dir(log_config)
        logging.config.dictConfig(log_config)
    else:
        logger.warning("Cannot find log configuration file.")


def ccea():
    from impl.problem import ccea as problem_ccea
    solver = CCEA(
        min_num_evals=problem_ccea.MIN_NUM_EVALS,
        archive_size=problem_ccea.ARCHIVE_SIZE,
        toolbox=problem_ccea.toolbox,
        time_budget=problem_ccea.TIME_BUDGET,
        max_iter=problem_ccea.MAX_ITERATIONS,
    )
    solution = solver.solve()


def scenario(algorithm: str):
    from impl.problem import scenario as problem_scenario
    if algorithm == "nsga2":
        solver = NSGA2(
            toolbox=problem_scenario.toolbox,
            pop_size=problem_scenario.POP_SIZE,
            cxpb=problem_scenario.CXPB,
            mutpb=problem_scenario.MUTPB
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}.")
    solver.solve()


if __name__ == "__main__":
    # Configuration initialization.
    init_config()

    # Parse command line.
    parser = argparse.ArgumentParser(
        prog="mtcg",
        description="Test case generator for metamorphic testing.",
        formatter_class=argformat.StructuredFormatter
    )
    subparsers = parser.add_subparsers(
        title="subcommands",
        description="Valid subcommands",
        required=True,
        help="subcommand help"
    )

    parser_ccea = subparsers.add_parser("ccea", help="")
    parser_ccea.set_defaults(func=lambda args: ccea())

    parser_scenario = subparsers.add_parser("scenario", aliases=["scen"], help="")
    parser_scenario.add_argument("-a", "--algorithm", required=False, choices=("nsga2", "mosa"),
                                 default="nsga2", help="")
    parser_scenario.set_defaults(func=lambda args: scenario(algorithm=args.algorithm))

    arguments = parser.parse_args()
    arguments.func(arguments)
