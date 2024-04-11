import argparse
import logging.config
import sys

import argformat

from impl import config
from impl.algorithm import *

logger = logging.getLogger("impl")


def ccea(algorithm: str, resume: bool):
    from impl import problem
    if algorithm == "ccea":
        solver = CCEA(
            archive_size=config.CONFIG["archive_size"],
            toolbox=problem.toolbox,
            max_sim=config.CONFIG["max_sim"],
            max_time=config.CONFIG["max_time"],
            max_iter=config.CONFIG["max_iter"],
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}.")
    solver.solve(resume)


def simulate():
    import sys
    import os
    root = "impl/scenario"
    for path in [
        os.path.join(root, "carla/PythonAPI"),
        os.path.join(root, "carla/PythonAPI/carla"),
        os.path.join(root, "carla/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg"),
        os.path.join(root, "leaderboard"),
        os.path.join(root, "leaderboard", "team_code"),
        os.path.join(root, "scenario_runner"),
    ]:
        sys.path.append(path)

    sys.argv.pop()
    for env, arg, value in [
        ("SCENARIOS", "scenarios", os.path.join(root, "leaderboard/data/scenarios/town05_all_scenarios.json")),
        ("ROUTES", "routes", os.path.join(root, "leaderboard/data/training_routes/routes_town05_long.xml")),
        ("REPETITIONS", "repetitions", 1),
        ("CHALLENGE_TRACK_CODENAME", "track", "SENSORS"),
        ("CHECKPOINT_ENDPOINT", "checkpoint",
         os.path.join(config.CONFIG["workspace"], config.CONFIG["simulation"]["result"], "sample_result.json")),
        ("TEAM_AGENT", "agent", os.path.join(root, "leaderboard/team_code/interfuser_agent.py")),
        ("TEAM_CONFIG", "agent-config", os.path.join(root, "leaderboard/team_code/interfuser_config.py")),
        ("DEBUG_CHALLENGE", "debug", 0),
        # ("RESUME", "resume", False),
        ("HOST", "host", config.CONFIG["simulation"]["host"]),
        ("PORT", "port", 2000),
        ("CARLA_ROOT", None, os.path.join(root, "carla")),
        ("CARLA_SERVER", None, os.path.join(root, "carla/CarlaUE4.sh")),
        ("LEADERBOARD_ROOT", None, os.path.join(root, "leaderboard")),
        ("SAVE_PATH", None, os.path.join(config.CONFIG["workspace"], config.CONFIG["simulation"]["save"])),
        ("TM_PORT", None, 2500),
    ]:
        if env is not None: os.environ[env] = str(value)
        if arg is not None: sys.argv.append(f"--{arg}={value}")

    from impl.scenario.interfuser_scenario_evaluator import main
    main()


if __name__ == "__main__":
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

    parser_search = subparsers.add_parser("search", aliases=["srch"], help="Start the search")
    parser_search.add_argument("-a", "--algorithm", choices=("ccea", "nsga2", "mosa"),
                               default="ccea", help="Choose the algorithm to use")
    parser_search.add_argument("-r", "--resume", action="store_true", help="Resume previous run")
    parser_search.set_defaults(func=lambda args: ccea(args.algorithm, args.resume))

    parser_sim = subparsers.add_parser("simulate", aliases=["sim"], help="")
    parser_sim.set_defaults(func=lambda args: simulate())

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    arguments = parser.parse_args()
    arguments.func(arguments)
