import argparse
import logging.config
import sys

import argformat

from impl import config
from impl.algorithm import *
from impl.scenario.scenario_definition import ScenarioDefinition
from impl.scenario.simulation_runner import run_scenarios, run_solutions
from impl.utils.visualization import Visualizer

logger = logging.getLogger("impl")


def ccea(algorithm: str, resume: bool):
    from impl import problem
    if algorithm == "ccea":
        solver = CCEA(toolbox=problem.toolbox, budget=problem.budget)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}.")
    solver.solve(resume=resume)

    visualizer = Visualizer()
    visualizer.visualize_gen_stats(
        stats=solver.logbook,
        show=False,
        out_dir=config.CONFIG["workspace"]["visualization"],
    )


def simulate(num: int, file: str):
    if file:
        logger.info(f"Loading solution file: {file}.")
        run_solutions(file, num)
    else:
        logger.info(f"Running random scenarios.")
        run_scenarios([ScenarioDefinition.generate_random() for _ in range(num)])


def visualize(file: str):
    logger.info(f"Plotting statistics data from file: {file}.")
    Visualizer.visualize_in_one(file)


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

    parser_sim = subparsers.add_parser("simulate", aliases=["sim"], help="Run the simulation")
    parser_sim.add_argument("-n", "--number", type=int, default=1, help="Number of simulations to run")
    parser_sim.add_argument("-f", "--file", default=None, help="Solution file")
    parser_sim.set_defaults(func=lambda args: simulate(args.number, args.file))

    parser_vis = subparsers.add_parser("visualize", aliases=["vis"], help="Visualize the results")
    parser_vis.add_argument("-f", "--file", required=True, help="Statistics file")
    parser_vis.set_defaults(func=lambda args: visualize(args.file))

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    arguments = parser.parse_args()
    arguments.func(arguments)
