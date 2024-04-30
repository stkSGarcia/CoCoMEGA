import argparse
import logging.config
import sys

import argformat

from impl.algorithm import *
from impl.utils.visualization import EvolutionVisualization

logger = logging.getLogger("impl")


def ccea(algorithm: str, resume: bool):
    from impl import problem
    if algorithm == "ccea":
        solver = CCEA(
            toolbox=problem.toolbox,
            budget=problem.budget,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}.")
    solver.solve(resume)

    visualizer = EvolutionVisualization(solver.logbook)
    visualizer.visualize(show=False)


def simulate():
    from impl.scenario.scenario_definition import ScenarioDefinition
    from impl.scenario.simulation_runner import run_scenarios
    run_scenarios([ScenarioDefinition.mock()])


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
