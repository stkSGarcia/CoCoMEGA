import argparse
import logging.config
import sys

import argformat

from impl import problem
from impl.algorithm.ccea import CCEA
from impl.algorithm.ga import GeneticAlgorithm
from impl.algorithm.rs import RandomSearch
from impl.scenario.scenario_definition import ScenarioDefinition
from impl.scenario.simulation_runner import run_scenarios, run_solutions, run_free_environments
from impl.utils.docker_utils import cleanup_containers

logger = logging.getLogger("impl")


def search(algorithm: str, resume: bool):
    if algorithm == "ccea":
        solver = CCEA(toolbox=problem.toolbox, budget=problem.budget)
    elif algorithm == "rs":
        solver = RandomSearch(toolbox=problem.toolbox, budget=problem.budget)
    elif algorithm == "ga":
        solver = GeneticAlgorithm(toolbox=problem.toolbox, budget=problem.budget)
    elif algorithm == "gawa":
        solver = GeneticAlgorithm(toolbox=problem.toolbox, budget=problem.budget, keep_best=True)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}.")
    solver.solve(resume=resume)


def simulate(num: int, file: str):
    if file:
        logger.info(f"Loading solution file: {file}.")
        run_solutions(file, num)
    else:
        logger.info(f"Running random scenarios.")
        run_scenarios([ScenarioDefinition.generate_random_or_leaderboard() for _ in range(num)])

def run_free_env(agent: str):
    logger.info(f"Running free simulation environment for agent {agent}.")
    run_free_environments(agent)

class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        is_kv = ["=" in value for value in values]
        if all(is_kv):
            pairs = {}
            for value in values:
                k, v = value.split("=")
                pairs[k] = v.split(",")
            setattr(namespace, self.dest, pairs)
        elif not any(is_kv):
            setattr(namespace, self.dest, values)
        else:
            parser.error("expected consistent type of arguments")


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

    parser_search = subparsers.add_parser("search", aliases=["srch"], help="start the search")
    parser_search.add_argument("-a", "--algorithm", choices=("ccea", "rs", "ga", "gawa"), default="ccea",
                               help="choose the algorithm to use. "
                                    "ccea: cooperative co-evolutionary algorithm; "
                                    "rs: random search algorithm; "
                                    "ga: standard genetic algorithm; "
                                    "gawa: genetic algorithm with archive strategy")
    parser_search.add_argument("-r", "--resume", action="store_true", help="resume the previous run")
    parser_search.set_defaults(func=lambda args: search(args.algorithm, args.resume))

    parser_sim = subparsers.add_parser("simulate", aliases=["sim"], help="run simulations")
    parser_sim.add_argument("-n", "--number", type=int, default=1, help="number of simulations to run")
    parser_sim.add_argument("-f", "--file", default=None, help="solution file")
    parser_sim.set_defaults(func=lambda args: simulate(args.number, args.file))

    parser_sim = subparsers.add_parser("run", help="Execute free runtime environment")
    parser_sim.add_argument("-a", "--agent", type=str, default="v1", help="Agent version or name")
    parser_sim.set_defaults(func=lambda args: run_free_env(args.agent))

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    arguments = parser.parse_args()
    arguments.func(arguments)
    cleanup_containers()
