import argparse
import logging.config

import argformat

from impl import config
from impl.algorithm.ccea import CCEA
from impl.algorithm.mosa import MOSA
from impl.algorithm.nsga2 import NSGA2

logger = logging.getLogger("impl")


def ccea():
    from impl.problem import ccea as problem
    solver = CCEA(
        archive_size=problem.ARCHIVE_SIZE,
        toolbox=problem.toolbox,
        time_budget=problem.TIME_BUDGET,
        max_iter=problem.MAX_ITERATIONS,
    )
    solver.solve()


def scenario(algorithm: str):
    from impl.problem import scenario as problem
    if algorithm == "nsga2":
        solver = NSGA2(
            toolbox=problem.toolbox,
            time_budget=problem.TIME_BUDGET,
            max_iter=problem.MAX_ITERATIONS,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}.")
    solver.solve()


def mr(algorithm: str):
    from impl.problem import mr as problem
    if algorithm == "mosa":
        solver = MOSA(
            objectives=problem,
            pop_size=problem.POP_SIZE,
            toolbox=problem.toolbox,
            time_budget=problem.TIME_BUDGET,
            max_iter=problem.MAX_ITERATIONS,
        )
    else:
        raise ValueError(f"Unsupported algorithm {algorithm}.")
    solver.solve()


if __name__ == "__main__":
    # Configuration initialization.
    config.init_config()

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

    parser_mr = subparsers.add_parser("mr", help="")
    parser_mr.add_argument("-a", "--algorithm", required=False, choices=("nsga2", "mosa"),
                           default="mosa", help="")
    parser_mr.set_defaults(func=lambda args: mr(algorithm=args.algorithm))

    arguments = parser.parse_args()
    arguments.func(arguments)
