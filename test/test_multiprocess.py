import random
from concurrent.futures import ProcessPoolExecutor

from deap import creator, base, tools

import test
from impl.scenario.scenario_definition import ScenarioDefinition

creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Scenario", ScenarioDefinition, fitness=creator.Fitness)
creator.create("Perturbation", list, fitness=creator.Fitness)

toolbox = base.Toolbox()
toolbox.register("scenario", tools.initIterate, creator.Scenario, creator.Scenario.generate_random)
toolbox.register("perturbation", tools.initIterate, creator.Perturbation, lambda: [random.random() for _ in range(10)])


def check_type(obj, typ):
    print(type(obj))
    print(isinstance(obj, typ))


def _eval(solution):
    return solution


def multi_process(obj):
    with ProcessPoolExecutor(max_workers=test.CONFIG["max_workers"]) as executor:
        c = executor.map(_eval, [obj])
    c = list(c)


check_type(toolbox.perturbation(), creator.Perturbation)
multi_process(toolbox.perturbation())
check_type(toolbox.perturbation(), creator.Perturbation)

check_type(toolbox.scenario(), creator.Scenario)
multi_process(toolbox.scenario())
check_type(toolbox.scenario(), creator.Scenario)
