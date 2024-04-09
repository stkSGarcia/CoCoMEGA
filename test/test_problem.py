import random

from deap import creator, base, tools

import test
from impl.mr.predefined import *
from impl.scenario.scenario_definition import ScenarioDefinition

mr_set = mr_set1

creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Solution", tuple, fitness=creator.Fitness, is_violated=False)
creator.create("Scenario", ScenarioDefinition, fitness=creator.Fitness)
creator.create("Perturbation", list, fitness=creator.Fitness)

toolbox = base.Toolbox()
toolbox.register("scenario", tools.initIterate, creator.Scenario, creator.Scenario.generate_random)
toolbox.register("perturbation", tools.initIterate, creator.Perturbation,
                 lambda: [random.choice(mr_set.mrs).generate_perturbation()])
toolbox.register("pop_scenario", tools.initRepeat, list, toolbox.scenario, n=test.CONFIG["scenario"]["pop_size"])
toolbox.register("pop_perturbation", tools.initRepeat, list, toolbox.perturbation,
                 n=test.CONFIG["perturbation"]["pop_size"])
