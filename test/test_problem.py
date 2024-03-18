import os

os.chdir("..")
from impl import config

config.init_config()

import random
from concurrent.futures import ProcessPoolExecutor

from deap import creator, base, tools

from impl.config import CONFIG
from impl.mr.predefined import *
from impl.scenario.scenario_definition import ScenarioDefinition

mr_set = mr_set1

creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Solution", tuple, fitness=creator.Fitness)
creator.create("Scenario", ScenarioDefinition, fitness=creator.Fitness)
creator.create("Perturbation", list, fitness=creator.Fitness)

toolbox = base.Toolbox()
toolbox.register("scenario", tools.initIterate, creator.Scenario, creator.Scenario.generate_random)
toolbox.register("perturbation", tools.initIterate, creator.Perturbation,
                 lambda: [random.choice(mr_set.mrs).generate_perturbation()])
toolbox.register("pop_scenario", tools.initRepeat, list, toolbox.scenario, n=CONFIG["scenario"]["pop_size"])
toolbox.register("pop_perturbation", tools.initRepeat, list, toolbox.perturbation, n=CONFIG["perturbation"]["pop_size"])

scenario = toolbox.scenario()
print(type(scenario))
print(isinstance(scenario, creator.Scenario))

def _eval(solution):
    return solution

with ProcessPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
    candidates = executor.map(_eval, [scenario])
candidates = list(candidates)

print(type(scenario))
print(isinstance(scenario, creator.Scenario))

print("-------------------------")
class A:
    pass
a = A()
print(type(a))
print(isinstance(a, A))
def _eval(solution):
    return solution
with ProcessPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
    c = executor.map(_eval, [a])
c = list(c)
print(type(a))
print(isinstance(a, A))
