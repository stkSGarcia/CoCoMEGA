import random

from deap import creator, base, tools

from impl.mr.predefined import *
from impl.scenario.scenario import Scenario

SCENARIO_POP_SIZE = 10
PERTURBATION_POP_SIZE = 10
CXPB = 0.8
MUTPB = 0.6

# Fitness functions
# For scenarios:
# 1. maximize the number of MRs violated
# 2. minimize the length of the representation of the individual
# For perturbations:
# 1. maximize the extent of violation
# 2. minimize the length of the perturbation sequence.
creator.create("Fitness", base.Fitness, weights=(1.0, -1.0))
creator.create("Scenario", list, fitness=creator.Fitness, covered_objectives=list)
creator.create("Perturbation", list, fitness=creator.Fitness, covered_objectives=list)

toolbox = base.Toolbox()
toolbox.register("scenario", tools.initIterate, creator.Scenario,
                 lambda: Scenario.generate_random_scenario().vector)
toolbox.register("perturbation", tools.initIterate, creator.Perturbation,
                 lambda: [random.choice(mr_set1.mrs).generate_perturbation()])
toolbox.register("pop_scenario", tools.initRepeat, list, toolbox.scenario)
toolbox.register("pop_perturbation", tools.initRepeat, list, toolbox.perturbation)

toolbox.register("mate", tools.cxSimulatedBinary, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, eta=20.0, indpb=MUTPB)
toolbox.register("select", tools.selNSGA2)


def _evaluate_complete_solution(individual):
    pass


toolbox.register("evaluate", _evaluate_complete_solution)
