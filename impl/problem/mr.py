import random

from deap import creator, base, tools

from impl.mr.predefined import *
from impl.scenario.scenario import ScenarioDefinition

TIME_BUDGET = 3600
MAX_ITERATIONS = 20
POP_SIZE = 10
SOURCE_SCENARIO_SIZE = 10
CXPB = 0.8
MUTPB = 0.5
SCENARIO_BOUNDARY = [
    (0.0, 1000.0),
    (0.0, 1000.0),
    (0.0, 360.0),
    (0.0, 360.0),
    (0.0, 100.0),
    (0.0, 100.0),
    (0, 1),
    (0.0, 100.0),

    (0.0, 1000.0),
    (0.0, 1000.0),
    (0.0, 360.0),
    (0.0, 360.0),
    (0.0, 100.0),
    (0.0, 100.0),
    (0, 2),
    (0.0, 100.0),
    (0.0, 100.0),
    (0.0, 100.0),

    (0.0, 1000.0),
    (0.0, 1000.0),
    (0.0, 360.0),
    (0.0, 360.0),

    (0, 10),
    (0, 5),
]

source_scenarios = [ScenarioDefinition.generate_random_scenario(SCENARIO_BOUNDARY) for _ in range(SOURCE_SCENARIO_SIZE)]
mr_set = mr_set1

# Fitness functions:
# 1. maximize the extent of violation.
# 2. minimize the length of the perturbation sequence.
creator.create("Fitness", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.Fitness, covered_objectives=list)

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual,
                 lambda: [random.choice(mr_set.mrs).generate_perturbation()])
toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=POP_SIZE)
toolbox.register("mate", tools.cxUniform, indpb=CXPB)


def _mutate(individual):
    """Mutate a give sequence of perturbations by appending more perturbations to the sequence.

    @param individual: The sequence of perturbations (`creator.Perturbation`) to be mutated.
    @return: The mutated sequence of perturbations.
    """
    # Add perturbations.
    times = 1
    while random.random() < MUTPB ** times:
        perturbation = random.choice(mr_set.mrs).generate_perturbation()
        individual.append(perturbation)
        times += 1
    # TODO: Squash the individual
    return individual


def _evaluate(individuals):
    """Evaluate the fitness values of the given individuals.

    @param individuals: The individuals to be evaluated.
    @return: The individuals with fitness evaluated.
    """
    for individual in individuals:
        for scenario in source_scenarios:
            source_results = []  # TODO: run simulation
            scenario = toolbox.clone(scenario)
            for perturbation in individual:
                perturbation.perturb(scenario)
            follow_up_results = []  # TODO: run simulation
            # TODO: calculate fitness
        individual.fitness.values = random.uniform(0.0, 100.0), 0  # For test
    return individuals


toolbox.register("mutate", _mutate)
toolbox.register("evaluate", _evaluate)
