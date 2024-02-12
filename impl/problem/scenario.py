import random

from deap import base, creator, tools

from impl.scenario.scenario import Scenario

TIME_BUDGET = 3600
MAX_ITERATIONS = 20
POP_SIZE = 10
CXPB = 0.8
MUTPB = 0.6
GUASSIAN_MUT_MEAN = 0
GUASSIAN_MUT_STD = 40
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

perturbations = []

# Fitness functions:
# 1. maximize the extent of violation.
# 2. minimize the length of the representation of the individual.
creator.create("Fitness", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.Fitness, covered_objectives=list)

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual,
                 lambda: Scenario.generate_random_scenario(SCENARIO_BOUNDARY).vector)
toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=POP_SIZE)
toolbox.register("mate", tools.cxUniform, indpb=CXPB)
toolbox.register("select", tools.selNSGA2, k=POP_SIZE)


def _mutate(individual):
    """Mutate a given scenario.

    @param individual: The scenario (`creator.Scenario`) to be mutated.
    @return: The mutated scenario.
    """
    mutated = []
    for i, element in enumerate(individual):
        if isinstance(element, float):
            element = tools.mutGaussian([element], mu=GUASSIAN_MUT_MEAN, sigma=GUASSIAN_MUT_STD, indpb=MUTPB)
        elif isinstance(element, int):
            element = tools.mutUniformInt([element], low=SCENARIO_BOUNDARY[i][0], up=SCENARIO_BOUNDARY[i][0],
                                          indpb=MUTPB)
        else:
            raise ValueError(f"Invalid element type: {type(element)}.")
        mutated.extend(element[0])
    individual[:] = mutated
    return individual


def _evaluate(individuals):
    """Evaluate the fitness values of the given individuals.

    @param individuals: The individuals to be evaluated.
    @return: The individuals with fitness evaluated.
    """
    for individual in individuals:
        source_results = []  # TODO: run simulation
        scenario = toolbox.clone(individual)
        for perturbation in perturbations:
            perturbation.perturb(scenario)
        follow_up_results = []  # TODO: run simulation
        # TODO: calculate fitness
        individual.fitness.values = random.uniform(0.0, 100.0), 0  # For test
    return individuals


toolbox.register("mutate", tools.mutPolynomialBounded, eta=20.0, indpb=MUTPB)
toolbox.register("evaluate", _evaluate)
