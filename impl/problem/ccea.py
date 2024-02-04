import random

from deap import creator, base, tools

from impl.mr.predefined import *
from impl.scenario.scenario import Scenario

TIME_BUDGET = 3600
MAX_ITERATIONS = 20
SCENARIO_POP_SIZE = 10
PERTURBATION_POP_SIZE = 10
MIN_NUM_EVALS = 1
ARCHIVE_SIZE = 3
CXPB = 0.8
TOURNAMENT_SIZE = 3
MUTPB = 0.9
GUASSIAN_MUT_MEAN = 0
GUASSIAN_MUT_STD = 40
PERT_MUTPB = 0.5
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

mr_set = mr_set1

# Fitness functions
# For scenarios:
# 1. maximize the extent of violation.
# 2. minimize the length of the representation of the individual.
# For perturbations:
# 1. maximize the extent of violation.
# 2. minimize the length of the perturbation sequence.
creator.create("Fitness", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", tuple, fitness=creator.Fitness)
creator.create("Scenario", list, fitness=creator.Fitness)
creator.create("Perturbation", list, fitness=creator.Fitness)

toolbox = base.Toolbox()
toolbox.register("scenario", tools.initIterate, creator.Scenario,
                 lambda: Scenario.generate_random_scenario(SCENARIO_BOUNDARY).vector)
toolbox.register("perturbation", tools.initIterate, creator.Perturbation,
                 lambda: [random.choice(mr_set.mrs).generate_perturbation()])
toolbox.register("pop_scenario", tools.initRepeat, list, toolbox.scenario, n=SCENARIO_POP_SIZE)
toolbox.register("pop_perturbation", tools.initRepeat, list, toolbox.perturbation, n=PERTURBATION_POP_SIZE)

toolbox.register("mate", tools.cxUniform, indpb=CXPB)
toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)


def _collaborate(scenario, perturbation):
    """Create a complete solution from two individuals.

    @param scenario: The scenario individual defined by `creator.Scenario`.
    @param perturbation: The perturbation individual defined by `creator.Perturbation`.
    @return: The complete solution individual defined by `creator.Individual`.
    """
    return creator.Individual((scenario, perturbation))


toolbox.register("collaborate", _collaborate)


def _mutate_scenario(individual):
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


def _mutate_perturbation(individual):
    """Mutate a give sequence of perturbations by appending more perturbations to the sequence.

    @param individual: The sequence of perturbations (`creator.Perturbation`) to be mutated.
    @return: The mutated sequence of perturbations.
    """
    # Add perturbations.
    times = 1
    while random.random() < PERT_MUTPB ** times:
        perturbation = random.choice(mr_set.mrs).generate_perturbation()
        individual.append(perturbation)
        times += 1
    # TODO: Squash the individual
    return individual


toolbox.register("mutate_scenario", _mutate_scenario)
toolbox.register("mutate_perturbation", _mutate_perturbation)


def _evaluate_complete_solution(solution):
    """Evaluate the joint fitness of a complete solution.

    @param solution: The complete solution (`creator.Individual`) to evaluate.
    @return: The complete solution with fitness evaluated.
    """
    source_results = []  # TODO: run simulation
    scenario = Scenario(toolbox.clone(solution[0]))
    for perturbation in solution[1]:
        perturbation.perturb(scenario)
    follow_up_results = []  # TODO: run simulation
    # TODO: calculate fitness
    solution.fitness.values = random.uniform(0.0, 100.0), 0  # For test
    return solution


def _evaluate_individual(individual, index, complete_solutions):
    """Evaluate the individual fitness of a scenario or a sequence of perturbations.

    @param individual: The individual to be evaluated.
    @param index: 0 for scenario, 1 for perturbations.
    @param complete_solutions: The list of complete solutions with fitness evaluated.
    @return: The individual with fitness evaluated.
    """
    involved = []
    for solution in complete_solutions:
        if solution[index] == individual:
            involved.append(solution.fitness.values[0])
    individual.fitness.values = max(involved), len(individual)
    return individual


toolbox.register("evaluate_joint", _evaluate_complete_solution)
toolbox.register("evaluate_individual", _evaluate_individual)
