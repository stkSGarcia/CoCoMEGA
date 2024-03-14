import random

import numpy as np
from deap import creator, base, tools
from scipy.spatial.distance import pdist, squareform

from impl.mr.mr import Perturbation
from impl.mr.predefined import *
from impl.scenario.scenario_definition import ScenarioDefinition

TIME_BUDGET = 3600
MAX_ITERATIONS = 20
SCENARIO_POP_SIZE = 10
PERTURBATION_POP_SIZE = 10
ARCHIVE_SIZE = 3
PUNISHMENT_FACTOR = 1
SCALING_FACTOR = 1.5
CXPB = 0.8
TOURNAMENT_SIZE = 3
SCENARIO_MUTPB = 0.9
GUASSIAN_MUT_MEAN = 0
GUASSIAN_MUT_STD = 40
PERT_ADD_PB = 0.6
PERT_REMOVE_PB = 0.2

mr_set = mr_set1

creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Individual", tuple, fitness=creator.Fitness)
creator.create("Scenario", list, fitness=creator.Fitness)
creator.create("Perturbation", list, fitness=creator.Fitness)

toolbox = base.Toolbox()
toolbox.register("scenario", tools.initIterate, creator.Scenario,
                 lambda: ScenarioDefinition.generate_random_scenario().vector)
toolbox.register("perturbation", tools.initIterate, creator.Perturbation,
                 lambda: [random.choice(mr_set.mrs).generate_perturbation()])
# TODO: Initialize diverse individuals
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
            element = tools.mutGaussian([element], mu=GUASSIAN_MUT_MEAN, sigma=GUASSIAN_MUT_STD, indpb=SCENARIO_MUTPB)
        elif isinstance(element, int):
            element = tools.mutUniformInt([element], low=ScenarioDefinition.BOUNDARY[i][0],
                                          up=ScenarioDefinition.BOUNDARY[i][1], indpb=SCENARIO_MUTPB)
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
    while random.random() < PERT_ADD_PB ** times:
        perturbation = random.choice(mr_set.mrs).generate_perturbation()
        individual.append(perturbation)
        times += 1

    # Remove one previous perturbation.
    if len(individual) > 1 and random.random() < PERT_REMOVE_PB:
        individual.pop()
    return individual


toolbox.register("mutate_scenario", _mutate_scenario)
toolbox.register("mutate_perturbation", _mutate_perturbation)


def _evaluate_complete_solution(solution):
    """Evaluate the joint fitness of a complete solution.

    @param solution: The complete solutions (`creator.Individual`) to be evaluated.
    @return: The complete solution with fitness evaluated.
    """
    source_results = []  # TODO: run simulation
    scenario = ScenarioDefinition(toolbox.clone(solution[0]))
    for perturbation in solution[1]:
        perturbation.perturb(scenario)
    follow_up_results = []  # TODO: run simulation
    # TODO: calculate fitness
    solution.fitness.values = random.uniform(0.0, 100.0),  # For test
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
    individual.fitness.values = max(involved),
    return individual


toolbox.register("evaluate_joint", _evaluate_complete_solution)
toolbox.register("evaluate_individual", _evaluate_individual)


def _fitness_sharing(population, index):
    """Adjust the individual fitness using fitness sharing.

    @param population: The population whose fitness needs to be adjusted.
    @param index: 0 for the population of scenarios, 1 for the population of perturbations.
    @return: The population with fitness adjusted.
    """
    individuals = [[ScenarioDefinition(scenario)] for scenario in population] if index == 0 \
        else [[Perturbation.squash(perturbation)] for perturbation in population]
    dist_matrix = squareform(pdist(individuals, lambda x, y: x[0].dist(y[0])))
    max_dist = np.max(dist_matrix)
    radius = max_dist / (2 * len(population))
    sh = np.vectorize(
        lambda raw: 1 - pow(raw / radius, PUNISHMENT_FACTOR) if raw < radius else 0)
    dist_matrix = sh(dist_matrix)
    dist_sum = dist_matrix.sum(axis=1)
    for i, individual in enumerate(population):
        raw_fitness = individual.fitness.values[0]
        individual.fitness.values = pow(raw_fitness, SCALING_FACTOR) / dist_sum[i],


toolbox.register("fitness_sharing", _fitness_sharing)
