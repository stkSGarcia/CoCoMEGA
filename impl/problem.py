import random

import numpy as np
from deap import creator, base, tools
from scipy.spatial.distance import pdist, squareform

from impl.config import CONFIG
from impl.mr.mr import Perturbation
from impl.mr.predefined import *
from impl.scenario.scenario_definition import ScenarioDefinition
from impl.scenario.simulation_runner import simulation_runner

mr_set = mr_set1

# Define individuals.
creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Solution", tuple, fitness=creator.Fitness)
creator.create("Scenario", ScenarioDefinition, fitness=creator.Fitness)
creator.create("Perturbation", list, fitness=creator.Fitness)

toolbox = base.Toolbox()
toolbox.register("scenario", tools.initIterate, creator.Scenario, creator.Scenario.generate_random)
toolbox.register("perturbation", tools.initIterate, creator.Perturbation,
                 lambda: [random.choice(mr_set.mrs).generate_perturbation()])
# TODO: Initialize diverse individuals
toolbox.register("pop_scenario", tools.initRepeat, list, toolbox.scenario, n=CONFIG["scenario"]["pop_size"])
toolbox.register("pop_perturbation", tools.initRepeat, list, toolbox.perturbation, n=CONFIG["perturbation"]["pop_size"])

# Create a complete solution from two individuals.
toolbox.register("collaborate", lambda scenario, perturbation: creator.Solution((scenario, perturbation)))

# Define genetic operators.
toolbox.register("select_scenario", tools.selTournament, tournsize=CONFIG["scenario"]["tournament"])
toolbox.register("select_perturbation", tools.selTournament, tournsize=CONFIG["scenario"]["tournament"])

toolbox.register("mate_scenario", lambda x, y: x.mate(y))
toolbox.register("mate_perturbation", tools.cxUniform, indpb=CONFIG["perturbation"]["cxpb"])


def _mutate_perturbation(individual):
    """Mutate a give sequence of perturbations by appending more perturbations to the sequence.

    @param individual: The sequence of perturbations (`creator.Perturbation`) to be mutated.
    @return: The mutated sequence of perturbations.
    """
    # Add perturbations.
    times = 1
    while random.random() < CONFIG["perturbation"]["mut_add"] ** times:
        perturbation = random.choice(mr_set.mrs).generate_perturbation()
        individual.append(perturbation)
        times += 1

    # Remove one previous perturbation.
    if len(individual) > 1 and random.random() < CONFIG["perturbation"]["mut_del"]:
        individual.pop()
    return individual


toolbox.register("mutate_scenario", ScenarioDefinition.mutate)
toolbox.register("mutate_perturbation", _mutate_perturbation)


def _determine_individual_type(individual):
    if isinstance(individual, creator.Scenario):
        # if str(type(individual)) == "<class 'deap.creator.Scenario'>":  # FIXME: isinstance(individual, creator.Scenario)
        return 0
    elif isinstance(individual, creator.Perturbation):
        # elif str(type(individual)) == "<class 'deap.creator.Perturbation'>":  # isinstance(individual, creator.Perturbation)
        return 1
    else:
        raise ValueError(f"Unrecognized individual type: {type(individual)}.")


toolbox.register("operators",
                 lambda individual: (toolbox.select_scenario, toolbox.mate_scenario, toolbox.mutate_scenario) \
                     if _determine_individual_type(individual) == 0 \
                     else (toolbox.select_perturbation, toolbox.mate_perturbation, toolbox.mutate_perturbation))


def _fitness(source, follow_up):  # TODO: test
    field = mr_set.field()
    idx = (source[field] - follow_up[field]).abs().idxmax()
    _, extent = mr_set.is_violated(source.loc[idx, field], follow_up.loc[idx, field])
    return extent,


def _evaluate_complete_solution(solution):
    """Evaluate the joint fitness of a complete solution.

    @param solution: The complete solutions (`creator.Individual`) to be evaluated.
    @return: The complete solution with fitness evaluated.
    """
    follow_up_scenario = toolbox.clone(solution[0])
    source = simulation_runner.run(solution[0])
    for perturbation in solution[1]:
        perturbation.perturb(follow_up_scenario)
    follow_up = simulation_runner.run(follow_up_scenario)
    solution.fitness.values = _fitness(source.results, follow_up.results)
    return solution


def _evaluate_solutions(solutions):
    # with ProcessPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
    #     candidates = executor.map(_evaluate_complete_solution, solutions)
    # evaluated_solutions = list(candidates)
    evaluated_solutions = list(map(_evaluate_complete_solution, solutions))
    return evaluated_solutions


def _evaluate_individual(individual, complete_solutions):
    """Evaluate the individual fitness of a scenario or a sequence of perturbations.

    @param individual: The individual to be evaluated.
    @param complete_solutions: The list of complete solutions with fitness evaluated.
    @return: The individual with fitness evaluated.
    """
    index = _determine_individual_type(individual)
    involved = []
    for solution in complete_solutions:
        if solution[index] == individual:
            involved.append(solution.fitness.values[0])
    individual.fitness.values = max(involved),
    return individual


toolbox.register("evaluate_solutions", _evaluate_solutions)
toolbox.register("evaluate_individual", _evaluate_individual)


def _fitness_sharing(population):
    """Adjust the individual fitness using fitness sharing.

    @param population: The population whose fitness needs to be adjusted.
    @return: The population with fitness adjusted.
    """
    assert len(population) > 0
    individuals = [[scenario] for scenario in population] if _determine_individual_type(population[0]) == 0 \
        else [[Perturbation.squash(perturbation)] for perturbation in population]
    dist_matrix = squareform(pdist(individuals, lambda x, y: x[0].dist(y[0])))
    max_dist = np.max(dist_matrix)
    radius = max_dist / (2 * len(population))
    sh = np.vectorize(lambda raw: 1 - pow(raw / radius, CONFIG["punishment"]) if raw < radius else 0)
    dist_matrix = sh(dist_matrix)
    if radius == 0.0:  # FIXME
        np.fill_diagonal(dist_matrix, 1.0)
    dist_sum = dist_matrix.sum(axis=1)
    for i, individual in enumerate(population):
        raw_fitness = individual.fitness.values[0]
        individual.fitness.values = pow(raw_fitness, CONFIG["scaling"]) / dist_sum[i],


toolbox.register("fitness_sharing", _fitness_sharing)
