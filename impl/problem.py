import math
import random

import numpy as np
from deap import creator, base, tools
from scipy.spatial.distance import pdist, squareform

from impl.config import CONFIG
from impl.mr.mr import Perturbation
from impl.mr.predefined import *
from impl.scenario.scenario_definition import ScenarioDefinition
from impl.scenario.simulation_runner import run_scenarios

mr_set = mr_set1

# Define individuals.
creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Solution", tuple, fitness=creator.Fitness, is_violated=False)
creator.create("Scenario", ScenarioDefinition, fitness=creator.Fitness)
creator.create("Perturbation", list, fitness=creator.Fitness)

toolbox = base.Toolbox()
toolbox.register("scenario", tools.initIterate, creator.Scenario, creator.Scenario.generate_random)
toolbox.register("perturbation", tools.initIterate, creator.Perturbation,
                 lambda: [random.choice(mr_set.mrs).generate_perturbation()])
# TODO: Initialize diverse individuals
toolbox.register("pop_scenario", tools.initRepeat, list, toolbox.scenario,
                 n=CONFIG["scenario"]["pop_size"] * CONFIG["scenario"]["init_selection_factor"])
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
    if random.random() > CONFIG["perturbation"]["mutpb"]: return individual
    if random.random() < CONFIG["perturbation"]["mut_del"]:
        # Remove one previous perturbation.
        if len(individual) > 1: individual.pop()
    else:
        # Add perturbations.
        times = 1
        while random.random() < CONFIG["perturbation"]["mut_add"] ** times:
            perturbation = random.choice(mr_set.mrs).generate_perturbation()
            individual.append(perturbation)
            times += 1
    return individual


toolbox.register("mutate_scenario", ScenarioDefinition.mutate)
toolbox.register("mutate_perturbation", _mutate_perturbation)


def _determine_individual_type(individual):
    if str(type(individual)) == str(creator.Scenario):
        # if isinstance(individual, creator.Scenario):
        return 0
    elif str(type(individual)) == str(creator.Perturbation):
        # elif isinstance(individual, creator.Perturbation):
        return 1
    else:
        raise ValueError(f"Unrecognized individual type: {type(individual)}.")


toolbox.register("operators",
                 lambda individual: (toolbox.select_scenario, toolbox.mate_scenario, toolbox.mutate_scenario) \
                     if _determine_individual_type(individual) == 0 \
                     else (toolbox.select_perturbation, toolbox.mate_perturbation, toolbox.mutate_perturbation))


def _fitness(source, follow_up, mr_set=mr_set):
    """Calculate the fitness value and check if it violates the relation.

    @param source: Simulation results of the source scenario.
    @param follow_up: Simulation results of the follow-up scenario.
    @return: A tuple containing a bool value indicating whether it violates the relation and the fitness value.
    """
    field = mr_set.field()
    if field == "velocity":
        func = lambda row: math.sqrt(row.velocity_x ** 2 + row.velocity_y ** 2)
        source["velocity"] = source.apply(func, axis=1)
        follow_up["velocity"] = follow_up.apply(func, axis=1)
    idx = (source[field] - follow_up[field]).abs().idxmax()
    is_violated, extent = mr_set.is_violated(source.loc[idx, field], follow_up.loc[idx, field])
    return is_violated, (extent,)


def _evaluate_solutions(solutions):
    """Evaluate the complete solutions.

    @return: A list of complete solutions evaluated and the number of simulations.
    """
    scenarios = []
    for solution in solutions:
        scenarios.append(solution[0])
        follow_up = toolbox.clone(solution[0])
        follow_up.assign_new_id()
        for perturbation in solution[1]:
            perturbation.perturb(follow_up)
        scenarios.append(follow_up)
    assert len(scenarios) == len(solutions) * 2
    results, sim_num = run_scenarios(scenarios)
    for solution, source, follow_up in zip(solutions, results[::2], results[1::2]):
        if source is not None and follow_up is not None:
            solution.is_violated, solution.fitness.values = _fitness(source, follow_up)
        else:
            solution.is_violated = False
            del solution.fitness.values
    return solutions, sim_num


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
    if len(involved) > 0:
        individual.fitness.values = max(involved),
    else:
        del individual.fitness.values
    return individual


toolbox.register("evaluate_solutions", _evaluate_solutions)
toolbox.register("evaluate_individual", _evaluate_individual)


def _prepare_ind_for_dist(population):
    assert len(population) > 0
    if _determine_individual_type(population[0]) == 0:
        return np.reshape(population, (-1, 1))
    else:
        return np.array([[Perturbation.squash(perturbation)] for perturbation in population])


toolbox.register("prepare_ind_for_dist", _prepare_ind_for_dist)


def _fitness_sharing(population):
    """Adjust the individual fitness using fitness sharing.

    @param population: The population whose fitness needs to be adjusted.
    @return: The population with fitness adjusted.
    """
    individuals = _prepare_ind_for_dist(population)
    dist_matrix = squareform(pdist(individuals, lambda x, y: x[0].dist(y[0])))
    max_dist = np.max(dist_matrix)
    radius = max_dist / (2 * len(population))
    sh = np.vectorize(lambda raw: 1 - pow(raw / radius, CONFIG["punishment"]) if raw < radius else 0)
    dist_matrix = sh(dist_matrix)
    if radius == 0.0:
        np.fill_diagonal(dist_matrix, 1.0)
    dist_sum = dist_matrix.sum(axis=1)
    for i, individual in enumerate(population):
        if individual.fitness.valid:
            raw_fitness = individual.fitness.values[0]
            individual.fitness.values = pow(raw_fitness, CONFIG["scaling"]) / dist_sum[i],


toolbox.register("fitness_sharing", _fitness_sharing)
