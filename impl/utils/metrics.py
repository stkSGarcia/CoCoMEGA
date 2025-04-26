from copy import deepcopy
import numpy as np
from scipy.spatial.distance import pdist

def pairwise_distance(solutions):
    """
    Calculate the pairwise distance between solutions by perturbing each scenario.

    :param solutions: A list of solution tuples where each tuple consists of a scenario and a perturbation object.
    :return: A pairwise distance matrix of the follow-up scenarios.
    """
    if len(solutions) < 2: return np.nan
    follow_ups = []
    for scenario, perturbation in solutions:
        scenario_copy = deepcopy(scenario)
        perturbation.perturb(scenario_copy)
        follow_ups.append(scenario_copy)
    return pdist(np.array(follow_ups, dtype=object).reshape((len(follow_ups), -1)), lambda x, y: x[0].dist(y[0]))


def num_sols(solutions):
    """
    Return the number of solutions in the list.

    :param solutions: A list of solutions.
    :return: The number of solutions in the list.
    """
    return len(solutions)


def avg_fitness(solutions):
    """
    Calculate the average fitness of a list of solutions.

    :param solutions: A list of solutions with fitness values.
    :return: The average fitness value of the solutions.
    """
    return np.mean([sol.fitness.values[0] for sol in solutions])


def avg_pw(solutions):
    """
    Calculate the average pairwise distance between solutions.

    :param solutions: A list of solutions.
    :return: The average pairwise distance.
    """
    return np.mean(pairwise_distance(solutions))

def avg_pw_from_matrix(dist_matrix, indices=None):
    """
    Compute the average pairwise distance from a precomputed distance matrix.

    :param dist_matrix: A precomputed distance matrix.
    :param indices: The indices of the solutions to consider. If :data:`None`, all solutions are considered.
    :return: The average pairwise distance between the selected solutions.
    """
    if indices is None:
        indices = list(range(dist_matrix.shape[0]))
    if len(indices) < 2:
        return np.nan
    subset_dist = dist_matrix[np.ix_(indices, indices)]
    return np.mean(subset_dist[np.triu_indices(len(indices), k=1)])

def pure_div(solutions):
    """
    Calculate the pure diversity of solutions.

    :param solutions: A list of solution tuples where each tuple consists of a scenario and a perturbation object.
    :return: The pure diversity score of the population.
    """
    if len(solutions) < 2: return np.nan
    follow_ups = []
    for scenario, perturbation in solutions:
        scenario_copy = deepcopy(scenario)
        perturbation.perturb(scenario_copy)
        follow_ups.append(scenario_copy)
    from impl.algorithm.base import BaseAlgorithm
    return BaseAlgorithm.population_diversity(follow_ups)

metrics = {
    'distinct_solution_num': num_sols,
    'avg_fit': avg_fitness,
    'avg_pw': avg_pw,
    'pure_div': pure_div,
}
