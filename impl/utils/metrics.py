from copy import deepcopy
import numpy as np
from scipy.spatial.distance import pdist

def pairwise_distance(solutions):
    if len(solutions) < 2: return np.nan
    follow_ups = []
    for scenario, perturbation in solutions:
        scenario_copy = deepcopy(scenario)
        perturbation.perturb(scenario_copy)
        follow_ups.append(scenario_copy)
    return pdist(np.array(follow_ups, dtype=object).reshape((len(follow_ups), -1)), lambda x, y: x[0].dist(y[0]))


def num_sols(solutions):
    return len(solutions)


def avg_fitness(solutions):
    return np.mean([sol.fitness.values[0] for sol in solutions])


def avg_pw(solutions):
    return np.mean(pairwise_distance(solutions))

def avg_pw_from_matrix(dist_matrix, indices):
    """Computes the average pairwise distance from a precomputed distance matrix."""
    if len(indices) < 2:
        return np.nan
    subset_dist = dist_matrix[np.ix_(indices, indices)]
    return np.mean(subset_dist[np.triu_indices(len(indices), k=1)])

def pure_div(solutions):
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