from collections import namedtuple

import numpy as np
from deap.tools import sortNondominated, sortLogNondominated

RNSGA2Memory = namedtuple("RNSGA2Memory", ["ref_points", "ideal_point", "nadir_point"])


class selRNSGA2WithMemory:
    """Class version of R-NSGA-II selection including memory for ideal and nadir points.
    Registering this operator in a toolbox is a bit different from classical operators,
    it requires instantiating the class instead of just registering the function:
        >>> from deap import base
        >>> ref_points = [[0.5, 0.2], [0.3, 0.2]]
        >>> toolbox = base.Toolbox()
        >>> toolbox.register("select", selRNSGA2WithMemory(ref_points))
    """

    def __init__(self, ref_points, epsilon=0.001, normalization="front", weights=None,
                 extreme_points_as_reference_points=False, nd="log"):
        """Initialize selection with reference points and parameters. 
        See also :func:`selRNSGA2` for more details.

        :param ref_points: Reference points for niching.
        :param epsilon: Threshold for diversity control.
        :param normalization: Type of normalization (:data:`ever`, :data:`front`, or :data:`no`).
        :param weights: Weight vector for objectives.
        :param extreme_points_as_reference_points: Whether to add extreme points to reference points.
        :param nd: Non-dominated sorting method (:data:`standard` or :data:`log`).
        """
        self.ref_points = np.array(ref_points)
        n_obj = self.ref_points.shape[1]  # Number of objectives.
        self.ideal_point = np.full(n_obj, np.inf)
        self.nadir_point = np.full(n_obj, -np.inf)
        self.epsilon = epsilon
        self.normalization = normalization
        self.weights = weights
        if self.weights is None:
            self.weights = np.full(n_obj, 1 / n_obj)
        self.extreme_points_as_reference_points = extreme_points_as_reference_points
        self.nd = nd

    def __call__(self, individuals, k):
        chosen, memory = selRNSGA2(individuals, k, self.ref_points, self.ideal_point, self.nadir_point, self.epsilon,
                                   self.normalization, self.weights, self.extreme_points_as_reference_points, self.nd,
                                   True)
        self.ref_points = memory.ref_points
        self.ideal_point = memory.ideal_point
        self.nadir_point = memory.nadir_point
        return chosen


def selRNSGA2(individuals, k, ref_points, ideal_point, nadir_point, epsilon=0.01, normalization="front",
              weights=None, extreme_points_as_reference_points=False, nd="log", return_memory=False):
    """Implementation of R-NSGA-II selection.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param ref_points: Reference points to use for niching.
    :param ideal_point: Ideal point found at previous generation.
    :param nadir_point: Nadir point found at previous generation.
    :param epsilon: Epsilon parameter for crowding distance calculation.
    :param normalization: Normalization method to use: :data:`ever` or :data:`front` or :data:`no`.
    :param weights: Weights for each objective.
    :param extreme_points_as_reference_points: If :data:`True`, extreme points are added to the reference points.
    :param nd: Specify the non-dominated algorithm to use: :data:`standard` or :data:`log`.
    :param return_memory: If :data:`True`, return the reference, ideal, and nadir points
        in addition to the chosen individuals.
    :returns: A list of selected individuals.
        If ``return_memory`` is :data:`True`, a namedtuple with the
        ``ref_points``, ``ideal_point``, and ``nadir_points``.
    """
    if nd == "standard":
        pareto_fronts = sortNondominated(individuals, k)
    elif nd == "log":
        pareto_fronts = sortLogNondominated(individuals, k)
    else:
        raise Exception(f"selRNSGA2: The choice of non-dominated sorting method '{nd}' is invalid.")

    n_obj = ref_points.shape[1]  # Number of objectives.

    # Extract fitnesses as a numpy array in the nd-sort order.
    # Use wvalues * -1 to tackle always as a minimization problem.
    fitnesses = []
    for front in pareto_fronts:
        front_fitnesses = np.array([ind.fitness.wvalues for ind in front])
        front_fitnesses *= -1
        fitnesses.append(front_fitnesses)

    if normalization == "ever":
        # Find or usually update the new ideal point - from feasible solutions.
        ideal_point = np.min(np.vstack((ideal_point, np.vstack(fitnesses))), axis=0)
        nadir_point = np.max(np.vstack((nadir_point, np.vstack(fitnesses))), axis=0)
    elif normalization == "front":
        if len(pareto_fronts[0]) > 1:
            ideal_point = np.min(fitnesses[0], axis=0)
            nadir_point = np.max(fitnesses[0], axis=0)
    elif normalization == "no":
        ideal_point = np.zeros(n_obj)
        nadir_point = np.ones(n_obj)

    if extreme_points_as_reference_points:
        ref_points = np.vstack([ref_points, _get_extreme_points_c(np.vstack(fitnesses), ideal_point)])

    chosen = []

    for i, front in enumerate(pareto_fronts):
        # Number of individuals remaining.
        n_remaining = k - len(chosen)

        # Calculate the distance matrix from every solution to all reference points.
        dist_to_ref_points = _calc_norm_pref_distance(fitnesses[i], ref_points, weights, ideal_point, nadir_point)

        # The ranking of each point regarding each reference point (two times argsort is necessary).
        rank_by_distance = np.argsort(np.argsort(dist_to_ref_points, axis=0), axis=0)

        # The reference point where the best ranking is coming from.
        ref_point_of_best_rank = np.argmin(rank_by_distance, axis=1)

        # The actual ranking which is used as crowding.
        ranking = rank_by_distance[np.arange(len(front)), ref_point_of_best_rank]

        if len(front) <= n_remaining:
            # we can simply copy the crowding to ranking. not epsilon selection here
            crowding = ranking
            I = np.arange(len(front))
        else:
            # Distance from solution to every other solution and set distance to itself to infinity.
            dist_to_others = _calc_norm_pref_distance(fitnesses[i], fitnesses[i], weights, ideal_point, nadir_point)
            np.fill_diagonal(dist_to_others, np.inf)

            # The crowding that will be used for selection.
            crowding = np.full(len(front), np.nan)

            # Solutions which are not already selected.
            not_selected = np.argsort(ranking)

            # Until we have saved a crowding for each solution.
            while len(not_selected) > 0:
                # Select the closest solution.
                idx = not_selected[0]

                # Set crowding for that individual.
                crowding[idx] = ranking[idx]

                # Need to remove myself from the not-selected array.
                to_remove = [idx]

                # Group of close solutions.
                dist = dist_to_others[idx][not_selected]
                group = not_selected[np.where(dist < epsilon)[0]]

                # If there exists a solution with a distance less than epsilon.
                if len(group):
                    # Discourage them by giving them a high crowding.
                    crowding[group] = ranking[group] + np.round(len(front) / 2)

                    # Remove the group from the not_selected array.
                    to_remove.extend(group)

                not_selected = np.array([i for i in not_selected if i not in to_remove])

            # Now sort by the crowding (actually modified rank) ascending and let the best survive.
            I = np.argsort(crowding)[:n_remaining]

        # Set the crowding to all individuals.
        for ind, c in zip(front, crowding):
            # Inverse of crowding because nsga2 does maximize it (then tournament selection can stay the same).
            ind.fitness.crowding_dist = -c

        # Extend the survivors by all or selected individuals.
        chosen.extend([front[i] for i in I])

    if return_memory:
        return chosen, RNSGA2Memory(ref_points, ideal_point, nadir_point)
    return chosen


def _calc_norm_pref_distance(A, B, weights, ideal, nadir):
    D = np.repeat(A, B.shape[0], axis=0) - np.tile(B, (A.shape[0], 1))
    N = ((D / (nadir - ideal)) ** 2) * weights
    N = np.sqrt(np.sum(N, axis=1) * len(weights))
    return np.reshape(N, (A.shape[0], B.shape[0]))


def _get_extreme_points_c(F, ideal_point, extreme_points=None):
    # Calculate the asf which is used for the extreme point decomposition.
    weights = np.eye(F.shape[1])
    weights[weights == 0] = 1e6

    # Add the old extreme points to never lose them for normalization.
    _F = F
    if extreme_points is not None:
        _F = np.concatenate([extreme_points, _F], axis=0)

    # Use __F because we substitute small values to be 0.
    __F = _F - ideal_point
    __F[__F < 1e-3] = 0

    # Update the extreme points for the normalization having the highest asf value each.
    F_asf = np.max(__F * weights[:, None, :], axis=2)

    I = np.argmin(F_asf, axis=1)
    extreme_points = _F[I, :]

    return extreme_points
