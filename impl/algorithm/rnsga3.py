from collections import namedtuple
from itertools import chain

import numpy as np
from deap import tools
from deap.tools import sortNondominated, sortLogNondominated
from deap.tools.emo import find_extreme_points, find_intercepts, associate_to_niche, niching

RNSGA3Memory = namedtuple("RNSGA3Memory", ["best_point", "worst_point", "extreme_points"])


class selRNSGA3WithMemory:
    """Class version of R-NSGA-III selection including memory for best, worst and
    extreme points. Registering this operator in a toolbox is a bit different
    than classical operators, it requires to instantiate the class instead
    of just registering the function:

        >>> from deap import base
        >>> ref_points = [[1.0, 0.5, 0.2], [0.3, 0.2, 0.6]]
        >>> toolbox = base.Toolbox()
        >>> toolbox.register("select", selRNSGA3WithMemory(ref_points))

    """

    def __init__(self, ref_points, p, mu=0.05, nd="log"):
        self.ref_points = np.array(ref_points)
        self.ref_dirs = tools.uniform_reference_points(self.ref_points.shape[1], p)
        self.mu = mu
        self.nd = nd
        self.best_point = np.full((1, ref_points.shape[1]), np.inf)
        self.worst_point = np.full((1, ref_points.shape[1]), -np.inf)
        self.extreme_points = None

    def __call__(self, individuals, k):
        chosen, memory = selRNSGA3(individuals, k, self.ref_points, self.ref_dirs, self.mu, self.nd,
                                   self.best_point, self.worst_point, self.extreme_points, True)
        self.best_point = memory.best_point.reshape((1, -1))
        self.worst_point = memory.worst_point.reshape((1, -1))
        self.extreme_points = memory.extreme_points
        return chosen


def selRNSGA3(individuals, k, ref_points, ref_dirs, mu=0.05, nd="log",
              best_point=None, worst_point=None, extreme_points=None, return_memory=False):
    """Implementation of R-NSGA-III selection.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param ref_points: Reference points to use for niching.
    :param ref_dirs: Reference points uniformly on the hyperplane intersecting each axis at 1.
    :param mu: Defines the init_simplex_scale of the reference lines used during survival selection.
        Increasing mu will result having solutions with a larger spread.
    :param nd: Specify the non-dominated algorithm to use: 'standard' or 'log'.
    :param best_point: Best point found at previous generation. If not provided
        find the best point only from current individuals.
    :param worst_point: Worst point found at previous generation. If not provided
        find the worst point only from current individuals.
    :param extreme_points: Extreme points found at previous generation. If not provided
        find the extreme points only from current individuals.
    :param return_memory: If :data:`True`, return the best, worst and extreme points
        in addition to the chosen individuals.
    :returns: A list of selected individuals.
    :returns: If `return_memory` is :data:`True`, a namedtuple with the
        `best_point`, `worst_point`, and `extreme_points`.
    """
    if nd == "standard":
        pareto_fronts = sortNondominated(individuals, k)
    elif nd == "log":
        pareto_fronts = sortLogNondominated(individuals, k)
    else:
        raise Exception("selRNSGA3: The choice of non-dominated sorting "
                        "method '{0}' is invalid.".format(nd))

    # Extract fitnesses as a numpy array in the nd-sort order.
    # Use wvalues * -1 to tackle always as a minimization problem.
    fitnesses = np.array([ind.fitness.wvalues for f in pareto_fronts for ind in f])
    fitnesses *= -1

    # Get best and worst point of population, contrary to pymoo we don't use memory.
    if best_point is not None and worst_point is not None:
        best_point = np.min(np.vstack((fitnesses, best_point, ref_points)), axis=0)
        worst_point = np.max(np.vstack((fitnesses, worst_point, ref_points)), axis=0)
    else:
        best_point = np.min(np.vstack((fitnesses, ref_points)), axis=0)
        worst_point = np.max(np.vstack((fitnesses, ref_points)), axis=0)

    extreme_points = find_extreme_points(fitnesses, best_point, extreme_points)
    front_worst = np.max(fitnesses[:sum(len(f) for f in pareto_fronts), :], axis=0)
    intercepts = find_intercepts(extreme_points, best_point, worst_point, front_worst)

    # R-NSGA-III.
    unit_ref_points = (ref_points - best_point) / (intercepts - best_point)
    aspiration_ref_dirs = _get_ref_dirs_from_points(unit_ref_points, ref_dirs, mu=mu)
    aspiration_ref_points = _denormalize(aspiration_ref_dirs, best_point, intercepts)

    niches, dist = associate_to_niche(fitnesses, aspiration_ref_points, best_point, intercepts)

    # Get counts per niche for individuals in all front but the last.
    niche_counts = np.zeros(len(aspiration_ref_points), dtype=np.int64)
    index, counts = np.unique(niches[:-len(pareto_fronts[-1])], return_counts=True)
    niche_counts[index] = counts

    # Choose individuals from all fronts but the last.
    chosen = list(chain(*pareto_fronts[:-1]))

    # Use niching to select the remaining individuals.
    sel_count = len(chosen)
    n = k - sel_count
    selected = niching(pareto_fronts[-1], n, niches[sel_count:], dist[sel_count:], niche_counts)
    chosen.extend(selected)

    if return_memory:
        return chosen, RNSGA3Memory(best_point, worst_point, extreme_points)
    return chosen


def _get_ref_dirs_from_points(ref_point, ref_dirs, mu=0.1):
    """
    This function takes user specified reference points, and creates smaller sets of equidistant
    Das-Dennis points around the projection of user points on the Das-Dennis hyperplane.

    :param ref_point: List of user specified reference points.
    :param ref_dirs: List of uniformly distributed reference points.
    :param mu: Shrinkage factor (0-1), Smaller = tigher convergence, Larger= larger convergence.
    :return: Set of reference points.
    """
    n_obj = ref_point.shape[1]
    val = []
    n_vector = np.ones(n_obj) / np.sqrt(n_obj)  # Normal vector of Das Dennis plane.
    point_on_plane = np.eye(n_obj)[0]  # Point on Das-Dennis.

    for point in ref_point:
        ref_dir_for_aspiration_point = np.copy(ref_dirs)  # Copy of computed reference directions.
        ref_dir_for_aspiration_point = mu * ref_dir_for_aspiration_point

        cent = np.mean(ref_dir_for_aspiration_point, axis=0)  # Find centroid of shrunken reference points.

        # Project shrunken Das-Dennis points back onto original Das-Dennis hyperplane.
        intercept = _line_plane_intersection(np.zeros(n_obj), point, point_on_plane, n_vector)
        shift = intercept - cent  # Shift vector.

        ref_dir_for_aspiration_point += shift

        # If reference directions are located outside of first octant, redefine points onto the border.
        if not (ref_dir_for_aspiration_point > 0).min():
            ref_dir_for_aspiration_point[ref_dir_for_aspiration_point < 0] = 0
            ref_dir_for_aspiration_point = ref_dir_for_aspiration_point / np.sum(ref_dir_for_aspiration_point, axis=1)[
                                                                          :, None]
        val.extend(ref_dir_for_aspiration_point)

    val.extend(np.eye(n_obj))  # Add extreme points.
    return np.array(val)


def _line_plane_intersection(l0, l1, p0, p_no, epsilon=1e-6):
    """
    l0, l1: define the line.
    p0, p_no: define the plane:
        p0 is a point on the plane (plane coordinate).
        p_no is a normal vector defining the plane direction;
             (does not need to be normalized).

    reference: https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
    return a Vector or None (when the intersection can't be found).
    """

    l = l1 - l0
    dot = np.dot(l, p_no)

    if abs(dot) > epsilon:
        # the factor of the point between p0 -> p1 (0 - 1)
        # if 'fac' is between (0 - 1) the point intersects with the segment.
        # otherwise:
        #  < 0.0: behind p0.
        #  > 1.0: in front of p1.
        w = p0 - l0
        d = np.dot(w, p_no) / dot
        l = l * d
        return l0 + l
    else:
        # The segment is parallel to plane then return the perpendicular projection.
        ref_proj = l1 - (np.dot(l1 - p0, p_no) * p_no)
        return ref_proj


def _denormalize(x, xl=None, xu=None):
    # If both xl and xu are None we are basically done because normalization is disabled.
    if x is None or (xl is None and xu is None):
        return x

    # If not set simply fall back no nan values.
    if xl is None:
        xl = np.full_like(xu, np.nan)
    if xu is None:
        xu = np.full_like(xl, np.nan)

    xl, xu = np.copy(xl).astype(float), np.copy(xu).astype(float)

    # If both are equal then set the upper bound to none (always the 0 or lower bound will be returned then).
    xu[xl == xu] = np.nan

    # Check out when the input values are nan.
    xl_nan, xu_nan = np.isnan(xl), np.isnan(xu)

    # Now create all the masks that are necessary.
    xl_only, xu_only = np.logical_and(~xl_nan, xu_nan), np.logical_and(xl_nan, ~xu_nan)
    both_nan = np.logical_and(np.isnan(xl), np.isnan(xu))
    neither_nan = ~both_nan

    # If neither is nan than xu must be greater or equal than xl.
    any_nan = np.logical_or(np.isnan(xl), np.isnan(xu))
    assert np.all(np.logical_or(xu >= xl, any_nan)), "xl must be less or equal than xu."

    X = x.copy()
    X[..., neither_nan] = xl[neither_nan] + x[..., neither_nan] * (xu[neither_nan] - xl[neither_nan])
    X[..., xl_only] = x[..., xl_only] + xl[xl_only]
    X[..., xu_only] = xu[xu_only] - (1.0 - x[..., xu_only])
    return X
