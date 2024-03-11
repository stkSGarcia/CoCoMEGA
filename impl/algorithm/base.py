import logging
import random
from abc import abstractmethod
from typing import List

import numpy as np
from deap import base, creator, tools
from scipy.spatial.distance import pdist, squareform

logger = logging.getLogger(__name__)


class BaseAlgorithm:
    def __init__(self,
                 toolbox: base.Toolbox,
                 time_budget,
                 max_iter,
                 seed=None):
        """Constructor.

        @param toolbox: `base.Toolbox` that defines the problem.
        @param time_budget: The maximum execution time for the search.
        @param max_iter: The maximum number of iterations for the search.
        @param seed: Random seed.
        """
        self.toolbox = toolbox
        self.time_budget = time_budget
        self.max_iter = max_iter
        random.seed(seed)

        # Replace the original `dominates` function.
        if getattr(creator.Fitness, "dominates", None) is not None:
            logger.info("Replace `dominates` function of fitness.")
            setattr(creator.Fitness, "dominates", BaseAlgorithm._dominates)

        # Initialize statistics object.
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean, axis=0)
        self.stats.register("std", np.std, axis=0)
        self.stats.register("min", np.min, axis=0)
        self.stats.register("max", np.max, axis=0)

        self.logbook = tools.Logbook()
        self.logbook.header = "pop", "gen", "evals", "std", "min", "avg", "max"

    @abstractmethod
    def solve(self):
        """Run the algorithm."""
        raise NotImplementedError

    def _record_statistics(self, population: List, num_of_generation: int, pop_name: str = ""):
        """Record the statistics of the population."""
        record = self.stats.compile(population)
        self.logbook.record(pop=pop_name, gen=num_of_generation, evals=len(population), **record)

    @staticmethod
    def population_diversity(population):
        """Calculate the Pure Diversity (PD) of the given population."""
        n = len(population)
        connected = np.eye(n, dtype=bool)
        dist_matrix = squareform(pdist(population, lambda x, y: x[0].heterogeneous_distance(y[0])))
        np.fill_diagonal(dist_matrix, np.inf)
        pd = 0
        for _ in range(n - 1):
            while True:
                d, indices = np.min(dist_matrix, axis=1), np.argmin(dist_matrix, axis=1)
                i = np.argmax(d)
                j = indices[i]
                if dist_matrix[j, i] != np.inf:
                    dist_matrix[j, i] = np.inf
                if dist_matrix[i, j] != np.inf:
                    dist_matrix[i, j] = np.inf
                p = connected[i, :]
                while not p[j]:
                    new_p = np.any(connected[p, :], axis=0)
                    if np.all(new_p == p):
                        break
                    else:
                        p = new_p
                if not p[j]: break
            connected[i, j] = True
            connected[j, i] = True
            dist_matrix[i, :] = -np.inf
            pd += d[i]
        return pd

    @staticmethod
    def _dominates(this, other, obj: List = None):
        """DO NOT CALL THIS FUNCTION.
        It is used to replace the original `dominates` function in `deap`.

        @param obj: Indices indicating on which objectives the domination is tested.
        """
        if not obj:
            obj = slice(None)
        not_equal = False
        for self_wvalue, other_wvalue in zip(np.array(this.wvalues)[obj], np.array(other.wvalues)[obj]):
            if self_wvalue > other_wvalue:
                not_equal = True
            elif self_wvalue < other_wvalue:
                return False
        return not_equal
