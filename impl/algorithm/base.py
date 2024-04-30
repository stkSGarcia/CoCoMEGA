import logging
import random
import time
from abc import abstractmethod
from operator import attrgetter
from typing import List

import numpy as np
from deap import base, creator, tools
from scipy.spatial.distance import pdist, squareform

logger = logging.getLogger(__name__)


class Budget:
    def __init__(self, max_sim, max_time, max_gen, convergence_threshold):
        """Constructor.

        @param max_sim: The maximum number of simulations for the search.
        @param max_time: The maximum execution time for the search.
        @param max_gen: The maximum number of iterations for the search.
        @param convergence_threshold: The minimum percentage of individuals in the archive that violate the given MRs.
        """
        self._max_sim = max_sim
        self._max_time = max_time
        self._max_gen = max_gen
        self.convergence_threshold = convergence_threshold
        self.sim_num = None
        self.start_time = None
        self.gen_num = None

    def initialize(self):
        self.sim_num = 0
        self.start_time = time.perf_counter()
        self.gen_num = 0

    def acc_sim(self, n):
        self.sim_num += n

    def acc_gen(self):
        self.gen_num += 1

    def is_reached(self):
        """Determine if the budget is reached.

        @return: Return `True` if the budget is reached, `False` otherwise.
        """
        return ((self._max_sim is not None and self.sim_num > self._max_sim) or
                (self._max_time is not None and time.perf_counter() - self.start_time > self._max_time) or
                (self._max_gen is not None and self.gen_num > self._max_gen))

    def print_budget(self):
        return f"Budget: max simulations: {self._max_sim}, max time: {self._max_time}, max generations: {self._max_gen}."


class BaseAlgorithm:
    def __init__(self, toolbox: base.Toolbox, budget: Budget, seed=None):
        """Constructor.

        @param toolbox: `base.Toolbox` that defines the problem.
        @param budget: `Budget` that defines the searching budget.
        @param seed: Random seed.
        """
        self.toolbox = toolbox
        self.budget = budget
        random.seed(seed)

        # Replace the original `dominates` function.
        if getattr(creator.Fitness, "dominates", None) is not None:
            logger.debug("Replace `dominates` function of fitness.")
            setattr(creator.Fitness, "dominates", BaseAlgorithm._dominates)

        # Initialize statistics object.
        self.stats = tools.Statistics(lambda ind: ind.fitness.values if ind.fitness.valid else (np.nan,))
        self.stats.register("avg", np.nanmean, axis=0)
        self.stats.register("std", np.nanstd, axis=0)
        self.stats.register("min", np.nanmin, axis=0)
        self.stats.register("max", np.nanmax, axis=0)

        self.logbook = tools.Logbook()
        self.logbook.header = "pop", "gen", "len", "std", "min", "avg", "max"

    @abstractmethod
    def solve(self):
        """Run the algorithm."""
        raise NotImplementedError

    def record_statistics(self, population: List, gen_num: int, pop_name: str = ""):
        """Record the statistics of the population.

        @param population: The population that requires recording statistics.
        @param gen_num: The number of generations.
        @param pop_name: The name of the population.
        """
        record = self.stats.compile(population) if len(population) > 0 \
            else {"avg": [np.nan], "std": [np.nan], "min": [np.nan], "max": [np.nan]}
        self.logbook.record(pop=pop_name, gen=gen_num, len=len(population), **record)

    def fitness_sharing(self, population, punishment=1.0, scaling=1.0):
        """Adjust the fitness using fitness sharing.

        @param population: The population whose fitness needs to be adjusted.
        @param punishment: Punishment factor.
        @param scaling: Scaling factor.
        @return: The population with fitness adjusted.
        """
        dist_matrix = squareform(pdist(self.toolbox.prepare_ind_for_dist(population), lambda x, y: x[0].dist(y[0])))
        max_dist = np.max(dist_matrix)
        radius = max_dist / (2 * len(population))  # TODO: to be justified.
        sharing_func = np.vectorize(lambda raw: 1 - pow(raw / radius, punishment) if raw < radius else 0)
        dist_matrix = sharing_func(dist_matrix)
        if radius == 0.0:
            np.fill_diagonal(dist_matrix, 1.0)
        dist_sum = dist_matrix.sum(axis=1)
        for i, individual in enumerate(population):
            if individual.fitness.valid:
                raw_fitness = individual.fitness.values[0]
                individual.fitness.values = pow(raw_fitness, scaling) / dist_sum[i],

    def fitness_clearing(self, population, capacity):
        """Adjust the fitness using fitness clearing.

        @param population: The population whose fitness needs to be adjusted.
        @return: The population with fitness adjusted.
        """
        dist_matrix = squareform(pdist(self.toolbox.prepare_ind_for_dist(population), lambda x, y: x[0].dist(y[0])))
        max_dist = np.max(dist_matrix)
        radius = max_dist / (2 * len(population))  # TODO: to be justified.

        individuals = sorted(population, key=attrgetter("fitness"), reverse=True)
        for i in range(len(individuals)):
            if individuals[i].fitness.valid:
                winner_count = 1
                for j in range(i + 1, len(individuals)):
                    if individuals[j].fitness.valid and dist_matrix[i][j] < radius:
                        if winner_count < capacity:
                            winner_count += 1
                        else:
                            del individuals[j].fitness.values

    def population_diversity(self, population):
        """Calculate the Pure Diversity (PD) of the given population."""
        n = len(population)
        individuals = self.toolbox.prepare_ind_for_dist(population)
        connected = np.eye(n, dtype=bool)
        dist_matrix = squareform(pdist(individuals, lambda x, y: x[0].dist(y[0])))
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
