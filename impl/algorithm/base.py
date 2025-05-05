import logging
import os
import pickle
import random
import time
from abc import abstractmethod
from operator import attrgetter
from typing import List

import numpy as np
from deap import base, creator, tools
from scipy.spatial.distance import pdist, squareform

from impl.config import CONFIG

logger = logging.getLogger(__name__)


class Budget:
    """Search budget."""

    def __init__(self, max_sim, max_time, max_gen):
        """Constructor.

        :param max_sim: The maximum number of simulations for the search.
        :param max_time: The maximum execution time for the search.
        :param max_gen: The maximum number of iterations for the search.
        """
        self.max_sim = max_sim
        self.max_time = max_time
        self.max_gen = max_gen
        self.sim_num = None  # The number of simulations actually runs.
        self.start_time = None  # The start time of the search.
        self.gen_num = None  # The current number of generations.

    def initialize(self, other=None):
        """Initialize the budget.

        :param other: Another Budget that is used to initialize the budget.
        """
        if other and isinstance(other, self.__class__):
            # Copy counters from another budget.
            self.sim_num = other.sim_num
            self.start_time = other.start_time
            self.gen_num = other.gen_num
        else:
            # Start new counters.
            self.sim_num = 0
            self.start_time = time.perf_counter()
            self.gen_num = 0

    def acc_sim(self, n):
        """Accumulate the number of simulations.

        :param n: The number of simulations.
        """
        self.sim_num += n

    def acc_gen(self):
        """ Accumulate the number of generations by one."""
        self.gen_num += 1

    def is_reached(self):
        """Determine if the budget is reached.

        :return: Return :data:`True` if the budget is reached, :data:`False` otherwise.
        """
        return ((self.max_sim is not None and self.sim_num > self.max_sim) or
                (self.max_time is not None and time.perf_counter() - self.start_time > self.max_time) or
                (self.max_gen is not None and self.gen_num > self.max_gen))

    def print_budget(self):
        """Print the budget."""
        return f"Budget: max simulations: {self.max_sim}, max time: {self.max_time}, max generations: {self.max_gen}."


class BaseAlgorithm:
    """Base class for search algorithms, providing shared utility functions and structure."""

    def __init__(self, toolbox: base.Toolbox, budget: Budget, seed=None):
        """Constructor.

        :param toolbox: :class:`deap.base.Toolbox` that defines the problem.
        :param budget: Search budget.
        :param seed: Random seed.
        """
        self.toolbox = toolbox
        self.budget = budget
        random.seed(seed)

        # Replace the original `dominates` function.
        if getattr(creator.Fitness, "dominates", None) is not None:
            logger.debug("Replace `dominates` function of fitness.")
            setattr(creator.Fitness, "dominates", BaseAlgorithm._dominates)

        # Initialize statistics collector for `deap` population.
        self.stats = tools.Statistics(lambda ind: ind.fitness.values if ind.fitness.valid else (np.nan,))
        self.stats.register("avg", np.nanmean, axis=0)
        self.stats.register("std", np.nanstd, axis=0)
        self.stats.register("min", np.nanmin, axis=0)
        self.stats.register("max", np.nanmax, axis=0)

        self.logbook = tools.Logbook()  # Logbook for recording statistics.
        self.logbook.header = "pop", "gen", "len", "sim", "std", "min", "avg", "max"

    @abstractmethod
    def solve(self, resume=False):
        """Run the algorithm.

        :param resume: Whether to resume from the latest checkpoint.
        """
        raise NotImplementedError

    def record_statistics(self, population: List, gen_num: int, pop_name: str = "", sim_num: int = None):
        """Record the statistics of the population.

        :param population: The population that requires recording statistics.
        :param gen_num: The number of generations.
        :param pop_name: The name of the population.
        :param sim_num: The number of simulations actually runs.
        """
        record = self.stats.compile(population) if len(population) > 0 \
            else {"avg": [np.nan], "std": [np.nan], "min": [np.nan], "max": [np.nan]}
        self.logbook.record(pop=pop_name, gen=gen_num, len=len(population), sim=sim_num, **record)

    def dump_results(self, results, evaluated_solutions, name=None):
        """Dump results and statistics.

        :param results: The solutions identified by the algorithm.
        :param evaluated_solutions: All the evaluated solutions during the search.
        """
        suffix = (f"{name}-" if name else "") + str(int(round(time.time() * 1000)))
        with open(os.path.join(CONFIG["workspace"]["solution"], f"solutions-{suffix}.pickle"), "wb") as f:
            pickle.dump(results, f)
        with open(os.path.join(CONFIG["workspace"]["solution"], f"evaluated-{suffix}.pickle"), "wb") as f:
            pickle.dump(evaluated_solutions, f)
        statistics_path = f"statistics-{suffix}.pickle"
        with open(os.path.join(CONFIG["workspace"]["solution"], statistics_path), "wb") as f:
            pickle.dump(self.logbook, f)
        logger.info(f"Results dumped at {suffix}.")

    def fitness_sharing(self, population, punishment=1.0, scaling=1.0):
        """Adjust the fitness using fitness sharing.

        :param population: The population whose fitness needs to be adjusted.
        :param punishment: Punishment factor.
        :param scaling: Scaling factor.
        :return: The population with fitness adjusted.
        """
        if len(population) == 0: return
        # Calculate the pairwise distance matrix.
        dist_matrix = squareform(pdist(np.array(population, dtype=object).reshape((len(population), -1)),
                                       lambda x, y: x[0].dist(y[0])))
        max_dist = np.max(dist_matrix)  # Maximum distance between any two individuals.
        radius = max_dist / (2 * len(population))  # Fitness sharing radius.
        sharing_func = np.vectorize(lambda raw: 1 - pow(raw / radius, punishment) if raw < radius else 0)
        dist_matrix = sharing_func(dist_matrix)  # Apply the sharing function to the distance matrix.
        if radius == 0.0:
            np.fill_diagonal(dist_matrix, 1.0)
        dist_sum = dist_matrix.sum(axis=1)
        for i, individual in enumerate(population):
            if individual.fitness.valid:
                raw_fitness = individual.fitness.values[0]
                individual.fitness.values = pow(raw_fitness, scaling) / dist_sum[i],

    def fitness_clearing(self, population, capacity=2):
        """Adjust the fitness using fitness clearing.

        :param population: The population whose fitness needs to be adjusted.
        :param capacity: The maximum number of winners in a niche.
        :return: The population with fitness adjusted.
        """
        if len(population) == 0: return
        # Calculate the pairwise distance matrix.
        dist_matrix = squareform(pdist(np.array(population, dtype=object).reshape((len(population), -1)),
                                       lambda x, y: x[0].dist(y[0])))
        max_dist = np.max(dist_matrix)  # Maximum distance between any two individuals.
        radius = max_dist / (2 * len(population))  # Fitness clearing radius.

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

    @staticmethod
    def population_diversity(population):
        """Calculate the Pure Diversity (PD) of the given population.

        :param population: The population to be evaluated.
        """
        n = len(population)
        if n == 0: return 0.0
        connected = np.eye(n, dtype=bool)  # Matrix denoting the connection between individuals.
        # Calculate the pairwise distance matrix.
        dist_matrix = population.copy() if isinstance(population, np.ndarray) \
            else squareform(pdist(np.array(population, dtype=object).reshape((len(population), -1)),
                                  lambda x, y: x[0].dist(y[0])))
        np.fill_diagonal(dist_matrix, np.inf)
        pd = 0.0  # PD value.
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
        It is used to replace the original :func:`dominates` function in :mod:`deap`.

        :param obj: Indices indicating on which objectives the domination is tested.
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
