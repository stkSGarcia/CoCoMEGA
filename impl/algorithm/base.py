import logging
import pickle
import random
import time
from operator import attrgetter
from typing import List

import numpy as np
from deap import base, creator, tools
from scipy.spatial.distance import pdist, squareform

from impl import config as cfg
from impl.algorithm.budget import Budget

logger = logging.getLogger(__name__)


class BaseAlgorithm:
    """Base class for search algorithms, providing shared utility functions and structure."""
    _name = "BASE"

    def __init__(self, toolbox: base.Toolbox, budget: Budget, seed=None):
        """Constructor.

        :param toolbox: `deap.base.Toolbox` that defines the problem.
        :param budget: `Budget` that defines the searching budget.
        :param seed: Random seed.
        """
        random.seed(seed)
        self.toolbox = toolbox
        self.budget = budget
        self.n_obj = len(creator.Fitness.weights)

        # Replace the original `dominates` function.
        if getattr(creator.Fitness, "dominates", None) is not None:
            logger.debug("Replace `dominates` function of fitness.")
            setattr(creator.Fitness, "dominates", BaseAlgorithm._dominates)

        # Initialize statistics object.
        self.stats = tools.Statistics(lambda ind: ind.fitness.values if ind.fitness.valid else (np.nan,) * self.n_obj)
        self.stats.register("avg", np.nanmean, axis=0)
        self.stats.register("std", np.nanstd, axis=0)
        self.stats.register("min", np.nanmin, axis=0)
        self.stats.register("max", np.nanmax, axis=0)

        self.logbook = tools.Logbook()
        self.logbook.header = "pop", "gen", "len", "sim", "std", "min", "avg", "max"

    def solve(self, resume=False):
        """Run the algorithm.

        :param resume: Whether to resume from the latest checkpoint.
        """
        logger.info(f"{self._name} started.")
        logger.info(self.budget.print_budget())
        cfg.dump_config_to_json()  # Dump configurations.

    def record_statistics(self, population: List, gen_num: int, pop_name: str = "", sim_num: int = None):
        """Record the statistics of the population.

        :param population: The population that requires recording statistics.
        :param gen_num: The number of generations.
        :param pop_name: The name of the population.
        :param sim_num: The number of simulations actually runs.
        """
        record = self.stats.compile(population) if len(population) > 0 else {
            "avg": [np.nan, ] * self.n_obj,
            "std": [np.nan, ] * self.n_obj,
            "min": [np.nan, ] * self.n_obj,
            "max": [np.nan, ] * self.n_obj,
        }
        self.logbook.record(pop=pop_name, gen=gen_num, len=len(population), sim=sim_num, **record)

    def dump_results(self, results, evaluated_solutions):
        """Dump results and statistics.

        :param results: The solutions identified by the algorithm.
        :param evaluated_solutions: All the evaluated solutions during the search.
        """
        suffix = f"{self._name.lower()}-{str(int(round(time.time() * 1000)))}"
        (cfg.CONFIG["workspace"]["solution"] / f"solutions-{suffix}.pickle").write_bytes(pickle.dumps(results))
        (cfg.CONFIG["workspace"]["solution"] / f"evaluated-{suffix}.pickle").write_bytes(
            pickle.dumps(evaluated_solutions))
        (cfg.CONFIG["workspace"]["solution"] / f"statistics-{suffix}.pickle").write_bytes(pickle.dumps(self.logbook))
        logger.info(f"Results dumped at {suffix}.")

    @staticmethod
    def remove_duplicates(population):
        """Remove duplicate individuals in the given population.

        :param population: The population requiring deduplication.
        """
        unique_individuals = []
        for ind in population:
            if ind not in unique_individuals:
                unique_individuals.append(ind)
        return unique_individuals

    @staticmethod
    def fitness_sharing(population, punishment=1.0, scaling=1.0):
        """Adjust the fitness using fitness sharing.

        :param population: The population whose fitness needs to be adjusted.
        :param punishment: Punishment factor.
        :param scaling: Scaling factor.
        :return: The population with fitness adjusted.
        """
        if len(population) == 0: return
        dist_matrix = squareform(pdist(np.array(population, dtype=object).reshape((len(population), -1)),
                                       lambda x, y: x[0].dist(y[0])))
        max_dist = np.max(dist_matrix)
        radius = max_dist / (2 * len(population))
        sharing_func = np.vectorize(lambda raw: 1 - pow(raw / radius, punishment) if raw < radius else 0)
        dist_matrix = sharing_func(dist_matrix)
        if radius == 0.0:
            np.fill_diagonal(dist_matrix, 1.0)
        dist_sum = dist_matrix.sum(axis=1)
        for i, individual in enumerate(population):
            if individual.fitness.valid:
                raw_fitness = np.array(individual.fitness.values)
                individual.fitness.values = tuple(pow(raw_fitness, scaling) / dist_sum[i])

    @staticmethod
    def fitness_clearing(population, capacity=2):
        """Adjust the fitness using fitness clearing.

        :param population: The population whose fitness needs to be adjusted.
        :param capacity: The maximum number of winners in a niche.
        :return: The population with fitness adjusted.
        """
        if len(population) == 0: return
        dist_matrix = squareform(pdist(np.array(population, dtype=object).reshape((len(population), -1)),
                                       lambda x, y: x[0].dist(y[0])))
        max_dist = np.max(dist_matrix)
        radius = max_dist / (2 * len(population))

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
        connected = np.eye(n, dtype=bool)
        dist_matrix = population.copy() if isinstance(population, np.ndarray) \
            else squareform(pdist(np.array(population, dtype=object).reshape((len(population), -1)),
                                  lambda x, y: x[0].dist(y[0])))
        np.fill_diagonal(dist_matrix, np.inf)
        pd = 0.0
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
