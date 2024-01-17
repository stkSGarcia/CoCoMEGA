import random
from abc import abstractmethod
from typing import Callable, List

import numpy as np
from deap import base, creator, tools

from impl.mr import MR


class BaseAlgorithm:
    def __init__(
            self,
            pop_size,  # initial population size
            evaluator: Callable,  # function for evaluating a population
            objectives: List[List],  # objective thresholds
            bounds: List[List],  # lower bound and upper bound of an individual
            mrs: List[MR],  # metamorphic relations
            cxpb,  # the probability of mating two individuals
            mutpb,  # the probability of mutating an individual
            time_budget,  # maximum execution time
            max_iter,  # maximum number of iterations
            seed=None  # random seed
    ):
        self.pop_size = pop_size
        self.evaluator = evaluator
        self.objectives = objectives
        self.bounds = bounds
        self.mrs = mrs
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.time_budget = time_budget
        self.max_iter = max_iter

        random.seed(seed)

        creator.create("FitnessMax", base.Fitness, weights=(1.0,) * len(objectives))  # maximize the fitness
        creator.create("Individual", list, fitness=creator.FitnessMax, covered_objectives=list)

        # Replace the original `dominates` function.
        if getattr(creator.FitnessMax, "dominates", None) is not None:
            setattr(creator.FitnessMax, "dominates", BaseAlgorithm._dominates)

        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self._initialize_vector)
        self.toolbox.register("evaluate", self._evaluate_population)
        self.toolbox.register("archive", self._update_archive)

        # Initialize statistics object
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean, axis=0)
        self.stats.register("std", np.std, axis=0)
        self.stats.register("min", np.min, axis=0)
        self.stats.register("max", np.max, axis=0)

        self.logbook = tools.Logbook()
        self.logbook.header = "gen", "evals", "std", "min", "avg", "max"

    @abstractmethod
    def solve(self):
        pass

    def _initialize_vector(self):
        vector = []
        for bound in self.bounds:
            if bound == "bool":
                vector.append(bool(random.getrandbits(1)))
            elif type(bound[0]) == int:
                vector.append(random.randint(bound[0], bound[1]))
            elif type(bound[0]) == float:
                vector.append(random.uniform(bound[0], bound[1]))
            else:
                raise ValueError(f"Unsupported type: {bound[0]}.")
        return vector

    def _evaluate_population(self, population):
        fitnesses = self.toolbox.map(self.evaluator, population)
        for individual, fitness in zip(population, fitnesses):
            individual.fitness.values = fitness

    def _update_archive(self, archive, population, uncovered_objectives):
        for individual in population:
            for idx, (fitness, objective) in enumerate(zip(individual.fitness.values, self.objectives)):
                if fitness < objective:
                    continue
                archive_idx, archive_individual = next(
                    ((i, ind) for i, ind in enumerate(archive) if idx in ind.covered_objectives),
                    (None, None)
                )
                if archive_individual is not None and archive_idx is not None:  # individual already in the archive
                    if archive_individual.fitness.values[idx] < fitness:
                        individual.covered_objectives.append(idx)
                        archive[archive_idx] = individual
                        if idx in uncovered_objectives:
                            uncovered_objectives.remove(idx)
                else:
                    individual.covered_objectives.append(idx)
                    archive.append(individual)
                    if idx in uncovered_objectives:
                        uncovered_objectives.remove(idx)

    def _record_statistics(self, population, generation):
        record = self.stats.compile(population)
        self.logbook.record(gen=generation, evals=len(population), **record)
        print(self.logbook.stream)

    @staticmethod
    def _dominates(this, other, obj):
        """DO NOT USE THIS FUNCTION.
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
