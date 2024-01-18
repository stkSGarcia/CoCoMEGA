import random
from abc import abstractmethod
from typing import Callable, List

import numpy as np
from deap import base, creator, tools

from impl.mr import MRSet
from impl.scenario import Scenario


class BaseAlgorithm:
    def __init__(self, evaluator: Callable, mrs: MRSet, scenario_size=10, pop_size=10, cxpb=0.8, mutpb=0.6,
                 time_budget=3600, max_iter=100, seed=None):
        """Constructor.

        @param evaluator: function for evaluating a population
        @param mrs: metamorphic relations
        @param scenario_size: size of source scenarios
        @param pop_size: initial populaiton size
        @param cxpb: the probability of mating two individuals
        @param mutpb: the probability of mutating an individual
        @param time_budget: maximum execution time
        @param max_iter: maximum number of iterations
        @param seed: random seed
        """
        self.evaluator = evaluator
        self.mrs = mrs
        self.scenario_size = scenario_size
        self.pop_size = pop_size
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.time_budget = time_budget
        self.max_iter = max_iter
        random.seed(seed)

        # Randomly generate source scenarios.
        self.source_scenarios = Scenario.generate_random_source_scenarios(self.scenario_size)
        simulation_results = list(map(self.evaluator, self.source_scenarios))

        # Transform MRs into fitness functions.
        self.objectives, self.weights = self.mrs.to_objectives(simulation_results)

        # Define individual
        creator.create("Fitness", base.Fitness, weights=self.weights)
        creator.create("Individual", list, fitness=creator.Fitness, covered_objectives=list)

        # Replace the original `dominates` function.
        if getattr(creator.Fitness, "dominates", None) is not None:
            setattr(creator.Fitness, "dominates", BaseAlgorithm._dominates)

        # Register functions
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", creator.Individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
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
        """Run the algorithm."""
        raise NotImplementedError

    def _evaluate_population(self, population: List):
        """Evaluate each individual in the population."""
        simulation_results = list(map(self.evaluator, population))
        for individual, fitness in zip(population, fitness_list):
            individual.fitness.values = fitness

    def _update_archive(self, archive: List, population: List, uncovered_objectives: List):
        """Add individual meeting the objective to the archive."""
        for individual in population:
            for idx, (fitness, objective) in enumerate(zip(individual.fitness.values, self.objectives)):
                if fitness < objective: continue
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

    def _record_statistics(self, population: List, num_of_generation: int):
        """Record the statistics of the population."""
        record = self.stats.compile(population)
        self.logbook.record(gen=num_of_generation, evals=len(population), **record)
        print(self.logbook.stream)

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
