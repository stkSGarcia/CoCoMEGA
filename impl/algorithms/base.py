import random
from abc import abstractmethod

import numpy as np
from deap import base, creator, tools


class BaseAlgorithm:
    def __init__(
            self,
            pop_size,  # initial population size
            evaluator,  # function for evaluating a population
            objectives,  # objective thresholds
            bounds,  # lower bound and upper bound of an individual
            cxpb,  # the probability of mating two individuals
            mutpb,  # the probability of mutating an individual
            time_budget,  # maximum execution time
            max_iter,  # maximum number of iterations
            seed  # random seed
    ) -> None:
        self.pop_size = pop_size
        self.evaluator = evaluator
        self.objectives = objectives
        self.bounds = bounds
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.time_budget = time_budget
        self.max_iter = max_iter

        random.seed(seed)

        creator.create("Fitness", base.Fitness, weight=(1.0,) * len(objectives))  # maximize the fitness
        creator.create("Individual", list, fitness=creator.Fitness, covered_objectives=list)

        # Replace the original `dominates` function.
        if getattr(creator.Fitness, "dominates", None) is not None:
            setattr(creator.Fitness, "dominates", self._dominates)

        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self._initialize_vector)
        self.toolbox.register("evaluate", self._evaluate_population)
        self.toolbox.register("archive", self._update_archive)

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
        self.toolbox.map(self.evaluator, population)

    def _update_archive(self, archive, population, uncovered_objectives):
        for individual in population:
            for idx, (fitness, objective) in enumerate(zip(individual.fitness.values, self.objectives)):
                if fitness > objective:
                    continue
                archive_individual, archive_idx = next(
                    ((i, ind) for i, ind in enumerate(archive) if idx in ind.covered_objectives),
                    (None, None)
                )
                if archive_individual is not None and archive_idx is not None:  # individual already in the archive
                    if archive_individual.fitness.values[idx] > objective:
                        individual.covered_objectives.append(idx)
                        archive[archive_idx] = individual
                        if idx in uncovered_objectives:
                            uncovered_objectives.remove(idx)
                else:
                    individual.covered_objectives.append(idx)
                    archive.append(individual)
                    if idx in uncovered_objectives:
                        uncovered_objectives.remove(idx)

    def _dominates(self, other, obj):
        """DO NOT USE THIS FUNCTION.
        It is used to replace the original `dominates` function in `deap`.

        :param obj: Indices indicating on which objectives the domination is tested.
        """
        not_equal = False
        for self_wvalue, other_wvalue in zip(np.array(self.wvalues)[obj], np.array(other.wvalues)[obj]):
            if self_wvalue > other_wvalue:
                not_equal = True
            elif self_wvalue < other_wvalue:
                return False
        return not_equal
