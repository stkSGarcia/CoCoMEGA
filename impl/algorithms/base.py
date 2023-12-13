from abc import abstractmethod
from deap import base, creator, tools
import random


class BaseAlgorithm:
    def __init__(
        self,
        pop_size,  # initial population size
        evaluator,  # function for evaluating a population
        objectives,
        bounds,  # lower bound and upper bound of an individual
        cxpb,  # the probability of mating two individuals
        mutpb,  # the probability of mutating an individual
        time_budget,
        max_iter,
        seed,
    ) -> None:
        self.pop_size = pop_size
        self.evaluator = evaluator
        self.objectives = objectives
        self.bounds = bounds
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.time_budget = time_budget
        self.max_iter = max_iter

        self.random = random(seed)

        creator.create("FitnessMax", base.Fitness, weight=(1.0,) * len(objectives))
        creator.create("Individual", list, fitness=creator.FitnessMax, covered_objectives=list)

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
                vector.append(bool(self.random.getrandbits(1)))
            elif type(bound[0]) == int:
                vector.append(self.random.randint(bound[0], bound[1]))
            elif type(bound[0]) == float:
                vector.append(self.random.uniform(bound[0], bound[1]))
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
                    (None, None),
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
