from .base import BaseAlgorithm
from deap import base, creator, tools
import numpy as np
import time


class MOSA(BaseAlgorithm):
    def __init__(
        self,
        pop_size,
        evaluator,
        objectives,
        bounds,
        cxpb,
        mutpb,
        time_budget,
        max_iter,
        seed,
    ) -> None:
        super().__init__(
            pop_size,
            evaluator,
            objectives,
            bounds,
            cxpb,
            mutpb,
            time_budget,
            max_iter,
            seed,
        )
        # Define the problem
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("breed", self._generate_offspring)

    def solve(self):
        # Initialize statistics object
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "std", "min", "avg", "max"

        # Initialize the uncovered objectives
        uncovered_objectives = list(range(len(self.objectives)))

        # Initialize the first generation and an archive
        population = self.toolbox.population(n=self.pop_size)
        archive = []

        # Evaluate the first generation
        self.toolbox.evaluate(population)

        # Update archive
        self.toolbox.archive(archive, population, uncovered_objectives)

        gen = 0
        start_time = time.perf_counter()
        execution_time = 0
        while gen < self.max_iter and execution_time < self.time_budget:
            # Generate offsprings
            offspring = self.toolbox.breed(population, uncovered_objectives)

            # Evaluate the offsprings
            self.toolbox.evaluate(offspring)

            # Update archive
            self.toolbox.archive(archive, offspring, uncovered_objectives)

            # Preference sort
            F = self._preference_sorting()

            if len(uncovered_objectives) == 0:
                break

            next_population = []
            index = 0
            while len(next_population) <= self.pop_size:
                if len(next_population) + len(F[index]) > self.pop_size:
                    break
                next_population.extend(F[index])
                index += 1

            # Crowding distance

            # Compile statistics about the new population
            record = stats.compile(next_population)
            logbook.record(gen=gen, evals=len(), **record)
            print(logbook.stream)

            gen += 1
            execution_time = time.perf_counter() - start_time

    def _generate_offspring(self, population, uncovered_objectives):
        offspring = [self.toolbox.clone(ind) for ind in population]
        while len(offspring) < len(population):
            parent1 = tools.selTournamentDCD(offspring, k=10, tournsize=2)
            # if self.random.uniform(0, 1) <= self.cxpb:

        return offspring

    def _preference_sorting(self, population, uncovered_objectives):
        population = [self.toolbox.clone(ind) for ind in population]
        F = []
        for idx in uncovered_objectives:
            max = -1
            best = population[0]
            for individual in population:
                if individual.fitness.values[idx] > max:
                    max = individual.fitness.values[idx]
                    best = individual
            F.append(best)
            population.remove(best)
        if len(F) > self.pop_size:
            return F
        if len(population) > 0:
            E = self._fast_nondominated_sort(population, uncovered_objectives)
            F += E
        return F

    def _fast_nondominated_sort(self, population, uncovered_objectives):
        F = []
        front = []
        count = 0
        while len(population) > 1:
            count = 0

    def _dominates(self, individual1, individual2, obj):
        """Dominance comparator for two individuals."""
        not_equal = False
        for ind1_wvalue, ind2_wvalue in zip(
            np.array(individual1.fitness.wvalues)[obj], np.array(individual2.fitness.wvalues)[obj]
        ):  # use weighted values in this comparison
            if ind1_wvalue > ind2_wvalue:
                not_equal = True
            elif ind1_wvalue < ind2_wvalue:
                return False
        return not_equal
