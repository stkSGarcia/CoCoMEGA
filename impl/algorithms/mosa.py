import random
import time
from collections import defaultdict
from operator import attrgetter

import numpy as np
from deap import tools

from base import BaseAlgorithm


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
            seed
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
            seed
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
            F = self._preference_sorting(population + offspring, uncovered_objectives)

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
            tools.emo.assignCrowdingDist(F[index])
            sorted_front = sorted(F[index], key=attrgetter("fitness.crowding_dist"), reverse=True)
            remain_len = self.pop_size - len(next_population)
            next_population.extend(sorted_front[:remain_len])

            # Compile statistics about the new population
            record = stats.compile(next_population)
            logbook.record(gen=gen, evals=len(next_population), **record)
            print(logbook.stream)

            gen += 1
            execution_time = time.perf_counter() - start_time

    def _generate_offspring(self, population, uncovered_objectives):
        offspring = [self.toolbox.clone(ind) for ind in population]
        while len(offspring) < len(population):
            parent1 = self._tournament_selection(offspring, 10, uncovered_objectives)
            parent2 = self._tournament_selection(offspring, 10, uncovered_objectives)
            if random.uniform(0, 1) <= self.cxpb:
                tools.cxOnePoint(parent1, parent2)
            # TODO mutation and so on
        return offspring

    @staticmethod
    def _tournament_selection(population, size, uncovered_objectives):
        candidates = []
        for i in range(size):
            idx = random.randint(0, len(population) - 1)
            candidates.append(population[idx])

        best = candidates[0]
        for i in range(size):
            candidate1 = candidates[i]
            for j in range(size):
                candidate2 = candidates[j]
                if candidate1.fitness.dominates(candidate2.fitness, uncovered_objectives):
                    best = candidate1
        return best

    def _preference_sorting(self, population, uncovered_objectives):
        population = [self.toolbox.clone(ind) for ind in population]
        F = []
        for idx in uncovered_objectives:
            max_fitness = -1  # TODO
            best = population[0]
            for individual in population:
                if individual.fitness.values[idx] > max_fitness:
                    max_fitness = individual.fitness.values[idx]
                    best = individual
            F.append(best)
            population.remove(best)
        if len(F) > self.pop_size:
            return F
        if len(population) > 0:
            E = self._fast_nondominated_sort(population, uncovered_objectives)
            F += E
        return F

    @staticmethod
    def _fast_nondominated_sort(population, uncovered_objectives):
        map_fit_ind = defaultdict(list)
        for ind in population:
            map_fit_ind[ind.fitness].append(ind)
        fits = list(map_fit_ind.keys())

        fronts = [[]]
        current_front = []
        next_front = []
        dominating_fits = defaultdict(int)
        dominated_fits = defaultdict(list)

        for i, fit_i in enumerate(fits):
            for fit_j in fits[i + 1:]:
                if fit_i.dominates(fit_j, uncovered_objectives):
                    dominating_fits[fit_j] += 1
                    dominated_fits[fit_i].append(fit_j)
                elif fit_j.dominates(fit_i, uncovered_objectives):
                    dominating_fits[fit_i] += 1
                    dominated_fits[fit_j].append(fit_i)
            if dominating_fits[fit_i] == 0:
                current_front.append(fit_i)
                fronts[-1].extend(map_fit_ind[fit_i])

        while current_front:
            fronts.append([])
            for fit_p in current_front:
                for fit_d in dominated_fits[fit_p]:
                    dominating_fits[fit_d] -= 1
                    if dominating_fits[fit_d] == 0:
                        next_front.append(fit_d)
                        fronts[-1].extend(map_fit_ind[fit_d])
            current_front = next_front
            next_front = []

        return fronts
