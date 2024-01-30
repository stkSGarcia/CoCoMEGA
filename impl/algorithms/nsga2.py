import random
import time

from deap import tools

from impl.algorithms.base import BaseAlgorithm


class NSGA2(BaseAlgorithm):
    def solve(self):
        # Initialize the first generation
        population = self.toolbox.population(n=self.pop_size)

        # Evaluate the individuals
        fitnesses = self.toolbox.evaluate(population)
        for individual, fitness in zip(population, fitnesses):
            individual.fitness.values = fitness

        # Assign the crowding distance to the individuals
        population = self.toolbox.select(population, len(population))

        gen = 0
        self._record_statistics(population, gen)

        start_time = time.perf_counter()
        while gen < self.max_iter and time.perf_counter() - start_time < self.time_budget:
            # Vary the population
            offspring = tools.selTournamentDCD(population, len(population))
            offspring = [self.toolbox.clone(ind) for ind in offspring]

            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.cxpb:
                    self.toolbox.mate(ind1, ind2)
                self.toolbox.mutate(ind1)
                self.toolbox.mutate(ind2)
                del ind1.fitness.values, ind2.fitness.values

            # Evaluate the individuals
            invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.evaluate(invalid_individuals)
            for individual, fitness in zip(invalid_individuals, fitnesses):
                individual.fitness.values = fitness

            population = self.toolbox.select(population + offspring, self.pop_size)

            gen += 1
            self._record_statistics(population, gen)

        return population
