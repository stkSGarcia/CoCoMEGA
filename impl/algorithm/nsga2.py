import logging
import time

from deap import tools

from impl.algorithm.base import BaseAlgorithm

logger = logging.getLogger(__name__)


class NSGA2(BaseAlgorithm):
    def solve(self):
        logger.info("NSGA2 started.")
        logger.info(f"Time budget: {self.time_budget}.")
        logger.info(f"Max iteration: {self.max_iter}.")

        # Initialize the first generation.
        population = self.toolbox.population()

        gen = 0
        start_time = time.perf_counter()
        while gen < self.max_iter and time.perf_counter() - start_time < self.time_budget:
            # Evaluate the individuals.
            invalid_individuals = [ind for ind in population if not ind.fitness.valid]
            self.toolbox.evaluate(invalid_individuals)

            population = self.toolbox.select(population)
            logger.info(self.logbook.stream)

            self._record_statistics(population, gen)

            # Vary the population.
            offspring = tools.selTournamentDCD(population, len(population))
            offspring = self.toolbox.clone(offspring)

            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                self.toolbox.mate(ind1, ind2)
                self.toolbox.mutate(ind1)
                self.toolbox.mutate(ind2)
                del ind1.fitness.values, ind2.fitness.values

            population += offspring
            gen += 1

        return population
