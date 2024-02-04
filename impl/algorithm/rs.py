import time

from impl.algorithm.base import BaseAlgorithm


class RandomSearch(BaseAlgorithm):
    def solve(self):
        # Initialize the uncovered objectives.
        uncovered_objectives = list(range(len(self.objectives)))

        archive = []

        gen = 0
        start_time = time.perf_counter()
        execution_time = 0
        while gen < self.max_iter and execution_time < self.time_budget:
            # Generate random population.
            population = self.toolbox.population(n=self.pop_size)

            # Evaluate the population.
            self.toolbox.evaluate(population)

            # Update archive.
            self.toolbox.archive(archive, population, uncovered_objectives)

            if len(uncovered_objectives) == 0: break

            self._record_statistics(archive, gen)
            gen += 1
            execution_time = time.perf_counter() - start_time

        return archive
