import time

from deap import base

from impl.algorithm.base import BaseAlgorithm


class CCEA(BaseAlgorithm):
    def __init__(self, scenario_pop_size, perturbation_pop_size, toolbox: base.Toolbox, pop_size=10, cxpb=0.8,
                 mutpb=0.6, time_budget=3600, max_iter=100, seed=None):
        super().__init__(toolbox, pop_size, cxpb, mutpb, time_budget, max_iter, seed)
        self.scenario_pop_size = scenario_pop_size
        self.perturbation_pop_size = perturbation_pop_size

    def solve(self):
        # Initialize the population
        pop_scenario = self.toolbox.pop_scenario(n=self.scenario_pop_size)
        pop_perturbation = self.toolbox.pop_perturbation(n=self.perturbation_pop_size)

        # Evaluate

        gen = 0
        self._record_statistics(pop_scenario, gen)
        self._record_statistics(pop_perturbation, gen)

        start_time = time.perf_counter()
        while gen < self.max_iter and time.perf_counter() - start_time < self.time_budget:
            gen += 1
