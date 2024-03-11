import logging
import random
import time
from concurrent.futures import ProcessPoolExecutor

from deap import base, tools

from impl import config
from impl.algorithm.base import BaseAlgorithm

logger = logging.getLogger(__name__)


class CCEA(BaseAlgorithm):
    def __init__(self,
                 archive_size,
                 toolbox: base.Toolbox,
                 time_budget,
                 max_iter,
                 seed=None):
        """Constructor.

        @param archive_size: The maximum number of individuals allowed in the archive.
        """
        super().__init__(toolbox, time_budget, max_iter, seed)
        self.archive_size = archive_size

    def solve(self):
        logger.info("CCEA started.")
        logger.info(f"Time budget: {self.time_budget}.")
        logger.info(f"Max iteration: {self.max_iter}.")

        # Initialize the population.
        pop_scenario = self.toolbox.pop_scenario()
        pop_perturbation = self.toolbox.pop_perturbation()
        archive_scenario = pop_scenario
        archive_perturbation = pop_perturbation
        archive_solution = []
        evaluated_solutions = []

        gen = 0
        start_time = time.perf_counter()
        while gen < self.max_iter and time.perf_counter() - start_time < self.time_budget:
            archive_solution = self._evaluate(pop_scenario, archive_scenario,
                                              pop_perturbation, archive_perturbation,
                                              evaluated_solutions)
            self._record_statistics(pop_scenario, gen, pop_name="pop_scen")
            self._record_statistics(pop_perturbation, gen, pop_name="pop_pert")
            self._record_statistics(archive_solution, gen, pop_name="solution")

            # Update archive.
            archive_scenario = self._update_archive(pop_scenario)
            archive_perturbation = self._update_archive(pop_perturbation)
            self._record_statistics(archive_scenario, gen, pop_name="arc_scen")
            self._record_statistics(archive_perturbation, gen, pop_name="arc_pert")

            # Generate offsprings.
            pop_scenario = self._breed(pop_scenario, len(pop_scenario) - len(archive_scenario),
                                       self.toolbox.mutate_scenario)
            pop_perturbation = self._breed(pop_perturbation, len(pop_perturbation) - len(archive_perturbation),
                                           self.toolbox.mutate_perturbation)

            pop_scenario += archive_scenario
            pop_perturbation += archive_perturbation
            gen += 1
            logger.info(self.logbook.stream)

        return archive_solution

    def _evaluate(self, pop_scenario, archive_scenario, pop_perturbation, archive_perturbation, evaluated_solutions):
        """Form complete solutions and evaluate their joint fitness as well as the individual fitness values.

        @param pop_scenario: The population of scenarios.
        @param archive_scenario: The archive of scenarios.
        @param pop_perturbation: The population of perturbations.
        @param archive_perturbation: The archive of perturbations.
        @param evaluated_solutions: The evaluated complete solutions.
        @return: An archived complete solutions.
        """
        # Generate complete solutions from archives.
        complete_solutions = ([self.toolbox.collaborate(self.toolbox.clone(scenario), self.toolbox.clone(perturbation))
                               for scenario in pop_scenario for perturbation in archive_perturbation] +
                              [self.toolbox.collaborate(self.toolbox.clone(scenario), self.toolbox.clone(perturbation))
                               for scenario in archive_scenario for perturbation in pop_perturbation])

        # If the size of archives doesn't reach the min_num_evals, then generate complete solutions from populations.
        # if self.min_num_evals > len(archive_scenario):
        #     complete_solutions += [
        #         self.toolbox.collaborate(self.toolbox.clone(scenario), self.toolbox.clone(perturbation))
        #         for scenario in [ind for ind in pop_scenario if ind not in archive_scenario]
        #         for perturbation in pop_perturbation]
        # if self.min_num_evals > len(archive_perturbation):
        #     complete_solutions += [
        #         self.toolbox.collaborate(self.toolbox.clone(scenario), self.toolbox.clone(perturbation))
        #         for scenario in pop_scenario for perturbation in
        #         [ind for ind in pop_perturbation if ind not in archive_perturbation]]

        # Remove repetitive complete solutions.
        unique_solutions = []
        for ind in complete_solutions:
            if ind not in unique_solutions:
                unique_solutions.append(ind)

        # Evaluate joint fitness.
        # TODO: avoid evaluating similar scenarios
        candidates = [ind for ind in unique_solutions if ind not in evaluated_solutions]
        with ProcessPoolExecutor(max_workers=config.CONFIG["max_workers"]) as executor:
            candidates = executor.map(self.toolbox.evaluate_joint, candidates)
        evaluated_solutions.extend(candidates)
        archive_solution = [self.toolbox.clone(ind) for ind in evaluated_solutions if ind in unique_solutions]

        # Evaluate individual fitness.
        for scenario in pop_scenario:
            self.toolbox.evaluate_individual(scenario, 0, archive_solution)
        self.toolbox.fitness_sharing(pop_scenario, 0)

        for perturbation in pop_perturbation:
            self.toolbox.evaluate_individual(perturbation, 1, archive_solution)
        self.toolbox.fitness_sharing(pop_perturbation, 1)

        return archive_solution

    def _update_archive(self, population):
        """Update the archive using the `best random` strategy."""
        population = self.toolbox.clone(population)
        best = tools.selBest(population, 1)
        # TODO: calculate diversity
        random_individuals = tools.selRandom(population, self.archive_size - 1)  # TODO: avoid similar individuals
        return best + random_individuals

    def _breed(self, population, size, mutate_operator):
        """Perform selection, crossover and mutation on individuals.

        @param population: The individuals to be bred.
        @param size: The size of the offsprings.
        @param mutate_operator: The function to perform the mutation.
        @return: A list of offsprings.
        """
        population = self.toolbox.clone(population)
        offsprings = []
        for _ in range(size):
            parents = self.toolbox.select(population, k=2)
            parents = self.toolbox.mate(parents[0], parents[1])
            offspring = random.choice(parents)
            offspring = mutate_operator(offspring)
            # TODO
            del offspring.fitness.values
            offsprings.append(offspring)
        return offsprings
