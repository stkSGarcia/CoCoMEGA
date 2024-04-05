import logging
import os
import pickle
import time

from deap import base, tools

from impl.algorithm.base import BaseAlgorithm
from impl.config import CONFIG

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

    def solve(self, resume=False):
        logger.info("CCEA started.")
        logger.info(f"Time budget: {self.time_budget}.")
        logger.info(f"Max iteration: {self.max_iter}.")

        # Initialize the population.
        if resume:
            (gen, pop_scenario, pop_perturbation,
             archive_scenario, archive_perturbation,
             archive_solution, evaluated_solutions) = self._resume()
        else:
            gen = 0
            pop_scenario = self.toolbox.pop_scenario()
            pop_perturbation = self.toolbox.pop_perturbation()
            archive_scenario = pop_scenario
            archive_perturbation = pop_perturbation
            archive_solution = []
            evaluated_solutions = []

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
            pop_scenario = self._breed(pop_scenario, len(pop_scenario) - len(archive_scenario))
            pop_perturbation = self._breed(pop_perturbation, len(pop_perturbation) - len(archive_perturbation))

            pop_scenario += archive_scenario
            pop_perturbation += archive_perturbation
            gen += 1
            logger.info("Generation info:\n" + self.logbook.stream)
            self._checkpoint(gen, pop_scenario, pop_perturbation,
                             archive_scenario, archive_perturbation,
                             archive_solution, evaluated_solutions)
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
        candidates = self.toolbox.evaluate_solutions(candidates)
        evaluated_solutions.extend(candidates)
        archive_solution = [self.toolbox.clone(ind) for ind in evaluated_solutions if ind in unique_solutions]

        # Evaluate individual fitness.
        for scenario in pop_scenario:
            self.toolbox.evaluate_individual(scenario, archive_solution)
        self.toolbox.fitness_sharing(pop_scenario)

        for perturbation in pop_perturbation:
            self.toolbox.evaluate_individual(perturbation, archive_solution)
        self.toolbox.fitness_sharing(pop_perturbation)

        return archive_solution

    def _update_archive(self, population):
        """Update the archive using the `best random` strategy."""
        population = self.toolbox.clone(population)
        archive = tools.selBest(population, 1)
        population.remove(archive[0])
        if len(population) == 0: return archive

        for _ in range(self.archive_size - 1):
            best = population[0]
            if len(population) <= 1: break
            for ind in population[1:]:
                if self.population_diversity(archive + [ind]) > self.population_diversity(archive + [best]):
                    best = ind
            archive.append(best)
            population.remove(best)

        return archive

    def _breed(self, population, size):
        """Perform selection, crossover and mutation on individuals.

        @param population: The individuals to be bred.
        @param size: The size of the offsprings.
        @return: A list of offsprings.
        """
        assert len(population) > 0
        select_operator, mate_operator, mutate_operator = self.toolbox.operators(population[0])

        population = self.toolbox.clone(population)
        offsprings = []
        for _ in range(size):
            parents = select_operator(population, k=2)
            mate_operator(parents[0], parents[1])
            mutate_operator(parents[0])
            mutate_operator(parents[1])
            del parents[0].fitness.values
            del parents[1].fitness.values
            if (self.population_diversity(offsprings + [parents[0]]) >
                    self.population_diversity(offsprings + [parents[1]])):
                offsprings.append(parents[0])
            else:
                offsprings.append(parents[1])
        return offsprings

    @staticmethod
    def _checkpoint(*objects):
        dir_path = os.path.join(CONFIG["workspace"], CONFIG["checkpoint"])
        os.makedirs(dir_path, exist_ok=True)
        with open(os.path.join(dir_path, f"{int(round(time.time() * 1000))}.pickle"), "wb") as f:
            for obj in objects:
                pickle.dump(obj, f)

    @staticmethod
    def _resume():
        dir_path = os.path.join(CONFIG["workspace"], CONFIG["checkpoint"])
        files = sorted(os.listdir(dir_path), reverse=True)
        if len(files) == 0:
            raise ValueError("No checkpoints found.")
        logger.info(f"Resuming from checkpoint: {files[0]}.")
        with open(os.path.join(dir_path, files[0]), "rb") as f:
            return [pickle.load(f) for _ in range(7)]
