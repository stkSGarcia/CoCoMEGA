import logging
import os
import pickle
import random
import time
from multiprocessing import Manager

from deap import base, tools

from impl.algorithm.base import BaseAlgorithm
from impl.config import CONFIG
from impl.scenario import simulation_runner

logger = logging.getLogger(__name__)


class CCEA(BaseAlgorithm):
    def __init__(self,
                 archive_size,
                 toolbox: base.Toolbox,
                 max_sim=None,
                 max_time=None,
                 max_iter=None,
                 seed=None):
        """Constructor.

        @param archive_size: The maximum number of individuals allowed in the archive.
        """
        super().__init__(toolbox, max_sim, max_time, max_iter, seed)
        self.archive_size = archive_size

    def solve(self, resume=False):
        logger.info("CCEA started.")
        logger.info(f"Max number of simulations: {self.max_sim}.")
        logger.info(f"Max time: {self.max_time}.")
        logger.info(f"Max iteration: {self.max_iter}.")

        # Initialize the population or resume from the latest checkpoint.
        checkpoint_dir = os.path.join(CONFIG["workspace"], CONFIG["checkpoint"])
        if resume:
            files = sorted(os.listdir(checkpoint_dir), reverse=True)
            if len(files) == 0:
                raise ValueError("No checkpoints found.")
            logger.info(f"Resuming from checkpoint: {files[0]}.")
            with open(os.path.join(checkpoint_dir, files[0]), "rb") as f:
                gen = pickle.load(f)
                sim_num = pickle.load(f)
                pop_scenario = pickle.load(f)
                pop_perturbation = pickle.load(f)
                archive_scenario = pickle.load(f)
                archive_perturbation = pickle.load(f)
                archive_solution = pickle.load(f)
                evaluated_solutions = pickle.load(f)
                simulation_runner.evaluated_scenarios = Manager().list(pickle.load(f))
        else:
            gen = 0
            sim_num = 0
            pop_scenario = self.toolbox.pop_scenario()
            if CONFIG["scenario"]["init_selection_factor"] > 1:
                pop_scenario = sorted(pop_scenario, key=lambda x: x.trajectory_collision_score(),
                                      reverse=True)[:CONFIG["scenario"]["pop_size"]]
            pop_perturbation = self.toolbox.pop_perturbation()
            archive_scenario = pop_scenario
            archive_perturbation = pop_perturbation
            archive_solution = []
            evaluated_solutions = []

        start_time = time.perf_counter()
        while ((self.max_sim is None or sim_num < self.max_sim) and
               (self.max_time is None or time.perf_counter() - start_time < self.max_time) and
               (self.max_iter is None or gen < self.max_iter)):
            # Evaluate the population.
            archive_solution, n = self._evaluate(pop_scenario, archive_scenario,
                                                 pop_perturbation, archive_perturbation,
                                                 evaluated_solutions)
            violated_solutions_count = [solution.is_violated for solution in archive_solution].count(True)
            logger.info(f"The number of solutions violating the relation: {violated_solutions_count}.")

            # Terminate if the archive has converged.
            if gen > 0 and violated_solutions_count > CONFIG["convergence_threshold"] * len(archive_solution):
                logger.info(f"Terminate due to the number of violations reaching the threshold: "
                            f"{violated_solutions_count} > {CONFIG['convergence_threshold']}*{len(archive_solution)}.")
                # Store the complete solutions.
                result_dir = os.path.join(CONFIG["workspace"], CONFIG["solution"])
                os.makedirs(result_dir, exist_ok=True)
                with open(os.path.join(result_dir, f"solutions-{int(round(time.time() * 1000))}.pickle"), "wb") as f:
                    pickle.dump(archive_solution, f)
                break

            sim_num += n
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
            logger.info(f"Number of simulations: {n}/{sim_num}.")

            # Store the current status into a checkpoint.
            os.makedirs(checkpoint_dir, exist_ok=True)
            with open(os.path.join(checkpoint_dir, f"{int(round(time.time() * 1000))}.pickle"), "wb") as f:
                pickle.dump(gen, f)
                pickle.dump(sim_num, f)
                pickle.dump(pop_scenario, f)
                pickle.dump(pop_perturbation, f)
                pickle.dump(archive_scenario, f)
                pickle.dump(archive_perturbation, f)
                pickle.dump(archive_solution, f)
                pickle.dump(evaluated_solutions, f)
                pickle.dump(list(simulation_runner.evaluated_scenarios), f)

        return archive_solution

    def _evaluate(self, pop_scenario, archive_scenario, pop_perturbation, archive_perturbation, evaluated_solutions):
        """Form complete solutions and evaluate their joint fitness as well as the individual fitness values.

        @param pop_scenario: The population of scenarios.
        @param archive_scenario: The archive of scenarios.
        @param pop_perturbation: The population of perturbations.
        @param archive_perturbation: The archive of perturbations.
        @param evaluated_solutions: The evaluated complete solutions.
        @return: An archived complete solutions and the number of simulations.
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
        # TODO: avoid evaluating similar scenarios.
        candidates = [ind for ind in unique_solutions if ind not in evaluated_solutions]
        if len(candidates) > 0:
            candidates, sim_num = self.toolbox.evaluate_solutions(candidates)
            evaluated_solutions.extend(candidates)
        else:
            logger.warning("Candidate solution list is empty!")
            sim_num = 0
        archive_solution = [self.toolbox.clone(ind) for ind in evaluated_solutions
                            if ind.fitness.valid and ind in unique_solutions]

        # Evaluate individual fitness.
        for scenario in pop_scenario:
            self.toolbox.evaluate_individual(scenario, archive_solution)
        self.toolbox.fitness_sharing(pop_scenario)

        for perturbation in pop_perturbation:
            self.toolbox.evaluate_individual(perturbation, archive_solution)
        self.toolbox.fitness_sharing(pop_perturbation)

        return archive_solution, sim_num

    def _update_archive(self, population):
        """Update the archive."""
        population = self.toolbox.clone(population)
        archive = tools.selBest(population, 1)
        population.remove(archive[0])
        if len(population) == 0: return archive

        if CONFIG["diversity_opt"]:
            # Select individuals able to maximize the diversity.
            for _ in range(self.archive_size - 1):
                best = population[0]
                if len(population) <= 1: break
                for ind in population[1:]:
                    if self.population_diversity(archive + [ind]) > self.population_diversity(archive + [best]):
                        best = ind
                archive.append(best)
                population.remove(best)
        else:
            # Randomly select individuals.
            archive += tools.selRandom(population, self.archive_size - 1)

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
            if CONFIG["diversity_opt"]:
                mutate_operator(parents[0])
                mutate_operator(parents[1])
                del parents[0].fitness.values
                del parents[1].fitness.values
                if (self.population_diversity(offsprings + [parents[0]]) >
                        self.population_diversity(offsprings + [parents[1]])):
                    offsprings.append(parents[0])
                else:
                    offsprings.append(parents[1])
            else:
                offspring = random.choice(parents)
                mutate_operator(offspring)
                del offspring.fitness.values
                offsprings.append(offspring)
        return offsprings
