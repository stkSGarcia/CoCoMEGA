import logging
import os
import pickle
import random
import time
from multiprocessing import Manager

from deap import tools

from impl.algorithm.base import BaseAlgorithm
from impl.config import CONFIG
from impl.scenario import simulation_runner
from impl.utils.trajectory import trajectory_score

logger = logging.getLogger(__name__)


class CCEA(BaseAlgorithm):
    def solve(self, resume=False):
        logger.info("CCEA started.")
        logger.info(self.budget.print_budget())

        # Initialize the population or resume from the latest checkpoint.
        if resume:
            files = sorted(os.listdir(CONFIG["workspace"]["checkpoint"]), reverse=True)
            if len(files) == 0:
                raise ValueError("No checkpoints found.")
            logger.info(f"Resuming from checkpoint: {files[0]}.")
            with open(os.path.join(CONFIG["workspace"]["checkpoint"], files[0]), "rb") as f:
                pop_scenario = pickle.load(f)
                pop_perturbation = pickle.load(f)
                archive_scenario = pickle.load(f)
                archive_perturbation = pickle.load(f)
                archive_solution = pickle.load(f)
                evaluated_solutions = pickle.load(f)
                simulation_runner.evaluated_scenarios = Manager().list(pickle.load(f))
                self.budget.initialize(other=pickle.load(f))
                self.logbook = pickle.load(f)
        else:
            pop_scenario = self.toolbox.pop_scenario()
            pop_perturbation = self.toolbox.pop_perturbation()
            archive_scenario = pop_scenario
            archive_perturbation = pop_perturbation
            archive_solution = []
            evaluated_solutions = []
            self.budget.initialize()

        while not self.budget.is_reached():
            # Evaluate the population.
            current_solutions, sim_num = self._evaluate(pop_scenario, archive_scenario,
                                                        pop_perturbation, archive_perturbation,
                                                        evaluated_solutions)
            violated_solutions = [solution for solution in current_solutions if solution.is_violated]
            violated_solutions_count = len(violated_solutions)
            logger.info(f"The number of solutions violating the relation: {violated_solutions_count}.")
            archive_solution += [solution for solution in current_solutions if solution not in archive_solution]

            # Terminate if the archive has converged.
            if (self.budget.gen_num > 0 and
                    violated_solutions_count > self.budget.convergence_threshold * len(current_solutions)):
                logger.info(f"Terminate due to the number of violations reaching the threshold: "
                            f"{violated_solutions_count} > {self.budget.convergence_threshold}*{len(current_solutions)}.")
                break

            self.budget.acc_sim(sim_num)
            self.record_statistics(pop_scenario, self.budget.gen_num, pop_name="pop_scen")
            self.record_statistics(pop_perturbation, self.budget.gen_num, pop_name="pop_pert")
            self.record_statistics(violated_solutions, self.budget.gen_num, pop_name="solution", sim_num=sim_num)
            self.record_statistics(archive_solution, self.budget.gen_num, pop_name="archive",
                                   sim_num=self.budget.sim_num)

            # Update archive.
            archive_scenario = self._update_archive(pop_scenario, CONFIG["scenario"]["archive_size"])
            archive_perturbation = self._update_archive(pop_perturbation, CONFIG["perturbation"]["archive_size"])
            self.record_statistics(archive_scenario, self.budget.gen_num, pop_name="arc_scen")
            self.record_statistics(archive_perturbation, self.budget.gen_num, pop_name="arc_pert")

            # Generate offsprings.
            pop_scenario = self._breed(pop_scenario, len(pop_scenario) - len(archive_scenario))

            perturbation_offspring_size = len(pop_perturbation) - len(archive_perturbation)
            pop_perturbation = self._breed(pop_perturbation, perturbation_offspring_size,
                                           overproduction_factor=CONFIG["perturbation"]["overproduction"])
            pop_perturbation = self._shrink(pop_perturbation, perturbation_offspring_size, pop_scenario)

            pop_scenario += archive_scenario
            pop_perturbation += archive_perturbation
            self.budget.acc_gen()
            logger.info("Generation info:\n" + self.logbook.stream)
            logger.info(f"Number of simulations: {sim_num}/{self.budget.sim_num}.")

            # Store the current status into a checkpoint.
            with open(os.path.join(CONFIG["workspace"]["checkpoint"], f"{int(round(time.time() * 1000))}.pickle"),
                      "wb") as f:
                pickle.dump(pop_scenario, f)
                pickle.dump(pop_perturbation, f)
                pickle.dump(archive_scenario, f)
                pickle.dump(archive_perturbation, f)
                pickle.dump(archive_solution, f)
                pickle.dump(evaluated_solutions, f)
                pickle.dump(list(simulation_runner.evaluated_scenarios), f)
                pickle.dump(self.budget, f)
                pickle.dump(self.logbook, f)

        logger.info(f"Terminate due to reaching the threshold.")
        # Store the complete solutions.
        suffix = int(round(time.time() * 1000))
        statistics_path = f"statistics-{suffix}.pickle"
        with open(os.path.join(CONFIG["workspace"]["solution"], f"solutions-{suffix}.pickle"), "wb") as f:
            pickle.dump(archive_solution, f)
        with open(os.path.join(CONFIG["workspace"]["solution"], statistics_path), "wb") as f:
            pickle.dump(self.logbook, f)

        return archive_solution, statistics_path

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
        logger.debug(f"#complete solutions: {len(complete_solutions)}, "
                     f"#unique solutions: {len(unique_solutions)}, "
                     f"#candidates: {len(candidates)}.")
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

        for perturbation in pop_perturbation:
            self.toolbox.evaluate_individual(perturbation, archive_solution)

        if CONFIG["opt"]["niching"]["strategy"] == "sharing":
            self.fitness_sharing(pop_scenario,
                                 CONFIG["opt"]["niching"]["punishment"],
                                 CONFIG["opt"]["niching"]["scaling"])
            self.fitness_sharing(pop_perturbation,
                                 CONFIG["opt"]["niching"]["punishment"],
                                 CONFIG["opt"]["niching"]["scaling"])
        elif CONFIG["opt"]["niching"]["strategy"] == "clearing":
            self.fitness_clearing(pop_scenario, CONFIG["opt"]["niching"]["capacity"])
            self.fitness_clearing(pop_perturbation, CONFIG["opt"]["niching"]["capacity"])
        elif CONFIG["opt"]["niching"]["strategy"] != "none":
            logger.warning("Unrecognized niching strategy, falling back to `none`.")

        return archive_solution, sim_num

    def _update_archive(self, population, archive_size):
        """Update the archive.

        @param population: The individuals to be archived.
        @param archive_size: The size of the archive.
        @return: An archive of the individuals.
        """
        population = list(map(self.toolbox.clone, population))
        archive = tools.selBest(population, 1)
        population.remove(archive[0])

        if CONFIG["opt"]["diversity"]:
            # Select individuals able to maximize the diversity.
            valid, invalid = [], []
            for ind in population:
                (invalid, valid)[ind.fitness.valid].append(ind)
            size = archive_size - 1
            for pop in [valid, invalid]:
                if len(pop) == 0: continue
                while size > 0:
                    if len(pop) == 1:
                        archive.append(pop[0])
                        size -= 1
                        break
                    best = pop[0]
                    for ind in pop[1:]:
                        if self.population_diversity(archive + [ind]) > self.population_diversity(archive + [best]):
                            best = ind
                    archive.append(best)
                    pop.remove(best)
                    size -= 1
        else:
            # Randomly select individuals.
            archive += tools.selRandom(population, archive_size - 1)
        return archive

    def _breed(self, population, size, overproduction_factor=1):
        """Perform selection, crossover and mutation on individuals.

        @param population: The individuals to be bred.
        @param size: The size of the offsprings.
        @param overproduction_factor: Overproduction factor
        @return: A list of offsprings.
        """
        assert len(population) > 0
        offsprings = []
        for _ in range(size * overproduction_factor):
            parents = population[0].select(population)
            parents = list(map(self.toolbox.clone, parents))
            parents[0].mate(parents[1])
            if CONFIG["opt"]["diversity"]:
                parents[0].mutate()
                parents[1].mutate()
                del parents[0].fitness.values
                del parents[1].fitness.values
                offspring = parents[0] if (self.population_diversity(offsprings + [parents[0]]) >
                                           self.population_diversity(offsprings + [parents[1]])) else parents[1]
            else:
                offspring = random.choice(parents)
                offspring.mutate()
                del offspring.fitness.values
            offspring.correct()
            offsprings.append(offspring)

        return offsprings

    def _shrink(self, population, size, co_population):
        """Select from over-producted population based on a heuristic function.
                @param population: The over-producted population.
                @param size: The size of the target population.
                @param co_population: The coop population
                @return: A list of selected individuals.
                """
        return sorted(population, key=lambda p: trajectory_score(p, co_population), reverse=True)[:size]
