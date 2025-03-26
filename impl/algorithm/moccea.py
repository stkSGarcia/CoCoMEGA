import logging
import os
import pickle
import random
import time
from multiprocessing import Manager

from impl import config
from impl.algorithm.base import BaseAlgorithm
from impl.scenario import simulation_runner

logger = logging.getLogger(__name__)


class MOCCEA(BaseAlgorithm):
    """Multi-Objective Cooperative Co-Evolutionary Algorithm."""
    _name = "MOCCEA"

    def solve(self, resume=False):
        super().solve()
        # Initialize the population or resume from the latest checkpoint.
        if resume:
            files = sorted(os.listdir(config.CONFIG["workspace"]["checkpoint"]), reverse=True)
            if len(files) == 0:
                raise ValueError("No checkpoints found.")
            logger.info(f"Resuming from checkpoint: {files[0]}.")
            with open(os.path.join(config.CONFIG["workspace"]["checkpoint"], files[0]), "rb") as f:
                pop_scenario = pickle.load(f)
                pop_perturbation = pickle.load(f)
                current_solutions = pickle.load(f)
                archive_solution = pickle.load(f)
                evaluated_solutions = pickle.load(f)
                simulation_runner.evaluated_scenarios = Manager().dict(pickle.load(f))
                self.budget.initialize(other=pickle.load(f))
                self.logbook = pickle.load(f)
        else:
            pop_scenario = self.toolbox.pop_scenario()
            pop_perturbation = self.toolbox.pop_perturbation()
            archive_solution = []
            evaluated_solutions = []
            self.budget.initialize()

        while not self.budget.is_reached():
            # Evaluate the population.
            current_solutions, sim_num = self._evaluate(pop_scenario, pop_perturbation, evaluated_solutions)
            violated_solutions = [solution for solution in current_solutions if solution.is_violated]
            violated_solutions_count = len(violated_solutions)
            logger.info(f"The number of solutions violating the relation: {violated_solutions_count}.")
            archive_solution += [solution for solution in violated_solutions if solution not in archive_solution]

            pop_scenario = self.toolbox.select_scenario(pop_scenario)
            pop_perturbation = self.toolbox.select_perturbation(pop_perturbation)

            self.budget.acc_sim(sim_num)
            self.record_statistics(pop_scenario, self.budget.gen_num, pop_name="pop_scen")
            self.record_statistics(pop_perturbation, self.budget.gen_num, pop_name="pop_pert")
            self.record_statistics(violated_solutions, self.budget.gen_num, pop_name="solution", sim_num=sim_num)
            self.record_statistics(archive_solution, self.budget.gen_num, pop_name="archive",
                                   sim_num=self.budget.sim_num)

            # Generate offsprings.
            off_scenario = self._breed(pop_scenario, int(len(pop_scenario) // 2) * 2)
            off_perturbation = self._breed(pop_perturbation, int(len(pop_perturbation) // 2) * 2)
            pop_scenario += off_scenario
            pop_perturbation += off_perturbation
            self.budget.acc_gen()
            logger.info("Generation info:\n" + self.logbook.stream)
            logger.info(f"Number of simulations: {sim_num}/{self.budget.sim_num}.")

            # Store the current status into a checkpoint.
            with open(os.path.join(config.CONFIG["workspace"]["checkpoint"],
                                   f"{int(round(time.time() * 1000))}.pickle"), "wb") as f:
                pickle.dump(pop_scenario, f)
                pickle.dump(pop_perturbation, f)
                pickle.dump(current_solutions, f)
                pickle.dump(archive_solution, f)
                pickle.dump(evaluated_solutions, f)
                pickle.dump(list(simulation_runner.evaluated_scenarios), f)
                pickle.dump(self.budget, f)
                pickle.dump(self.logbook, f)

        logger.info(f"Terminate due to reaching the threshold.")
        # Store the complete solutions.
        self.dump_results(archive_solution, evaluated_solutions)

    def _evaluate(self, pop_scenario, pop_perturbation, evaluated_solutions):
        """Form complete solutions and evaluate their joint fitness as well as the individual fitness values.

        :param pop_scenario: The population of scenarios.
        :param pop_perturbation: The population of perturbations.
        :param evaluated_solutions: The evaluated complete solutions.
        :return: An archived complete solutions and the number of simulations.
        """
        # Generate complete solutions from archives.
        complete_solutions = [self.toolbox.collaborate(self.toolbox.clone(scenario), self.toolbox.clone(perturbation))
                              for scenario in pop_scenario for perturbation in pop_perturbation]

        # Remove repetitive complete solutions.
        unique_solutions = self.remove_duplicates(complete_solutions)

        # Evaluate joint fitness.
        candidates = [ind for ind in unique_solutions if ind not in evaluated_solutions]
        logger.info(f"#complete solutions: {len(complete_solutions)}, "
                    f"#unique solutions: {len(unique_solutions)}, "
                    f"#candidates: {len(candidates)}.")
        if len(candidates) > 0:
            candidates, sim_num = self.toolbox.evaluate_solutions(candidates)
            evaluated_solutions.extend(candidates)
        else:
            logger.warning("Candidate solution list is empty!")
            sim_num = 0
        archive_solution = [self.toolbox.clone(ind) for ind in evaluated_solutions if ind in unique_solutions]

        # Evaluate individual fitness.
        for scenario in pop_scenario:
            self.toolbox.evaluate_individual(scenario, archive_solution)
        for perturbation in pop_perturbation:
            self.toolbox.evaluate_individual(perturbation, archive_solution)

        return archive_solution, sim_num

    def _breed(self, population, size):
        """Perform selection, crossover and mutation on individuals.

        :param population: The individuals to be bred.
        :param size: The size of the offsprings.
        :return: A list of offsprings.
        """
        assert len(population) > 0
        if size % 2 != 0:
            raise ValueError("The size of offsprings must be even.")

        offsprings = self._tournament_DCD(population, size)
        offsprings = list(map(self.toolbox.clone, offsprings))
        for ind1, ind2 in zip(offsprings[::2], offsprings[1::2]):
            ind1.mate(ind2)
            ind1.mutate()
            ind2.mutate()
            del ind1.fitness.values
            del ind2.fitness.values
            ind1.correct()
            ind2.correct()
        return offsprings

    @staticmethod
    def _tournament_DCD(individuals, k):
        """Tournament selection based on dominance (D) between two individuals, if
        the two individuals do not inter-dominate the selection is made based on
        crowding distance (CD). This selection requires the individuals to have a
        :attr:`crowding_dist` attribute.

        :param individuals: A list of individuals to select from.
        :param k: The number of individuals to select.
        :returns: A list of selected individuals.
        """
        chosen = []
        for _ in range(k):
            ind1, ind2 = random.choice(individuals), random.choice(individuals)
            if ind1.fitness.dominates(ind2.fitness):
                chosen.append(ind1)
            elif ind2.fitness.dominates(ind1.fitness):
                chosen.append(ind2)
            elif ind1.fitness.crowding_dist < ind2.fitness.crowding_dist:
                chosen.append(ind2)
            elif ind1.fitness.crowding_dist > ind2.fitness.crowding_dist:
                chosen.append(ind1)
            elif random.random() <= 0.5:
                chosen.append(ind1)
            else:
                chosen.append(ind2)
        return chosen
