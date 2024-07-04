import logging
import os
import pickle
import time
from multiprocessing import Manager

from deap import tools

from impl.algorithm.base import BaseAlgorithm
from impl.config import CONFIG
from impl.scenario import simulation_runner

logger = logging.getLogger(__name__)


class GeneticAlgorithm(BaseAlgorithm):
    def solve(self, resume=False):
        logger.info("Genetic algorithm started.")
        logger.info(self.budget.print_budget())

        # Initialize the population or resume from the latest checkpoint.
        if resume:
            files = sorted(os.listdir(CONFIG["workspace"]["checkpoint"]), reverse=True)
            if len(files) == 0:
                raise ValueError("No checkpoints found.")
            logger.info(f"Resuming from checkpoint: {files[0]}.")
            with open(os.path.join(CONFIG["workspace"]["checkpoint"], files[0]), "rb") as f:
                complete_solutions = pickle.load(f)
                archive_solution = pickle.load(f)
                evaluated_solutions = pickle.load(f)
                simulation_runner.evaluated_scenarios = Manager().list(pickle.load(f))
                self.budget.initialize(other=pickle.load(f))
                self.logbook = pickle.load(f)
        else:
            complete_solutions = [self.toolbox.collaborate(self.toolbox.scenario(), self.toolbox.perturbation()) for _
                                  in range(max(CONFIG["scenario"]["pop_size"], CONFIG["perturbation"]["pop_size"]))]
            archive_solution = []
            evaluated_solutions = []
            self.budget.initialize()

        while not self.budget.is_reached():
            # Remove repetitive complete solutions.
            unique_solutions = []
            for ind in complete_solutions:
                if ind not in unique_solutions:
                    unique_solutions.append(ind)

            # Evaluate joint fitness.
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

            violated_solutions = [solution for solution in archive_solution if solution.is_violated]
            violated_solutions_count = len(violated_solutions)
            logger.info(f"The number of solutions violating the relation: {violated_solutions_count}.")

            # Terminate if the archive has converged.
            if (self.budget.gen_num > 0 and
                    violated_solutions_count > self.budget.convergence_threshold * len(archive_solution)):
                logger.info(f"Terminate due to the number of violations reaching the threshold: "
                            f"{violated_solutions_count} > {self.budget.convergence_threshold}*{len(archive_solution)}.")
                break

            self.budget.acc_sim(sim_num)
            self.record_statistics(complete_solutions, self.budget.gen_num, pop_name="pop")
            self.record_statistics(violated_solutions, self.budget.gen_num, pop_name="solution", sim_num=sim_num)

            # Generate offsprings.
            complete_solutions = self._breed(complete_solutions)

            self.budget.acc_gen()
            logger.info("Generation info:\n" + self.logbook.stream)
            logger.info(f"Number of simulations: {sim_num}/{self.budget.sim_num}.")

            # Store the current status into a checkpoint.
            with open(os.path.join(CONFIG["workspace"]["checkpoint"], f"{int(round(time.time() * 1000))}.pickle"),
                      "wb") as f:
                pickle.dump(complete_solutions, f)
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

    def _breed(self, population):
        """Perform selection, crossover and mutation on individuals."""
        assert len(population) > 0
        offsprings = tools.selTournament(population, k=len(population), tournsize=CONFIG["scenario"]["tournament"])
        offsprings = list(map(self.toolbox.clone, offsprings))

        for child1, child2 in zip(offsprings[::2], offsprings[1::2]):
            child1[0].mate(child2[0])
            child1[1].mate(child2[1])
            child1[0].mutate()
            child1[1].mutate()
            child2[0].mutate()
            child2[1].mutate()
            del child1.fitness.values
            del child2.fitness.values
            child1[0].correct()
            child1[1].correct()
            child2[0].correct()
            child2[1].correct()

        return offsprings
