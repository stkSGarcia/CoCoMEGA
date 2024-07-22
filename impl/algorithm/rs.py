import logging
import os
import pickle
import time
from multiprocessing import Manager

from impl.algorithm.base import BaseAlgorithm
from impl.config import CONFIG
from impl.scenario import simulation_runner

logger = logging.getLogger(__name__)


class RandomSearch(BaseAlgorithm):
    def solve(self, resume=False):
        logger.info("Random search started.")
        logger.info(self.budget.print_budget())

        # Initialize the population or resume from the latest checkpoint.
        if resume:
            files = sorted(os.listdir(CONFIG["workspace"]["checkpoint"]), reverse=True)
            if len(files) == 0:
                raise ValueError("No checkpoints found.")
            logger.info(f"Resuming from checkpoint: {files[0]}.")
            with open(os.path.join(CONFIG["workspace"]["checkpoint"], files[0]), "rb") as f:
                current_solutions = pickle.load(f)
                archive_solution = pickle.load(f)
                evaluated_solutions = pickle.load(f)
                simulation_runner.evaluated_scenarios = Manager().list(pickle.load(f))
                self.budget.initialize(other=pickle.load(f))
                self.logbook = pickle.load(f)
        else:
            archive_solution = []
            evaluated_solutions = []
            self.budget.initialize()

        while not self.budget.is_reached():
            # Randomly generate complete solutions.
            complete_solutions = [self.toolbox.collaborate(self.toolbox.scenario(), self.toolbox.perturbation())
                                  for _ in range(CONFIG["scenario"]["pop_size"])]

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
            current_solutions = [self.toolbox.clone(ind) for ind in evaluated_solutions
                                 if ind.fitness.valid and ind in unique_solutions]

            violated_solutions = [solution for solution in current_solutions if solution.is_violated]
            violated_solutions_count = len(violated_solutions)
            logger.info(f"The number of solutions violating the relation: {violated_solutions_count}.")
            archive_solution += [solution for solution in violated_solutions if solution not in archive_solution]

            self.budget.acc_sim(sim_num)
            self.record_statistics(violated_solutions, self.budget.gen_num, pop_name="solution", sim_num=sim_num)
            self.record_statistics(archive_solution, self.budget.gen_num, pop_name="archive",
                                   sim_num=self.budget.sim_num)
            self.budget.acc_gen()
            logger.info("Generation info:\n" + self.logbook.stream)
            logger.info(f"Number of simulations: {sim_num}/{self.budget.sim_num}.")

            # Store the current status into a checkpoint.
            with open(os.path.join(CONFIG["workspace"]["checkpoint"], f"{int(round(time.time() * 1000))}.pickle"),
                      "wb") as f:
                pickle.dump(current_solutions, f)
                pickle.dump(archive_solution, f)
                pickle.dump(evaluated_solutions, f)
                pickle.dump(list(simulation_runner.evaluated_scenarios), f)
                pickle.dump(self.budget, f)
                pickle.dump(self.logbook, f)

        logger.info(f"Terminate due to reaching the threshold.")
        # Store the complete solutions.
        self.dump_results(archive_solution, "rs")
