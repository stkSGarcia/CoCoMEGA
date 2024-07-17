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
    def __init__(self, toolbox, budget, seed=None, keep_best=False):
        super().__init__(toolbox, budget, seed)
        self.keep_best = keep_best

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
            complete_solutions = [self.toolbox.collaborate(self.toolbox.scenario(), self.toolbox.perturbation())
                                  for _ in range(CONFIG["scenario"]["pop_size"])]
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
            current_solutions = [self.toolbox.clone(ind) for ind in evaluated_solutions
                                 if ind.fitness.valid and ind in unique_solutions]

            violated_solutions = [solution for solution in current_solutions if solution.is_violated]
            violated_solutions_count = len(violated_solutions)
            logger.info(f"The number of solutions violating the relation: {violated_solutions_count}.")
            archive_solution += [solution for solution in violated_solutions if solution not in archive_solution]

            self.budget.acc_sim(sim_num)
            self.record_statistics(complete_solutions, self.budget.gen_num, pop_name="pop")
            self.record_statistics(violated_solutions, self.budget.gen_num, pop_name="solution", sim_num=sim_num)
            self.record_statistics(archive_solution, self.budget.gen_num, pop_name="archive",
                                   sim_num=self.budget.sim_num)

            # Generate offsprings.
            if self.keep_best:
                population = list(map(self.toolbox.clone, complete_solutions))
                archive = tools.selBest(population, CONFIG["scenario"]["archive_size"])
                complete_solutions = self._breed(complete_solutions, len(complete_solutions) - len(archive))
                complete_solutions += archive
            else:
                complete_solutions = self._breed(complete_solutions, len(complete_solutions))

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
        self.dump_results(archive_solution, "gawa" if self.keep_best else "ga")

    def _breed(self, population, size):
        """Perform selection, crossover and mutation on individuals.

        @param population: The individuals to be bred.
        @param size: The size of the offsprings.
        @return: A list of offsprings.
        """
        assert len(population) > 0
        offsprings = tools.selTournament(population, k=size, tournsize=CONFIG["scenario"]["tournament"])
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

        if size % 2 == 1:
            child = offsprings[-1]
            child[0].mutate()
            child[1].mutate()
            del child.fitness.values
            child[0].correct()
            child[1].correct()

        return offsprings
