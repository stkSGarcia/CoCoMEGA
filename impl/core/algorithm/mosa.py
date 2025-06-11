import logging
import random
import time
from collections import defaultdict
from operator import attrgetter
from typing import List

from deap import tools, base

from impl.core.algorithm.base import BaseAlgorithm

logger = logging.getLogger(__name__)


class MOSA(BaseAlgorithm):
    """Multi-Objective Search Algorithm.

    .. deprecated:: 0.0.1
       Not supported.
    """

    def __init__(self,
                 objectives,
                 pop_size,
                 toolbox: base.Toolbox,
                 time_budget,
                 max_iter,
                 seed=None):
        """Constructor.

        :param objectives: Objectives to optimize.
        :param pop_size: The size of the population.
        """
        super().__init__(toolbox, time_budget, max_iter, seed)
        self.objectives = objectives
        self.pop_size = pop_size

    def solve(self):
        """Main optimization loop for MOSA. Supports resuming from a checkpoint."""
        logger.info("MOSA started.")
        logger.info(f"Time budget: {self.time_budget}.")
        logger.info(f"Max iteration: {self.max_iter}.")

        # Initialize the uncovered objectives.
        uncovered_objectives = list(range(len(self.objectives)))

        # Initialize the first generation and an archive.
        population = self.toolbox.population(n=self.pop_size)
        archive = []

        # Evaluate the first generation.
        self.toolbox.evaluate(population)

        # Update archive.
        self._update_archive(archive, population, uncovered_objectives)

        gen = 0
        self._record_statistics(population, gen)
        start_time = time.perf_counter()
        while gen < self.max_iter and time.perf_counter() - start_time < self.time_budget:
            # Generate offsprings.
            offspring = self._generate_offspring(population, uncovered_objectives)

            # Evaluate the offsprings.
            invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
            self.toolbox.evaluate(invalid_individuals)

            # Update archive.
            self._update_archive(archive, offspring, uncovered_objectives)

            # Preference sort.
            F = self._preference_sorting(population + offspring, uncovered_objectives)

            if len(uncovered_objectives) == 0: break

            next_population = []
            index = 0
            while len(next_population) <= self.pop_size:
                if len(next_population) + len(F[index]) > self.pop_size: break
                next_population.extend(F[index])
                index += 1

            # Assign crowding distance.
            tools.emo.assignCrowdingDist(F[index])
            sorted_front = sorted(F[index], key=attrgetter("fitness.crowding_dist"), reverse=True)
            remain_len = self.pop_size - len(next_population)
            next_population.extend(sorted_front[:remain_len])

            gen += 1
            self._record_statistics(next_population, gen)

        return archive

    def _update_archive(self, archive: List, population: List, uncovered_objectives: List):
        """Add individuals meeting the objectives to the archive.

        :param archive: The list of archived individuals.
        :param population: The list of individuals to be archived.
        :param uncovered_objectives: The indices of uncovered objectives.
        """
        for individual in population:
            for idx, (fitness, objective) in enumerate(zip(individual.fitness.values, self.objectives)):
                if not objective(fitness): continue
                archive_idx, archive_individual = next(
                    ((i, ind) for i, ind in enumerate(archive) if idx in ind.covered_objectives),
                    (None, None)
                )
                if archive_individual is not None and archive_idx is not None:  # Individuals already in the archive.
                    if individual.fitness.dominates(archive_individual.fitness, [idx]):
                        individual.covered_objectives.append(idx)
                        archive[archive_idx] = individual
                        if idx in uncovered_objectives:
                            uncovered_objectives.remove(idx)
                else:
                    individual.covered_objectives.append(idx)
                    archive.append(individual)
                    if idx in uncovered_objectives:
                        uncovered_objectives.remove(idx)

    def _generate_offspring(self, population: List, uncovered_objectives: List) -> List:
        """Perform selection, crossover and mutation on individuals.

        :param population: The list of parent individuals.
        :param uncovered_objectives: The indices of uncovered objectives.
        :return: The list of offsprings.
        """
        population = self.toolbox.clone(population)
        offspring = []
        while len(offspring) < len(population):
            ind1 = self._tournament_selection(population, 10, uncovered_objectives)
            ind2 = self._tournament_selection(population, 10, uncovered_objectives)
            self.toolbox.mate(ind1, ind2)
            self.toolbox.mutate(ind1)
            self.toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values
            offspring.extend([ind1, ind2])
        return offspring

    @staticmethod
    def _tournament_selection(population: List, size: int, uncovered_objectives: List):
        """Tournament selection.

        :param population: The list of individuals to be selected.
        :param size: Tournament size.
        :param uncovered_objectives: The indices of uncovered objectives.
        :return: The best individual.
        """
        candidates = []
        for i in range(size):
            idx = random.randint(0, len(population) - 1)
            candidates.append(population[idx])

        best = candidates[0]
        for candidate in candidates:
            if candidate.fitness.dominates(best.fitness, uncovered_objectives):
                best = candidate
        return best

    def _preference_sorting(self, population: List, uncovered_objectives: List):
        population = self.toolbox.clone(population)
        F = []
        for idx in uncovered_objectives:
            best = population[0]
            for individual in population:
                if individual.fitness.dominates(best.fitness, [idx]):
                    best = individual
            F.append(best)
            population.remove(best)
        F = [F]
        if len(F[0]) > self.pop_size: return F
        if len(population) > 0:
            E = self._fast_nondominated_sort(population, uncovered_objectives)
            F += E
        return F

    @staticmethod
    def _fast_nondominated_sort(population: List, uncovered_objectives: List):
        map_fit_ind = defaultdict(list)
        for ind in population:
            map_fit_ind[ind.fitness].append(ind)
        fits = list(map_fit_ind.keys())

        fronts = [[]]
        current_front = []
        next_front = []
        dominating_fits = defaultdict(int)
        dominated_fits = defaultdict(list)

        for i, fit_i in enumerate(fits):
            for fit_j in fits[i + 1:]:
                if fit_i.dominates(fit_j, uncovered_objectives):
                    dominating_fits[fit_j] += 1
                    dominated_fits[fit_i].append(fit_j)
                elif fit_j.dominates(fit_i, uncovered_objectives):
                    dominating_fits[fit_i] += 1
                    dominated_fits[fit_j].append(fit_i)
            if dominating_fits[fit_i] == 0:
                current_front.append(fit_i)
                fronts[-1].extend(map_fit_ind[fit_i])

        while current_front:
            fronts.append([])
            for fit_p in current_front:
                for fit_d in dominated_fits[fit_p]:
                    dominating_fits[fit_d] -= 1
                    if dominating_fits[fit_d] == 0:
                        next_front.append(fit_d)
                        fronts[-1].extend(map_fit_ind[fit_d])
            current_front = next_front
            next_front = []

        return fronts
