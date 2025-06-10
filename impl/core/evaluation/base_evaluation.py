from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Any
import numpy as np

from multiprocessing import Manager, Process
from impl.core.scenario.base_scenario import AbstractScenarioDefinition


class BaseEvaluator(ABC):
    """Abstract base class for evaluation of scenarios."""

    def __init__(self, mr_set):
        self.mr_set = mr_set
        self.evaluated_scenarios = Manager().dict()

    @staticmethod
    def _get_scenarios(solutions):
        scenarios = []
        for solution in solutions:
            scenarios.append(solution[0])
            follow_up = deepcopy(solution[0])
            follow_up.assign_new_id()
            solution[1].perturb(follow_up)
            scenarios.append(follow_up)
        assert len(scenarios) == len(solutions) * 2
        return scenarios

    def evaluate_solutions(self, solutions):
        scenarios = self._get_scenarios(solutions)
        results, sim_num = self.run_scenarios(scenarios)
        for solution, source, follow_up in zip(solutions, results[::2], results[1::2]):
            if source is not None and follow_up is not None:
                solution.source = source
                solution.follow_up = follow_up
                solution.is_violated, extent = self.fitness(source, follow_up)
                if extent:
                    solution.fitness.values = extent
                else:
                    del solution.fitness.values
            else:
                solution.is_violated = False
                del solution.fitness.values

        return solutions, sim_num

    @staticmethod
    def evaluate_individual(individual, complete_solutions):
        """Evaluate the individual fitness of a scenario or a sequence of perturbations.

        :param individual: The individual to be evaluated.
        :param complete_solutions: The list of complete solutions with fitness evaluated.
        :return: The individual with fitness evaluated.
        """
        index = 0 if isinstance(individual, AbstractScenarioDefinition) else 1
        involved = []
        for solution in complete_solutions:
            if solution[index] == individual and solution.fitness.valid:
                involved.append(solution.fitness.values)
        if len(involved) > 0:
            sorted_involved = sorted(involved, key=lambda x: tuple(np.array(x) * -np.array((1.0,))))
            individual.fitness.values = sorted_involved[0]
        else:
            del individual.fitness.values
        return individual

    def fitness(self, source: Any, follow_up: Any):
        """
        Calculate the fitness value and check if it violates the metamorphic relations.
        By default, it returns (violated, extent) the same as mr_set. Override to modify fitness logic.

        :param source: Simulation results of the source scenario.
        :param follow_up: Simulation results of the follow-up scenario.
        :return: A tuple containing a bool value indicating whether it violates the relation and the fitness value.

        """
        return self.mr_set.is_violated(source, follow_up)

    @abstractmethod
    def run_scenarios(self, scenarios: List[AbstractScenarioDefinition], **kwargs):
        """Run a list of scenarios and retrieve the results.

        :param scenarios: List of scenarios to be executed.
        """
        raise NotImplementedError("Subclasses must implement `run_scenarios`")

    def set_evaluated_scenarios(self, evaluated_scenarios):
        self.evaluated_scenarios = evaluated_scenarios

    def get_evaluated_scenarios(self):
        return self.evaluated_scenarios
