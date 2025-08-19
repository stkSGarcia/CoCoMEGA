import uuid
from typing import Any

from deap import tools

from impl import config as cfg


class AbstractScenarioDefinition:
    """Abstract base class for a scenario definition."""

    def __new__(cls, instance=None):
        if isinstance(instance, cls):
            return instance
        return super().__new__(cls)

    def __init__(self, instance: 'AbstractScenarioDefinition'):
        if instance is None:
            self.id_ = uuid.uuid4().hex
        elif isinstance(instance, AbstractScenarioDefinition):
            self.id_ = instance.id_

    def dist(self, other: 'AbstractScenarioDefinition', **kwargs) -> float:
        """Compute a distance measure between two scenarios.
        The distance accounts for both scenario attributes and actor configurations.

        :param other: Another :class:`AbstractScenarioDefinition` instance to compare against.
        :return: Computed distance between two scenarios.
        """
        raise NotImplementedError("Subclasses must implement `dist`")

    def mate(self, other: 'AbstractScenarioDefinition', cxpb=cfg.CONFIG["scenario"]["cxpb"], **kwargs) -> None:
        """Apply crossover on two scenarios.

        :param other: Another :class:`AbstractScenarioDefinition` instance to mate with.
        :param cxpb: Crossover probability.
        """
        raise NotImplementedError("Subclasses must implement `mate`")

    def mutate(self, mutpb=cfg.CONFIG["scenario"]["mutpb"], **kwargs) -> None:
        """Mutate the scenario.

        :param mutpb: Mutation probability.
        """
        raise NotImplementedError("Subclasses must implement `mutate`")

    def __eq__(self, other: Any) -> bool:
        raise NotImplementedError("Subclasses must implement `__eq__`")

    @staticmethod
    def select(population, k=2):
        """Perform selection on a population (default: tournament selection).

        :param population: List of :class:`AbstractScenarioDefinition` instances.
        :param k: Number of individuals to select (default: 2).
        :return: Selected individuals.
        """
        return tools.selTournament(population, k=k, tournsize=cfg.CONFIG["scenario"]["tournament"])

    def assign_new_id(self):
        """Assign a new UUID to the scenario."""
        self.id_ = uuid.uuid4().hex

    def correct(self):
        """Correct the scenario by reassigning a new unique ID."""
        self.assign_new_id()
