from typing import Any

import impl.config as cfg
from impl.core.scenario.base_scenario import AbstractScenarioDefinition


class ScenarioDefinition(AbstractScenarioDefinition):
    """Base class for scenario definitions in a domain-specific implementation."""

    def dist(self, other: 'ScenarioDefinition', **kwargs) -> float:
        """
        Compute a distance measure between two scenarios.
        The distance accounts for both scenario attributes and actor configurations.

        :param other: Another :class:`ScenarioDefinition` instance to compare against.
        :return: Computed distance between two scenarios.
        """
        # TODO Implement distance function
        pass

    def mate(self, other: 'ScenarioDefinition', cxpb=cfg.CONFIG["scenario"]["cxpb"], **kwargs) -> None:
        """
        Apply crossover on two scenarios.

        :param other: Another :class:`ScenarioDefinition` instance to mate with.
        :param cxpb: Crossover probability.
        """
        # TODO Implement crossover operator
        pass

    def mutate(self, mutpb=cfg.CONFIG["scenario"]["mutpb"], **kwargs) -> None:
        """
        Mutate the scenario.

        :param mutpb: Mutation probability.
        """
        # TODO Implement mutation operator
        pass

    def __eq__(self, other: Any) -> bool:
        """
        Check if this scenario is equal to another scenario.
        """
        # TODO Implement equality criteria
        pass
