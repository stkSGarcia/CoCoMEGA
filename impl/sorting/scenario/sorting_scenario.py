from typing import Any
import random
import numpy as np
from impl.core.scenario.base_scenario import AbstractScenarioDefinition
import impl.config as cfg


class ScenarioDefinition(AbstractScenarioDefinition):
    """Base class for scenario definitions in a domain-specific implementation."""
    def __init__(self, instance=None):
        """
        Initialize the scenario definition with an optional instance.
        """
        super().__init__(instance)
        if instance is None:
            self.input_data = None
            self.input_size = None
        elif isinstance(instance, ScenarioDefinition):
            self.input_data = instance.input_data.copy()
            self.input_size = instance.input_size

    @staticmethod
    def generate_random(min_val=0, max_val=100, input_size=10):
        """
        Generate a random scenario definition with input data of specified size and value range.
        """
        scenario = ScenarioDefinition()
        scenario.set_input_data([random.randint(min_val, max_val) for _ in range(input_size)])
        return scenario

    def dist(self, other, **kwargs) -> float:
        """
        Compute a distance measure between two scenarios.
        """
        if not isinstance(other, ScenarioDefinition):
            return float('inf')
        min_len = min(len(self.input_data), len(other.input_data))
        if min_len == 0:
            return 1.0
        return float(sum(abs(a - b) for a, b in zip(self.input_data[:min_len], other.input_data[:min_len]))) / min_len

    def mate(self, other: 'AbstractScenarioDefinition', cxpb=cfg.CONFIG["scenario"]["cxpb"], **kwargs) -> None:
        """
        Apply crossover on two scenarios.
        """
        if not isinstance(other, ScenarioDefinition):
            return
        if random.random() < cxpb:
            pivot = random.randint(1, min(len(self.input_data), len(other.input_data)) - 1)
            self.input_data[:pivot], other.input_data[:pivot] = (
                other.input_data[:pivot].copy(), self.input_data[:pivot].copy()
            )

    def mutate(self, mutpb=cfg.CONFIG["scenario"]["mutpb"], **kwargs) -> None:
        """
        Mutate the scenario by randomly changing an element in the input data.
        """
        if random.random() < mutpb and self.input_data:
            index = random.randint(0, len(self.input_data) - 1)
            delta = random.choice([-1, 1]) * random.randint(1, 5)
            self.input_data[index] = max(0, self.input_data[index] + delta)

    def __eq__(self, other: Any) -> bool:
        """
        Check if this scenario is equal to another scenario.
        """
        return isinstance(other, ScenarioDefinition) and self.input_data == other.input_data

    def set_input_data(self, input_data):
        """
        Set the input data for the scenario.
        """
        self.input_data = input_data
        self.input_size = len(input_data)