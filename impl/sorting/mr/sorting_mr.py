import random
from typing import Any
import numpy as np

from impl.core.mr.base_mr import AbstractPerturbation, AbstractPerturbationFactory, Perturbations, AbstractRelation
from impl.core.scenario.base_scenario import AbstractScenarioDefinition


class InputPermutationPerturbation(AbstractPerturbation):
    """
    Perturbation that applies a permutation to the input data of a scenario.
    """

    def __init__(self, permutation, enabled=True):
        """
        Initialize the perturbation with a specific permutation.
        """
        super().__init__(enabled)
        self.permutation = permutation  # store applied permutation for reproducibility

    def apply_perturbation(self, scenario: AbstractScenarioDefinition):
        """
        Apply the permutation to the input data of the scenario.
        """
        original_input = scenario.input_data
        scenario.input_data = [original_input[i] for i in self.permutation]

    def dist(self, other: 'AbstractPerturbation', **kwargs) -> float:
        """
        Calculate the distance between this perturbation and another perturbation.
        """
        if not isinstance(other, InputPermutationPerturbation):
            return float('inf')
        if self.permutation is None or other.permutation is None:
            return 1.0
        return float(np.sum(self.permutation != other.permutation)) / len(self.permutation)

    def mate(self, other, cxpb, **kwargs) -> None:
        """
        Perform crossover between this perturbation and another perturbation.
        """
        if random.random() < cxpb and self.permutation is not None and other.permutation is not None:
            size = len(self.permutation)
            a, b = sorted(random.sample(range(size), 2))

            # Extract slices
            slice1 = self.permutation[a:b]
            slice2 = other.permutation[a:b]

            # Helper to fill rest of permutation maintaining order and uniqueness
            def fill(preserved_slice, donor_perm):
                result = [-1] * size
                result[a:b] = preserved_slice
                fill_values = [x for x in donor_perm if x not in preserved_slice]
                i = 0
                for idx in list(range(0, a)) + list(range(b, size)):
                    result[idx] = fill_values[i]
                    i += 1
                return result

            new_self_perm = fill(slice1, other.permutation)
            new_other_perm = fill(slice2, self.permutation)

            self.permutation = np.array(new_self_perm)
            other.permutation = np.array(new_other_perm)

    def mutate(self, mutpb, *args, **kwargs) -> None:
        """
        Mutate the permutation by swapping two elements with a probability of `mutpb`.
        """
        if random.random() < mutpb and self.permutation is not None:
            i, j = random.sample(range(len(self.permutation)), 2)
            self.permutation[i], self.permutation[j] = self.permutation[j], self.permutation[i]

    def __eq__(self, other: Any) -> bool:
        """
        Check if this perturbation is equal to another perturbation.
        """
        return isinstance(other, InputPermutationPerturbation) and (sum(self.permutation != other.permutation) == 0)


class InputPermutationFactory(AbstractPerturbationFactory):
    """Factory for creating input permutation perturbations."""

    def __init__(self, input_size):
        """
        Initialize the factory with the size of the input data.
        :param input_size: Size of the input data to be permuted.
        """
        self.input_size = input_size

    def spawn(self) -> AbstractPerturbation:
        """
        Spawn a new input permutation perturbation with a random permutation.
        """
        permutation = np.random.permutation(self.input_size)
        return InputPermutationPerturbation(permutation, enabled=random.random() < 0.5)

    def has(self, perturbations: Perturbations):
        """
        Check if the factory has an input permutation perturbation in the provided set of perturbations.
        """
        return any(isinstance(p, InputPermutationPerturbation) for p in perturbations)


class AdditiveShiftPerturbation(AbstractPerturbation):
    """Perturbation that adds a random shift to the input data of a scenario."""

    def __init__(self, shift=None, enabled=True):
        """
        Initialize the perturbation with a specific shift value.
        """
        super().__init__(enabled)
        self.shift = shift if shift is not None else random.randint(0, 10)

    def apply_perturbation(self, scenario: AbstractScenarioDefinition):
        """
        Apply the additive shift to the input data of the scenario.
        """
        scenario.input_data = [x + self.shift for x in scenario.input_data]

    def dist(self, other, **kwargs) -> float:
        """
        Calculate the distance between this perturbation and another perturbation.
        """
        if not isinstance(other, AdditiveShiftPerturbation):
            return float('inf')
        return abs(self.shift - other.shift)

    def mate(self, other, cxpb, **kwargs) -> None:
        """
        Perform crossover between this perturbation and another perturbation.
        """
        if random.random() < cxpb:
            new_shift1, new_shift2 = (self.shift + other.shift), np.abs(self.shift - other.shift)
            self.shift, other.shift = new_shift1, new_shift2

    def mutate(self, mutpb, *args, **kwargs) -> None:
        """
        Mutate the shift by adding a random value with a probability of `mutpb`.
        """
        if random.random() < mutpb:
            self.shift += random.randint(0, 10)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, AdditiveShiftPerturbation) and self.shift == other.shift


class AdditiveShiftFactory(AbstractPerturbationFactory):
    """Factory for creating additive shift perturbations."""

    def spawn(self) -> AbstractPerturbation:
        """
        Spawn a new additive shift perturbation with a random shift value.
        """
        return AdditiveShiftPerturbation(enabled=random.random() < 0.5)

    def has(self, perturbations: Perturbations):
        """
        Check if the factory has an additive shift perturbation in the provided set of perturbations.
        """
        return any(isinstance(p, AdditiveShiftPerturbation) for p in perturbations)


class SortedOutputEqualityRelation(AbstractRelation):
    """Relation that checks if the perturbed output has the same sorted result as the original output."""

    def is_violated(self, original_output: Any, perturbed_output: Any, **kwargs) -> (bool, float):
        """
        Check whether the perturbed output has the same sorted result as original.
        Use Jaccard similarity or exact match depending on use-case.

        :param original_output: The original output of the scenario.
        :param perturbed_output: The perturbed output of the scenario.
        :return: A tuple containing a boolean indicating if the relation is violated and the extent of violation.
        """
        shift = (np.sum(perturbed_output) - np.sum(original_output)) / len(original_output)
        perturbed_output_adjusted = [p - shift for p in perturbed_output]
        is_violation = original_output != perturbed_output_adjusted
        violation_extent = float(np.sum(np.array(original_output) != np.array(perturbed_output_adjusted))) / len(
            original_output)
        return is_violation, violation_extent
