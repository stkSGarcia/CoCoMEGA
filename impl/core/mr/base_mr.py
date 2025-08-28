import math
from abc import ABC, abstractmethod
from typing import Any, List, Tuple

from deap import tools

from impl import config as cfg
from impl.core.scenario.base_scenario import AbstractScenarioDefinition


class AbstractPerturbation(ABC):
    """Abstract base class for a single perturbation."""

    def __init__(self, enabled: bool):
        """Constructor.

        :param enabled: Whether this perturbation is enabled.
        """
        self.enabled = enabled

    @abstractmethod
    def apply_perturbation(self, scenario: AbstractScenarioDefinition):
        """Apply the perturbation to the scenario.

        :param scenario: The scenario to be perturbed.
        """
        raise NotImplementedError("Subclasses must implement `apply_perturbation`")

    @abstractmethod
    def dist(self, other: 'AbstractPerturbation', **kwargs) -> float:
        """Compute distance between two perturbations.

        :param other: Another :class:`AbstractPerturbation`.
        :return: The distance between this perturbation and another one.
        """
        raise NotImplementedError("Subclasses must implement `dist`")

    @abstractmethod
    def mate(self, other: 'AbstractPerturbation', cxpb, **kwargs) -> None:
        """Perform crossover operation with another perturbation.

        :param other: Another :class:`AbstractPerturbation`.
        :param cxpb: Crossover probability.
        """
        raise NotImplementedError("Subclasses must implement `mate`")

    @abstractmethod
    def mutate(self, mutpb, *args, **kwargs) -> None:
        """Mutate the perturbation.

        :param mutpb: Mutation probability.
        """
        raise NotImplementedError("Subclasses must implement `mutate`")

    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        raise NotImplementedError("Subclasses must implement `__eq__`")

    def perturb(self, scenario: AbstractScenarioDefinition):
        """Perturb scenario if enabled.

        :param scenario: The scenario to be perturbed.
        """
        if self.is_enabled():
            self.apply_perturbation(scenario)

    def score(self, co_population, **kwargs):
        """An optional score function for offspring selection. Override this to improve perturbation breeding.

        :param co_population: The co-population of perturbations.
        """
        return 0

    def is_enabled(self) -> bool:
        """Check whether this perturbation is enabled."""
        return self.enabled

    def enable(self):
        """Enable this perturbation."""
        self.enabled = True

    def disable(self):
        """Disable this perturbation."""
        self.enabled = False


class Perturbations(list):
    """Representation of a sequence of perturbations."""

    def perturb(self, scenario: AbstractScenarioDefinition):
        """Perturb the source scenario in place.

        :param scenario: The source scenario to perturb.
        """
        for perturbation in self:
            perturbation.perturb(scenario)

    def dist(self, other, **kwargs):
        """Calculate the heterogeneous distance between this sequence of perturbations and another one.

        :param other: Another :class:`Perturbations`.
        :return: The heterogeneous distance between this sequence of perturbations and another one.
        """
        # if not isinstance(other, self.__class__):
        if str(type(other)) != str(type(self)):
            raise ValueError(f"Unmatched types: [{type(self)}, {type(other)}].")
        squared_dist = 0.0
        for this, that in zip(self, other):
            squared_dist += pow(this.dist(that, **kwargs), 2)
        return math.sqrt(squared_dist)

    def score(self, co_population, **kwargs):
        """An optional score function for offspring selection.

        :param co_population: The co-population of perturbations.
        """
        score = 0
        for perturbation in self:
            score += perturbation.score(co_population, **kwargs)

        return score

    @staticmethod
    def select(population, k=2):
        """Selection within a list of perturbation sequences.

        :param population: A list of perturbation sequences.
        :param k: Number of sequences to select.
        :return: The selected perturbation sequences.
        """
        return tools.selTournament(population, k=k, tournsize=cfg.CONFIG["perturbation"]["tournament"])

    def mate(self, other, cxpb=cfg.CONFIG["perturbation"]["cxpb"], **kwargs):
        """Crossover two sequences of perturbations in place.

        :param other: Another :class:`Perturbations`.
        :param cxpb: Crossover probability.
        """
        if not isinstance(other, self.__class__):
            raise ValueError(f"Unmatched types: [{type(self)}, {type(other)}].")
        for this, that in zip(self, other):
            this.mate(that, cxpb, **kwargs)

    def mutate(self, mutpb=cfg.CONFIG["perturbation"]["mutpb"], **kwargs):
        """Mutate this sequence of perturbation in place.

        :param mutpb: Mutation probability.
        """
        for perturbation in self:
            perturbation.mutate(mutpb, **kwargs)

    def correct(self):
        """Correction function after breeding."""
        return


class AbstractPerturbationFactory(ABC):
    """Abstract base class for Perturbation Factory"""

    @abstractmethod
    def spawn(self) -> AbstractPerturbation:
        """Randomly spawn a perturbation."""
        pass

    @abstractmethod
    def has(self, perturbations: Perturbations):
        """Check whether ``perturbations`` contain perturbations spawned by this factory.

        :param perturbations: A :class:`Perturbations` instance.
        :return: Whether the ``perturbations`` contain this perturbation.
        """
        pass


class AbstractRelation(ABC):
    """Abstract base class for defining output relations."""

    @abstractmethod
    def is_violated(self, original_output: Any, perturbed_output: Any, **kwargs) -> Tuple[bool, float]:
        """Check whether the output relation is violated.

        :param original_output: The original output.
        :param perturbed_output: The perturbed output.
        :return: A tuple of (violation_detected, extent_of_violation).
        """
        pass


class MR:
    """Representation of a metamorphic relation."""

    def __init__(self, perturbation_factories: List[AbstractPerturbationFactory], relation: AbstractRelation):
        """Constructor.

        :param perturbation_factories: A list of :class:`PerturbationFactory` instances.
        :param relation: A common output relation.
        """
        self.perturbation_factories = perturbation_factories
        self.relation = relation

    def initialize(self) -> List[AbstractPerturbation]:
        """Randomly initialize a list of perturbations."""
        perturbations = []
        for factory in self.perturbation_factories:
            perturbations.append(factory.spawn())
        if len([p for p in perturbations if p.is_enabled()]) == 0:
            perturbations[0].enable()
        return perturbations

    def has(self, perturbations: Perturbations) -> bool:
        """Check whether ``perturbations`` contain this perturbation.

        :param perturbations: A :class:`Perturbations` instance.
        :return: Whether the ``perturbations`` contain this perturbation.
        """
        for perturbation_factory in self.perturbation_factories:
            if perturbation_factory.has(perturbations):
                return True
        return False


class MRSet:
    """Representation of a set of metamorphic relations sharing common source scenario constraints and output relation."""

    def __init__(self, mrs: List[MR], source_gen_func):
        """Constructor.

        :param mrs: A list of :class:`MR` instances.
        :param source_gen_func: A function that generates random source scenarios suitable for this MR set.
        """
        # Check relations.
        assert mrs
        assert all(mr.relation == mrs[0].relation for mr in mrs)

        self.mrs = mrs
        self.source_gen_func = source_gen_func
        self.relation = mrs[0].relation

    def is_violated(self, source, follow_up, **kwargs) -> Tuple[bool, float]:
        """Determine if this relation is violated and quantify the extent of violation.

        :param source: The :class:`DataFrame` of the source result.
        :param follow_up: The :class:`DataFrame` of the follow-up result.
        :return: The ``bool`` value indicates whether the relation is violated.
            The ``float`` value denotes the extent to which this relation is violated.
        """
        return self.relation.is_violated(source, follow_up, **kwargs)

    def violated_mrs(self, perturbations_list: List[Perturbations]) -> List[List[int]]:
        """Determine the violated metamorphic relations within the list of perturbation sequences.

        :param perturbations_list: A list of :class:`Perturbations`.
        :return: The indices of the violated metamorphic relations.
        """
        return [[i for i, mr in enumerate(self.mrs) if mr.has(perturbations)]
                for perturbations in perturbations_list]

    def generate_scenario(self, **kwargs) -> AbstractScenarioDefinition:
        """Generate a random source scenario for this MR."""
        return self.source_gen_func(**kwargs)

    def generate_perturbation(self) -> Perturbations:
        """Randomly initialize a sequence of perturbations."""
        perturbations = []
        for mr in self.mrs:
            perturbations += mr.initialize()
        return Perturbations(perturbations)
