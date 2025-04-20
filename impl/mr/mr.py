import logging
import math
import random
from abc import ABC
from collections.abc import Sequence
from enum import Enum, auto
from functools import partial
from typing import List

import numpy as np
import pandas as pd
from deap import tools
from tslearn.metrics import dtw_path

from impl import config as cfg
from impl.scenario import scenario_definition
from impl.scenario.scenario_definition import ScenarioDefinition

logger = logging.getLogger(__name__)


class Operation(Enum):
    ADD = auto()
    REMOVE = auto()
    REPLACE = auto()


class Perturbation:
    """Representation of a perturbation."""

    def __init__(self, category, boundary, operation: Operation, value, mark, enabled=True):
        """Constructor.

        :param category: The category of the object to be perturbed.
        :param boundary: The boundary of the object to be perturbed.
        :param operation: The type of perturbation.
        :param value: The value of the object to be perturbed.
        :param mark: The mark of the object to be perturbed.
        :param enabled: Whether the perturbation is enabled.
        """
        self.category = category
        self.boundary = boundary
        self.operation = operation
        self.value = value
        self.mark = mark
        self.enabled = enabled

    def perturb(self, scenario: ScenarioDefinition):
        """Perturb the source scenario in place.

        :param scenario: The source scenario to perturb.
        """
        if self.enabled is False: return
        if self.category in ScenarioDefinition.DYNAMIC:
            if self.operation == Operation.ADD:
                scenario.add_actor(self.category, self.value, mark=self.mark,
                                   tilt_dir=scenario.trajectory.get("direction", None))
            elif self.operation == Operation.REMOVE:
                scenario.remove_actor(self.category, self.value)
            elif self.operation == Operation.REPLACE:
                scenario.replace_actor(self.category, self.boundary, self.value, mark=self.mark,
                                       tilt_dir=scenario.trajectory.get("direction", None))
            else:
                raise ValueError(f"Unsupported operation: {self.operation}.")
        elif self.category in ScenarioDefinition.ATTRIBUTES:
            scenario.update_attribute(self.category, self.value)
        elif isinstance(self.category, Sequence):
            if self.category[0] == "ego":
                scenario.update_ego(self.category[1], self.value)
            elif self.category[0] in ScenarioDefinition.DYNAMIC:
                scenario.update_actor(self.category[0], self.category[1], self.value)
            else:
                raise ValueError(f"Unsupported category: {self.category}.")
        else:
            raise ValueError(f"Unsupported category: {self.category}.")

    def dist(self, other, scaling=cfg.CONFIG["perturbation"]["dist_scaling"]):
        """Calculate the heterogeneous distance between this perturbation and another one.

        :param other: Another perturbation.
        :param scaling: The scaling factor for categorical values.
        :return: The heterogeneous distance between this perturbation and another one.
        """
        if not isinstance(other, self.__class__):
            raise ValueError(f"Unmatched types: [{type(self)}, {type(other)}].")
        if self.enabled == other.enabled and self.category == other.category and self.operation == other.operation:
            if self.category in ScenarioDefinition.DYNAMIC:
                if self.operation == Operation.ADD or self.operation == Operation.REPLACE:
                    return self.value.dist(other.value)
                if self.operation == Operation.REMOVE:
                    return 0 if self.value == other.value else pow(scaling, 2)
            elif self.category in ScenarioDefinition.ATTRIBUTES:
                return self._dist_values(self.value, other.value, *self.boundary[self.category], scaling)
            elif isinstance(self.category, Sequence):
                return self._dist_values(self.value, other.value, *self.boundary[self.category[1]], scaling)
        return pow(1.0, 2)

    @staticmethod
    def _dist_values(this, that, lower, upper, scaling):
        if isinstance(lower, float):
            return pow(abs(this - that) / (upper - lower), 2) if upper != lower else 0
        elif isinstance(lower, int):
            # TODO: within the same category.
            return pow(scaling * (0.0 if this == that else 1.0), 2)
        else:
            raise ValueError(f"Unsupported boundaries: [{lower}, {upper}].")

    def mate(self, other, cxpb=cfg.CONFIG["perturbation"]["cxpb"]):
        """Crossover two perturbations in place.

        :param other: Another perturbation.
        :param cxpb: Crossover probability.
        """
        if not isinstance(other, self.__class__):
            raise ValueError(f"Unmatched types: [{type(self)}, {type(other)}].")
        if self.category != other.category or self.operation != other.operation: return
        if random.random() < cxpb:
            self.enabled, other.enabled = other.enabled, self.enabled
        if self.category in ScenarioDefinition.DYNAMIC:
            if self.operation == Operation.ADD or self.operation == Operation.REPLACE:
                self.value.mate(other.value, cxpb=cxpb)
        elif (self.category in ScenarioDefinition.ATTRIBUTES or
              (isinstance(self.category, Sequence)) and self.category[0] in ScenarioDefinition.DYNAMIC + ["ego"]):
            if random.random() < cxpb:
                self.value, other.value = other.value, self.value
        else:
            raise ValueError(f"Unsupported category: {self.category}.")

    def mutate(self, mutpb=cfg.CONFIG["perturbation"]["mutpb"],
               eta=cfg.CONFIG["perturbation"]["mut_eta"],
               std=cfg.CONFIG["perturbation"]["mut_std"]):
        """Mutate this perturbation in place.

        :param mutpb: Mutation probability.
        :param eta: Crowding degree of the mutation. A high eta will produce a mutant resembling its parent,
        while a small eta will produce a solution much more different.
        :param std: Standard deviation for the gaussian addition mutation.
        """
        if random.random() < mutpb:
            self.enabled = not self.enabled
        if self.category in ScenarioDefinition.DYNAMIC:
            if self.operation == Operation.ADD or self.operation == Operation.REPLACE:
                self.value.mutate(mutpb=mutpb, eta=eta, std=std)
        elif self.category in ScenarioDefinition.ATTRIBUTES:
            if random.random() < mutpb:
                self.value = self.boundary.random(self.category)
        elif isinstance(self.category, Sequence) and self.category[0] in ScenarioDefinition.DYNAMIC + ["ego"]:
            if random.random() < mutpb:
                self.value = self.boundary.random(self.category[1])
        else:
            raise ValueError(f"Unsupported category: {self.category}.")

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                self.enabled == other.enabled and
                self.category == other.category and
                self.operation == other.operation and
                self.value == other.value and
                self.mark == other.mark)

    def __repr__(self):
        return (f"{self.__class__.__name__}(enabled={self.enabled}, "
                f"category={self.category}, "
                f"operation={self.operation.name}, "
                f"value={self.value}, "
                f"mark={self.mark})")


class Perturbations(list):
    """Representation of a sequence of perturbations."""

    def perturb(self, scenario: ScenarioDefinition):
        """Perturb the source scenario in place.

        :param scenario: The source scenario to perturb.
        """
        for perturbation in self:
            perturbation.perturb(scenario)

    def dist(self, other, scaling=cfg.CONFIG["perturbation"]["dist_scaling"]):
        """Calculate the heterogeneous distance between this sequence of perturbations and another one.

        :param other: Another sequence of perturbations.
        :param scaling: The scaling factor for categorical values.
        :return: The heterogeneous distance between this sequence of perturbations and another one.
        """
        # if not isinstance(other, self.__class__):
        if str(type(other)) != str(type(self)):
            raise ValueError(f"Unmatched types: [{type(self)}, {type(other)}].")
        dist = 0.0
        for this, that in zip(self, other):
            dist += this.dist(that, scaling=scaling)
        return math.sqrt(dist)

    @staticmethod
    def select(population, k=2):
        """Selection within a list of perturbation sequences.

        :param population: A list of perturbation sequences.
        :param k: Number of sequences to select.
        :return: The selected perturbation sequences.
        """
        return tools.selTournament(population, k=k, tournsize=cfg.CONFIG["perturbation"]["tournament"])

    def mate(self, other, cxpb=cfg.CONFIG["perturbation"]["cxpb"]):
        """Crossover two sequences of perturbations in place.

        :param other: Another sequence of perturbations.
        :param cxpb: Crossover probability.
        """
        if not isinstance(other, self.__class__):
            raise ValueError(f"Unmatched types: [{type(self)}, {type(other)}].")
        for this, that in zip(self, other):
            this.mate(that, cxpb=cxpb)

    def mutate(self, mutpb=cfg.CONFIG["perturbation"]["mutpb"],
               eta=cfg.CONFIG["perturbation"]["mut_eta"],
               std=cfg.CONFIG["perturbation"]["mut_std"]):
        """Mutate this sequence of perturbation in place.

        :param mutpb: Mutation probability.
        :param eta: Crowding degree of the mutation. A high eta will produce a mutant resembling its parent,
        while a small eta will produce a solution much more different.
        :param std: Standard deviation for the gaussian addition mutation.
        """
        for perturbation in self:
            perturbation.mutate(mutpb=mutpb, eta=eta, std=std)

    def correct(self):
        """Correction function after breeding."""
        pass


class PerturbationFactory:
    """Factory for perturbation generation."""

    def __init__(self, category, boundary, operation: Operation = Operation.REPLACE, mark=False):
        """Constructor.

        :param category: The category of the object to be perturbed.
        :param boundary: The boundary of the object to be perturbed.
        :param operation: The type of perturbation.
        :param mark: The mark of the object to be perturbed.
        """
        self.category = category
        self.boundary = boundary
        self.operation = operation
        self.mark = mark

        if category in ScenarioDefinition.DYNAMIC:
            cls = getattr(scenario_definition, category.capitalize())
            if operation == Operation.ADD or operation == Operation.REPLACE:
                self._spawn_func = lambda: cls.generate_random(region=boundary)
            elif operation == Operation.REMOVE:
                self._spawn_func = lambda: boundary
            else:
                raise ValueError(f"Unsupported perturbation operation: {operation}.")
        elif category in ScenarioDefinition.ATTRIBUTES:
            self._spawn_func = lambda: boundary.random(category)
        elif isinstance(category, Sequence):
            self._spawn_func = lambda: boundary.random(category[1])
        else:
            raise ValueError(f"Unsupported perturbation category: {category}.")

    def spawn(self) -> Perturbation:
        """Randomly spawn a perturbation."""
        return Perturbation(self.category, self.boundary, self.operation, self._spawn_func(), mark=self.mark)

    def get_label(self):
        """Get the region of the marked actor."""
        if self.category in ScenarioDefinition.DYNAMIC and self.boundary:
            return f"{self.boundary.name.lower()}{'-mark' if self.mark else ''}"
        return None

    def has(self, perturbations: Perturbations):
        """Check whether `perturbations` contain perturbations spawned by this factory.

        :param perturbations: A sequence of perturbations.
        :return: Whether the `perturbations` contain this perturbation.
        """
        for perturbation in perturbations:
            if (perturbation.enabled and
                    perturbation.category == self.category and
                    perturbation.boundary == self.boundary and
                    perturbation.operation == self.operation):
                return True
        return False


class Relation(ABC):
    """Representation of an output relation."""
    _s, _f, _d = "source", "follow-up", "fov-nearest-distance"

    def __init__(self, field, threshold, percentage: bool):
        """Constructor.

        :param field: The field used to examine the relation.
        :param threshold: The threshold in percentage or absolute value.
        :param percentage: Whether the threshold is a percentage.
        """
        self.field = field
        self.threshold = threshold
        self.percentage = percentage
        self._extent_func = None

    def is_violated(self, source, follow_up, labels=None) -> (bool, float):
        """Determine if this relation is violated and quantify the extent of violation.

        :param source: The `DataFrame` of the source result.
        :param follow_up: The `DataFrame` of the follow-up result.
        :param labels: Labels determining the perturbed objects.
        :return: The `bool` value indicates whether the relation is violated.
        The `float` value denotes the extent to which this relation is violated.
        """
        labels = Relation.convert_labels(labels)
        _, df = Relation.dtw_dataframe(source, follow_up, self.field, labels) if cfg.CONFIG["violation"]["dtw"] \
            else Relation.pairwise_dataframe(source, follow_up, self.field, labels)
        critical_intervals = Relation.critical_intervals(df, labels)
        df = df.loc[critical_intervals]
        if df.empty: return False, None

        df["extent"] = df.apply(self._extent_func, axis=1, result_type="reduce")
        extent = df["extent"].mean()
        return extent > 0, extent

    @staticmethod
    def convert_labels(labels):
        """Convert the given labels into those in the result `DataFrame`."""
        return [f"{Relation._d}-{label}" for label in labels or set()]

    @staticmethod
    def dtw_dataframe(source, follow_up, field, labels):
        """Generate a `DataFrame` from source and follow-up results using DTW algorithm.

        :param source: The `DataFrame` of the source result.
        :param follow_up: The `DataFrame` of the follow-up result.
        :param field: The metric to be compared.
        :param labels: Labels determining the perturbed objects.
        :return: A tuple of the DTW path and the generated `DataFrame`.
        """
        if cfg.CONFIG["violation"]["strategy"] == "position":
            func, columns = partial(dtw_path), ["position_x", "position_y"]
        else:
            func, columns = partial(dtw_path, global_constraint="sakoe_chiba", sakoe_chiba_radius=5), field
        matches = [(source.index.values[i], follow_up.index.values[j])
                   for i, j in func(source[columns].to_numpy(), follow_up[columns].to_numpy())[0]]
        df = pd.DataFrame([(
            source.loc[i, field], follow_up.loc[j, field],
            *[np.nanmin([source.loc[i].get(label, np.nan), follow_up.loc[j].get(label, np.nan)])
              for label in [Relation._d] + labels],
        ) for i, j in matches], columns=(Relation._s, Relation._f, *([Relation._d] + labels)))
        return matches, df

    @staticmethod
    def pairwise_dataframe(source, follow_up, field, labels):
        """Generate a `DataFrame` from source and follow-up results based on common indices.

        :param source: The `DataFrame` of the source result.
        :param follow_up: The `DataFrame` of the follow-up result.
        :param field: The metric to be compared.
        :param labels: Labels determining the perturbed objects.
        :return: A tuple of matches and the generated `DataFrame`.
        """
        indices = source.index.union(follow_up.index)
        df = pd.DataFrame([(
            source[field].get(i, np.nan), follow_up[field].get(i, np.nan),
            *[np.nanmin([d.get(label, pd.Series()).get(i, np.nan) for d in [source, follow_up]])
              for label in [Relation._d] + labels],
        ) for i in indices], columns=(Relation._s, Relation._f, *([Relation._d] + labels)))
        df.set_index(indices, inplace=True)
        return [(i, i) for i in indices], df

    @staticmethod
    def critical_intervals(dataframe, labels):
        """Fiter the given `DataFrame` by removing data where the perturbed
        objects are not in the field of view of the ego vehicle.

        :param dataframe: The `DataFrame` to be filtered.
        :param labels: Labels determining the perturbed objects.
        :return: A sequence of indices that indicate the critical intervals.
        """
        return dataframe.index[(dataframe[labels].min(axis=1) if len(labels) > 0 else dataframe[Relation._d])
                               < cfg.CONFIG["violation"]["max_ego_distance"]]

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                self.field == other.field and
                self.threshold == other.threshold and
                self.percentage == other.percentage)


class Invariance(Relation):
    """Invariance output relation."""

    def __init__(self, field, threshold=0.01, percentage: bool = True):
        super().__init__(field, threshold, percentage)
        if percentage:
            self._extent_func = lambda row: abs(row[Relation._f] - row[Relation._s]) - row[Relation._s] * self.threshold
        else:
            self._extent_func = lambda row: abs(row[Relation._f] - row[Relation._s]) - self.threshold


class Decreasing(Relation):
    """Decreasing output relation."""

    def __init__(self, field, threshold=0.1, percentage: bool = True):
        super().__init__(field, threshold, percentage)
        if percentage:
            self._extent_func = lambda row: row[Relation._f] - row[Relation._s] * (1.0 - self.threshold)
        else:
            self._extent_func = lambda row: row[Relation._f] - row[Relation._s] + self.threshold


class Increasing(Relation):
    """Increasing output relation."""

    def __init__(self, field, threshold=0.1, percentage: bool = True):
        super().__init__(field, threshold, percentage)
        if percentage:
            self._extent_func = lambda row: row[Relation._s] * (1.0 + self.threshold) - row[Relation._f]
        else:
            self._extent_func = lambda row: row[Relation._s] + self.threshold - row[Relation._f]


class MR:
    """Representation of a metamorphic relation."""

    def __init__(self, perturbation_factories: List[PerturbationFactory], relation: Relation):
        """Constructor.

        :param perturbation_factories: A list of `PerturbationFactory` instances.
        :param relation: A common output relation.
        """
        self.perturbation_factories = perturbation_factories
        self.relation = relation

    def initialize(self) -> List[Perturbation]:
        """Randomly initialize a list of perturbations."""
        perturbations = []
        for factory in self.perturbation_factories:
            perturbations.append(factory.spawn())
        if len([p for p in perturbations if p.enabled]) == 0:
            perturbations[0].enabled = True
        return perturbations

    def has(self, perturbations: Perturbations) -> bool:
        """Check whether `perturbations` contain this perturbation.

        :param perturbations: A sequence of perturbations.
        :return: Whether the `perturbations` contain this perturbation.
        """
        for perturbation_factory in self.perturbation_factories:
            if perturbation_factory.has(perturbations):
                return True
        return False


class MRSet:
    """Representation of a set of metamorphic relations sharing a common output relation."""

    def __init__(self, mrs: List[MR]):
        """Constructor.

        :param mrs: A list of `MR` instances.
        """
        # Check relations.
        assert mrs
        assert all(mr.relation == mrs[0].relation for mr in mrs)

        self.mrs = mrs
        self.labels = set(factory.get_label()
                          for mr in mrs
                          for factory in mr.perturbation_factories
                          if factory.category in ScenarioDefinition.DYNAMIC)
        self.relation = mrs[0].relation
        self.field = self.relation.field

    def initialize(self) -> Perturbations:
        """Randomly initialize a sequence of perturbations."""
        perturbations = []
        for mr in self.mrs:
            perturbations += mr.initialize()
        return Perturbations(perturbations)

    def is_violated(self, source, follow_up) -> (bool, float):
        """Determine if this relation is violated and quantify the extent of violation.

        :param source: The `DataFrame` of the source result.
        :param follow_up: The `DataFrame` of the follow-up result.
        :return: The `bool` value indicates whether the relation is violated.
        The `float` value denotes the extent to which this relation is violated.
        """
        return self.relation.is_violated(source, follow_up, self.labels)

    def violated_mrs(self, perturbations_list: List[Perturbations]) -> List[List[int]]:
        """Determine the violated metamorphic relations within the list of perturbation sequences.

        :param perturbations_list: A list of perturbation sequences.
        :return: The indices of the violated metamorphic relations.
        """
        return [[i for i, mr in enumerate(self.mrs) if mr.has(perturbations)]
                for perturbations in perturbations_list]
