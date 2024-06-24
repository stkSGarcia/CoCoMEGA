import logging
import math
import random
from abc import ABC
from enum import Enum, auto
from typing import List

import numpy as np
import pandas as pd
from deap import tools
from tslearn.metrics import dtw_path

from impl.config import CONFIG
from impl.scenario import scenario_definition
from impl.scenario.scenario_definition import ScenarioDefinition

logger = logging.getLogger(__name__)


class Operation(Enum):
    ADD = auto()
    REMOVE = auto()
    REPLACE = auto()


class Perturbation:
    def __init__(self, category: str, operation: Operation, value, mark, enabled=True):
        self.category = category
        self.operation = operation
        self.value = value
        self.mark = mark
        self.enabled = enabled

    def perturb(self, scenario: ScenarioDefinition):
        if self.enabled is False: return
        if self.category in ScenarioDefinition.DYNAMIC:
            if self.operation == Operation.ADD:
                scenario.add_actor(self.category, self.value, mark=self.mark)
            elif self.operation == Operation.REMOVE:
                scenario.remove_actor(self.category, self.value)
            elif self.operation == Operation.REPLACE:
                scenario.replace_actor(self.category, self.value[0], self.value[1], mark=self.mark)
            else:
                raise ValueError(f"Unsupported operation: {self.operation}.")
        elif self.category in ScenarioDefinition.ATTRIBUTES:
            scenario.update_attribute(self.category, self.value)
        else:
            raise ValueError(f"Unsupported category: {self.category}.")

    def dist(self, other, scaling=CONFIG["perturbation"]["dist_scaling"]):
        if not isinstance(other, self.__class__):
            raise ValueError(f"Unmatched types: [{type(self)}, {type(other)}].")
        if self.enabled == other.enabled and self.category == other.category and self.operation == other.operation:
            if self.operation == Operation.ADD:
                return self.value.dist(other.value)
            if self.operation == Operation.REMOVE:
                return 0 if self.value == other.value else pow(scaling, 2)
            elif self.operation == Operation.REPLACE:
                return self.value[1].dist(other.value[1])
        return pow(scaling, 2)

    def mate(self, other, cxpb=CONFIG["perturbation"]["cxpb"]):
        if not isinstance(other, self.__class__):
            raise ValueError(f"Unmatched types: [{type(self)}, {type(other)}].")
        if self.category != other.category or self.operation != other.operation: return
        if random.random() < cxpb:
            self.enabled, other.enabled = other.enabled, self.enabled
        if self.operation == Operation.ADD:
            self.value.mate(other.value, cxpb=cxpb)
        elif self.operation == Operation.REPLACE:
            self.value[1].mate(other.value[1], cxpb=cxpb)

    def mutate(self, mutpb=CONFIG["perturbation"]["mutpb"],
               eta=CONFIG["perturbation"]["mut_eta"],
               std=CONFIG["perturbation"]["mut_std"]):
        if random.random() < mutpb:
            self.enabled = not self.enabled
        if self.operation == Operation.ADD:
            self.value.mutate(mutpb=mutpb, eta=eta, std=std)
        elif self.operation == Operation.REPLACE:
            self.value[1].mutate(mutpb=mutpb, eta=eta, std=std)

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
    def perturb(self, scenario: ScenarioDefinition):
        for perturbation in self:
            perturbation.perturb(scenario)

    def dist(self, other, scaling=CONFIG["perturbation"]["dist_scaling"]):
        if not isinstance(other, self.__class__):
            raise ValueError(f"Unmatched types: [{type(self)}, {type(other)}].")
        dist = 0.0
        for this, that in zip(self, other):
            dist += this.dist(that, scaling=scaling)
        return math.sqrt(dist)

        # if len(self) == 0 or len(other) == 0: return 0.0
        # dp = [np.inf] * (len(other) + 1)
        # prev = 0.0
        # for p in self:
        #     for i in range(1, len(other) + 1):
        #         temp = dp[i]
        #         dp[i] = min(min(prev, temp), dp[i - 1]) + p.dist(other[i - 1])
        #         prev = temp
        #     prev = np.inf
        # return math.sqrt(dp[-1])

    @staticmethod
    def select(population, k=2):
        return tools.selTournament(population, k=k, tournsize=CONFIG["perturbation"]["tournament"])

    def mate(self, other, cxpb=CONFIG["perturbation"]["cxpb"]):
        if not isinstance(other, self.__class__):
            raise ValueError(f"Unmatched types: [{type(self)}, {type(other)}].")
        for this, that in zip(self, other):
            this.mate(that, cxpb=cxpb)

    def mutate(self, mutpb=CONFIG["perturbation"]["mutpb"],
               eta=CONFIG["perturbation"]["mut_eta"],
               std=CONFIG["perturbation"]["mut_std"]):
        for perturbation in self:
            perturbation.mutate(mutpb=mutpb, eta=eta, std=std)

    def correct(self):
        pass


class PerturbationFactory:
    def __init__(self, category: str, boundary, operation: Operation = None, mark=False):
        self.category = category
        self.boundary = boundary
        self.operation = operation
        self.mark = mark

        if category in ScenarioDefinition.DYNAMIC:
            cls = getattr(scenario_definition, category.capitalize())
            if operation == Operation.ADD:
                self._spawn_func = lambda: cls.generate_random(region=boundary)
            elif operation == Operation.REMOVE:
                self._spawn_func = lambda: boundary
            elif operation == Operation.REPLACE:
                self._spawn_func = lambda: (boundary, cls.generate_random(region=boundary))
            else:
                raise ValueError(f"Unsupported perturbation operation: {operation}.")
        elif category in ScenarioDefinition.ATTRIBUTES:
            self._spawn_func = lambda: boundary.random(category)
        else:
            raise ValueError(f"Unsupported perturbation category: {category}.")

    def spawn(self) -> Perturbation:
        return Perturbation(self.category, self.operation, self._spawn_func(), mark=self.mark)

    def get_label(self):
        if self.category in ScenarioDefinition.DYNAMIC and self.boundary:
            return f"{self.boundary.name.lower()}{'-mark' if self.mark else ''}"
        return None


class Relation(ABC):
    _s, _f, _d = "source", "follow-up", "fov-nearest-distance"

    def __init__(self, field, threshold):
        self.field = field
        self.threshold = threshold
        self._extent_func = None

    def is_violated(self, source, follow_up, labels=None) -> (bool, float):
        """Determine if this relation is violated and quantify the extent of violation.

        @param source: The `DataFrame` of the source result.
        @param follow_up: The `DataFrame` of the follow-up result.
        @param labels: Used in simulation-based strategy.
        @return: The `bool` value indicates whether the relation is violated.
        The `float` value denotes the extent to which this relation is violated.
        """
        labels = [f"{Relation._d}-{label}" for label in labels or set()]
        matches = [(source.index.values[i], follow_up.index.values[j])
                   for i, j in dtw_path(source[self.field], follow_up[self.field])[0]]
        df = pd.DataFrame([(
            source.loc[i, self.field],
            follow_up.loc[j, self.field],
            *[np.nanmin([source.loc[i].get(label, np.nan), follow_up.loc[j].get(label, np.nan)])
              for label in [Relation._d] + labels],
        ) for i, j in matches], columns=(Relation._s, Relation._f, *([Relation._d] + labels)))

        df = df.loc[(df[labels].min(axis=1) if len(labels) > 0 else df[Relation._d])
                    < CONFIG["violation"]["threshold"]["max_ego_distance"]]
        if df.empty: return False, None

        df["extent"] = df.apply(self._extent_func, axis=1, result_type="reduce")
        # df = df.loc[df["extent"].abs() > CONFIG["violation"]["threshold"][self.field]]
        # if df.empty: return False, None

        extent = df["extent"].mean()
        return extent > 0, extent

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                self.field == other.field and
                self.threshold == other.threshold)


class Invariance(Relation):
    def __init__(self, field, threshold=0.01):
        super().__init__(field, threshold)
        self._extent_func = lambda row: abs(row[Relation._f] - row[Relation._s]) - row[Relation._s] * self.threshold


class Decreasing(Relation):
    def __init__(self, field, threshold=0.1):
        super().__init__(field, threshold)
        self._extent_func = lambda row: row[Relation._f] - row[Relation._s] * (1.0 - self.threshold)


class Increasing(Relation):
    def __init__(self, field, threshold=0.1):
        super().__init__(field, threshold)
        self._extent_func = lambda row: row[Relation._s] * (1.0 + self.threshold) - row[Relation._f]


class MR:
    def __init__(self, perturbation_factories: List[PerturbationFactory], relation: Relation):
        self.perturbation_factories = perturbation_factories
        self.relation = relation

    def initialize(self) -> List[Perturbation]:
        perturbations = []
        for factory in self.perturbation_factories:
            perturbations.append(factory.spawn())
        if len([p for p in perturbations if p.enabled]) == 0:
            perturbations[0].enabled = True
        return perturbations


class MRSet:
    def __init__(self, mrs: List[MR]):
        # Check relations
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
        perturbations = []
        for mr in self.mrs:
            perturbations += mr.initialize()
        return Perturbations(perturbations)

    def is_violated(self, source, follow_up) -> (bool, float):
        return self.relation.is_violated(source, follow_up, self.labels)
