import logging
import math
import random
from abc import ABC
from enum import Enum, auto
from typing import List

import numpy as np
import pandas as pd
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
    def __init__(self, category: str, operation: Operation, value):
        self.category = category
        self.operation = operation
        self.value = value

    def perturb(self, scenario: ScenarioDefinition):
        if self.category in ScenarioDefinition.DYNAMIC:
            if self.operation == Operation.ADD:
                scenario.add_actor(self.category, self.value)
            elif self.operation == Operation.REMOVE:
                scenario.remove_actor(self.category, self.value)
            elif self.operation == Operation.REPLACE:
                scenario.replace_actor(self.category, self.value[0], self.value[1])
            else:
                raise ValueError(f"Unsupported operation: {self.operation}.")
        elif self.category in ScenarioDefinition.ATTRIBUTES:
            scenario.update_attribute(self.category, self.value)
        else:
            raise ValueError(f"Unsupported category: {self.category}.")

    def dist(self, other):
        if self.category == other.category and self.operation == other.operation:
            if self.operation == Operation.ADD:
                return self.value.dist(other.value)
            if self.operation == Operation.REMOVE:
                return 0 if self.value == other.value else pow(CONFIG["perturbation"]["dist_scaling"], 2)
            elif self.operation == Operation.REPLACE:
                return self.value[1].dist(other.value[1])
        return pow(CONFIG["perturbation"]["dist_scaling"], 2)

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                self.category == other.category and
                self.operation == other.operation and
                self.value == other.value)

    def __repr__(self):
        return f"{self.__class__.__name__}(category={self.category}, operation={self.operation}, value={self.value})"


class Perturbations(list):
    def perturb(self, scenario: ScenarioDefinition):
        for perturbation in self:
            perturbation.perturb(scenario)

    def dist(self, other):
        if not isinstance(other, self.__class__):
            raise ValueError(f"Unmatched types: [{type(self)}, {type(other)}].")
        if len(self) == 0 or len(other) == 0: return 0.0

        dp = [np.inf] * (len(other) + 1)
        prev = 0.0
        for p in self:
            for i in range(1, len(other) + 1):
                temp = dp[i]
                dp[i] = min(min(prev, temp), dp[i - 1]) + p.dist(other[i - 1])
                prev = temp
            prev = np.inf
        return math.sqrt(dp[-1])

    def mate(self, other):
        if not isinstance(other, self.__class__):
            raise ValueError(f"Unmatched types: [{type(self)}, {type(other)}].")
        common = min(len(self), len(other))
        for i in range(common):
            if random.random() < CONFIG["perturbation"]["cxpb"]:
                self[i], other[i] = other[i], self[i]
        less, more = (self, other) if len(self) < len(other) else (other, self)
        while len(more) > common:
            if random.random() < CONFIG["perturbation"]["cxpb"]:
                less.append(more.pop(common))
            else:
                common += 1


class PerturbationFactory:
    def __init__(self, category: str, boundary, operation: Operation = None):
        self.category = category
        self.boundary = boundary
        self.operation = operation

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
        return Perturbation(self.category, self.operation, self._spawn_func())


class Relation(ABC):
    _s, _f, _d = "source", "follow-up", "fov-nearest-distance"

    def __init__(self, field, threshold):
        self.field = field
        self.threshold = threshold
        self._extent_func = None

    def is_violated(self, source, follow_up, regions=None) -> (bool, float):
        """Determine if this relation is violated and quantify the extent of violation.

        @param source: The `DataFrame` of the source result.
        @param follow_up: The `DataFrame` of the follow-up result.
        @param regions: Used in simulation-based strategy.
        @return: The `bool` value indicates whether the relation is violated.
        The `float` value denotes the extent to which this relation is violated.
        """
        regions = [f"{Relation._d}-{region.name.lower()}" for region in regions or set()] \
            if CONFIG["violation"]["strategy"] == "simulation" else []
        matches = [(source.index.values[i], follow_up.index.values[j])
                   for i, j in dtw_path(source[self.field], follow_up[self.field])[0]] \
            if CONFIG["violation"]["dtw"] else [(i, i) for i in source.index.intersection(follow_up.index)]

        df = pd.DataFrame([(
            source.loc[i, self.field],
            follow_up.loc[j, self.field],
            *[np.nanmin([source.loc[i].get(region, np.nan), follow_up.loc[j].get(region, np.nan)])
              for region in [Relation._d] + regions],
        ) for i, j in matches], columns=(Relation._s, Relation._f, *([Relation._d] + regions)))

        if CONFIG["violation"]["strategy"] == "simulation":
            df = df.loc[df[regions].min(axis=1) < CONFIG["violation"]["threshold"]["max_ego_distance"]]
        else:
            if CONFIG["violation"]["strategy"] != "curve":
                logger.warning("Unrecognized strategy, falling back to `curve`.")
            df = df.loc[(df[Relation._s] - df[Relation._f]).abs() > CONFIG["violation"]["threshold"][self.field]]

        if df.empty: return False, 0.0
        extents = df.apply(self._extent_func, axis=1, result_type="reduce")
        extent = extents.mean()
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

    def spawn(self) -> Perturbation:
        return random.choice(self.perturbation_factories).spawn()


class MRSet:
    def __init__(self, mrs: List[MR]):
        # Check relations
        assert len(mrs) > 0
        assert all(mr.relation == mrs[0].relation for mr in mrs)

        self.mrs = mrs
        self.regions = set(factory.boundary
                           for mr in mrs
                           for factory in mr.perturbation_factories
                           if factory.category in ScenarioDefinition.DYNAMIC)
        self.relation = mrs[0].relation
        self.field = self.relation.field

    def spawn(self) -> Perturbation:
        return random.choice(self.mrs).spawn()

    def is_violated(self, source, follow_up) -> (bool, float):
        return self.relation.is_violated(source, follow_up, self.regions)

    def mutate(self, perturbations: Perturbations):
        """Mutate a give sequence of perturbations.

        @param perturbations: The sequence of perturbations to be mutated.
        @return: The mutated sequence of perturbations.
        """
        if random.random() > CONFIG["perturbation"]["mutpb"]: return perturbations
        if random.random() < CONFIG["perturbation"]["mut_del"]:
            # Remove one previous perturbation.
            if len(perturbations) > 1: perturbations.pop()
        else:
            # Add perturbations.
            times = 1
            while random.random() < CONFIG["perturbation"]["mut_add"] ** times:
                perturbations.append(self.spawn())
                times += 1
        return perturbations
