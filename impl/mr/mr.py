import logging
import math
import random
from abc import ABC
from typing import List

import pandas as pd
from tslearn.metrics import dtw_path

from impl.config import CONFIG
from impl.scenario.scenario_definition import Boundary, ScenarioDefinition, Vehicle, Walker, Static

logger = logging.getLogger(__name__)


class Perturbation:
    def __init__(self, category: str, value):
        self.category = category
        self.value = value

    def perturb(self, scenario: ScenarioDefinition):
        scenario.update(self.category, self.value)

    @staticmethod
    def squash(perturbations: List) -> ScenarioDefinition:
        assert len(perturbations) > 0
        scenario = ScenarioDefinition()
        for perturbation in perturbations:
            scenario.update(perturbation.category, perturbation.value)
        return scenario

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                self.category == other.category and
                self.value == other.value)

    def __repr__(self):
        return f"{self.category}: {self.value}"


class PerturbationFactory:
    _ATTR_MAP = {"vehicle": Vehicle, "walker": Walker, "static": Static}

    def __init__(self, category: str, boundary: Boundary, id_: str = None):
        self.id_ = id_
        self.category = category
        self.boundary = boundary
        if self.category in ["vehicle", "walker", "static"]:
            cls = PerturbationFactory._ATTR_MAP[self.category]
            self._spawn_func = lambda x: cls.generate_random(self.boundary, id_=x)
        elif self.category in ["weather", "darkness"]:
            self._spawn_func = lambda x: self.boundary.random(self.category)
        else:
            raise ValueError(f"Unsupported actor category: {self.category}.")

    def spawn(self) -> Perturbation:
        return Perturbation(self.category, self._spawn_func(self.id_))  # TODO: generate an actor with different id.


class Relation(ABC):
    _s, _f = "source", "follow-up"

    def __init__(self, field, threshold):
        self.field = field
        self.threshold = threshold
        self._extent_func = None

    def is_violated(self, source, follow_up) -> (bool, float):
        """Determine if this relation is violated and quantify the extent of violation.

        @param source: The `DataFrame` of the source result.
        @param follow_up: The `DataFrame` of the follow-up result.
        @return: The `bool` value indicates whether the relation is violated.
        The `float` value denotes the extent to which this relation is violated.
        """
        if CONFIG["violation"]["dtw"]:
            matches, _ = dtw_path(source[self.field], follow_up[self.field])
            df = pd.DataFrame([(source.loc[source.index.values[i], self.field],
                                follow_up.loc[follow_up.index.values[j], self.field]) for i, j in matches],
                              columns=(Relation._s, Relation._f))
        else:
            df = pd.merge(source, follow_up, left_index=True, right_index=True)
            df.rename(columns={f"{self.field}_x": Relation._s, f"{self.field}_y": Relation._f}, inplace=True)

        if CONFIG["violation"]["strategy"] == "simulation":
            pass  # TODO
        else:
            if CONFIG["violation"]["strategy"] != "curve":
                logger.warning("Unrecognized strategy, falling back to `curve`.")
            df = df.loc[(df[Relation._s] - df[Relation._f]).abs() > CONFIG["violation"]["threshold"][self.field]]

        if df.empty: return False, 0.0
        extents = df.apply(self._extent_func, axis=1, result_type="reduce")
        extents = extents[extents > 0]
        if extents.empty: return False, 0.0
        extent = extents.pow(2).sum()
        return True, math.sqrt(extent)

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
        self._extent_func = lambda row: max(0, row[Relation._f] - row[Relation._s] * (1.0 - self.threshold))


class Increasing(Relation):
    def __init__(self, field, threshold=0.1):
        super().__init__(field, threshold)
        self._extent_func = lambda row: max(0, row[Relation._s] * (1.0 + self.threshold) - row[Relation._f])


class MR:
    def __init__(self, perturbation_factories: List[PerturbationFactory], relation: Relation):
        self.perturbation_factories = perturbation_factories
        self.relation = relation

    def generate_perturbation(self) -> Perturbation:
        return random.choice(self.perturbation_factories).spawn()


class MRSet:
    def __init__(self, mrs: List[MR]):
        # Check relations
        assert len(mrs) > 0
        assert all(mr.relation == mrs[0].relation for mr in mrs)

        self.mrs = mrs
        self.relation = mrs[0].relation

    def field(self):
        return self.relation.field

    def is_violated(self, source, result) -> (bool, float):
        return self.relation.is_violated(source, result)
