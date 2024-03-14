import random
from abc import abstractmethod, ABC
from typing import List

from impl.config import CONFIG
from impl.scenario.scenario_definition import Boundary, ScenarioDefinition, Vehicle, Walker, Static


class Perturbation:
    def __init__(self, category: str, value):
        self.category = category
        self.value = value

    def perturb(self, scenario: ScenarioDefinition):
        scenario.update(self.category, self.value, CONFIG["perturbation"]["replace_pb"])

    @staticmethod
    def squash(perturbations: List) -> ScenarioDefinition:
        assert len(perturbations) > 0
        scenario = ScenarioDefinition()
        for perturbation in perturbations:
            scenario.update(perturbation.category, perturbation.value)
        return scenario

    def __repr__(self):
        return f"{self.category}: {self.value}"


class PerturbationFactory:
    _ATTR_MAP = {"vehicle": Vehicle, "walker": Walker, "static": Static}

    def __init__(self, category: str, boundary: Boundary):
        self.category = category
        self.boundary = boundary
        if self.category in ["vehicle", "walker", "static"]:
            cls = PerturbationFactory._ATTR_MAP[self.category]
            self._spawn_func = lambda: cls.generate_random(self.boundary)
        elif self.category in ["weather", "darkness"]:
            self._spawn_func = lambda: self.boundary.random(self.category)
        else:
            raise ValueError(f"Unsupported actor category: {self.category}.")

    def spawn(self) -> Perturbation:
        return Perturbation(self.category, self._spawn_func())


class Relation(ABC):
    def __init__(self, field, threshold):
        self.field = field
        self.threshold = threshold

    @abstractmethod
    def is_violated(self, source, result) -> (bool, float):
        """Determine if this relation is violated and quantify the extent of violation.

        @param source: The value of source result.
        @param result: The value of follow-up result.
        @return: The `bool` value indicates whether the relation is violated.
        The `float` value denotes the extent to which this relation is violated.
        """
        raise NotImplementedError

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                self.field == other.field and
                self.threshold == other.threshold)


class Invariance(Relation):
    def __init__(self, field, threshold=0.01):
        super().__init__(field, threshold)

    def is_violated(self, source, result) -> (bool, float):
        diff = abs(result - source)
        extent = min(abs(result - source * (1 + self.threshold)), abs(result - source * (1 - self.threshold)))
        return diff > self.threshold, extent


class Decreasing(Relation):
    def __init__(self, field, threshold=0.1):
        super().__init__(field, threshold)

    def is_violated(self, source, result) -> (bool, float):
        diff = source - result
        extent = result - source * (1 - self.threshold)
        return diff < source * self.threshold, extent


class Increasing(Relation):
    def __init__(self, field, threshold=0.1):
        super().__init__(field, threshold)

    def is_violated(self, source, result) -> (bool, float):
        diff = result - source
        extent = source * (1 + self.threshold) - result
        return diff < source * self.threshold, extent


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

    def is_violated(self, source, result) -> (bool, float):
        return self.relation.is_violated(source, result)
