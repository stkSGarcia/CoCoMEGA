import random
from abc import abstractmethod, ABC
from typing import List

from impl.scenario.scenario_definition import ScenarioDefinition


class Perturbation:
    def __init__(self, vector):
        self.vector = vector

    def perturb(self, scenario: ScenarioDefinition):
        """Perturb the given scenario in place."""
        scenario.update(self.vector)

    @staticmethod
    def squash(perturbations: List):
        """Squash the given perturbations into one perturbation."""
        assert len(perturbations) > 0
        res = [None] * len(perturbations[0].vector)
        for perturbation in perturbations:
            for i, attr in enumerate(perturbation.vector):
                if attr is None: continue
                if isinstance(res[i], float):
                    res[i] += attr
                else:
                    res[i] = attr
        return Perturbation(res)

    def heterogeneous_distance(self, other):
        assert len(self.vector) == len(other.vector)
        scenario1, scenario2 = ScenarioDefinition(self.vector), ScenarioDefinition(other.vector)
        return scenario1.heterogeneous_distance(scenario2)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.vector == other.vector
        else:
            return False


class PerturbationFactory(ABC):
    @abstractmethod
    def spawn(self) -> Perturbation:
        raise NotImplementedError

    @staticmethod
    def _random(value_range):
        if value_range is None:
            return None
        if type(value_range[0]) is not type(value_range[1]):
            raise ValueError(f"Unmatched boundary types: [{type(value_range[0])}, {type(value_range[1])}].")
        if isinstance(value_range[0], float):
            return None if random.random() < 0.1 else random.uniform(value_range[0], value_range[1])
        elif isinstance(value_range[0], int):
            return None if random.random() < 0.1 else random.randint(value_range[0], value_range[1])
        else:
            raise ValueError(f"Unsupported boundary type: {type(value_range[0])}.")


class ActorPerturbationFactory(PerturbationFactory):
    def __init__(self, category, loc_x_range=None, loc_y_range=None, yaw_range=None, pitch_range=None,
                 speed_x_range=None, speed_y_range=None, typ_range=None, freeze_time=None, acc_x_range=None,
                 acc_y_range=None):
        self.category = category
        self.loc_x_range = loc_x_range
        self.loc_y_range = loc_y_range
        self.pitch_range = pitch_range
        self.yaw_range = yaw_range
        self.speed_x_range = speed_x_range
        self.speed_y_range = speed_y_range
        self.typ_range = typ_range
        self.freeze_time = freeze_time
        self.acc_x_range = acc_x_range
        self.acc_y_range = acc_y_range

    def spawn(self) -> Perturbation:
        vector = [
            self._random(self.loc_x_range),
            self._random(self.loc_y_range),
            self._random(self.yaw_range),
            self._random(self.pitch_range),
            self._random(self.speed_x_range),
            self._random(self.speed_y_range),
            self._random(self.typ_range),
            self._random(self.freeze_time),
            self._random(self.acc_x_range),
            self._random(self.acc_y_range),
        ]
        if self.category == "pedestrian":
            return Perturbation(vector[:8] + [None] * 16)
        elif self.category == "vehicle":
            return Perturbation([None] * 8 + vector[:] + [None] * 6)
        elif self.category == "object":
            return Perturbation([None] * 18 + vector[:4] + [None] * 2)
        else:
            raise ValueError(f"Unsupported actor category: {self.category}.")


class EnvPerturbationFactory(PerturbationFactory):
    def __init__(self, category, value_range):
        self.category = category
        self.value_range = value_range

    def spawn(self) -> Perturbation:
        vector = [self._random(self.value_range)]
        if self.category == "weather":
            return Perturbation([None] * 22 + vector + [None])
        elif self.category == "darkness":
            return Perturbation([None] * 23 + vector)
        else:
            raise ValueError(f"Unsupported env category: {self.category}.")


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
        if isinstance(other, self.__class__):
            return self.field == other.field and self.threshold == other.threshold
        else:
            return False


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
