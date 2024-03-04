import random
from abc import abstractmethod, ABC
from typing import List

from impl.scenario.scenario import ScenarioDefinition


class Perturbation(ABC):
    def __init__(self, uid):
        self.uid = uid

    @abstractmethod
    def perturb(self, scenario: ScenarioDefinition):
        """Perturb the given scenario in place."""
        raise NotImplementedError


class ActorPerturbation(Perturbation):
    def __init__(self, uid, loc_x=None, loc_y=None, yaw=None, pitch=None, speed_x=None, speed_y=None, typ=None,
                 freeze_time=None, acc_x=None, acc_y=None):
        super().__init__(uid)
        self.loc_x = loc_x
        self.loc_y = loc_y
        self.yaw = yaw
        self.pitch = pitch
        self.speed_x = speed_x
        self.speed_y = speed_y
        self.typ = typ
        self.freeze_time = freeze_time
        self.acc_x = acc_x
        self.acc_y = acc_y

    def perturb(self, scenario: ScenarioDefinition):
        if self.uid == "pedestrian":
            scenario.update_pedestrian(self.loc_x, self.loc_y, self.yaw, self.pitch, self.speed_x, self.speed_y,
                                       self.typ, self.freeze_time)
        elif self.uid == "vehicle":
            scenario.update_vehicle(self.loc_x, self.loc_y, self.yaw, self.pitch, self.speed_x, self.speed_y, self.typ,
                                    self.freeze_time)
        elif self.uid == "object":
            scenario.update_object(self.loc_x, self.loc_y, self.yaw, self.pitch)
        else:
            raise ValueError(f"Unsupported actor type: {self.uid}.")

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (self.uid == other.uid and
                    self.loc_x == other.loc_x and
                    self.loc_y == other.loc_y and
                    self.yaw == other.yaw and
                    self.pitch == other.pitch and
                    self.speed_x == other.speed_x and
                    self.speed_y == other.speed_y and
                    self.typ == other.typ and
                    self.freeze_time == other.freeze_time and
                    self.acc_x == other.acc_x and
                    self.acc_y == other.acc_y)
        else:
            return False


class EnvPerturbation(Perturbation):
    def __init__(self, uid, value):
        super().__init__(uid)
        self.value = value

    def perturb(self, scenario: ScenarioDefinition):
        if self.uid == "weather":
            scenario.update_weather(self.value)
        elif self.uid == "darkness":
            scenario.update_darkness(self.value)
        else:
            raise ValueError(f"Unsupported env type: {self.uid}.")

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.uid == other.uid and self.value == other.value
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

    def spawn(self) -> ActorPerturbation:
        return ActorPerturbation(
            self.category,
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
        )


class EnvPerturbationFactory(PerturbationFactory):
    def __init__(self, category, value_range):
        self.category = category
        self.value_range = value_range

    def spawn(self) -> EnvPerturbation:
        return EnvPerturbation(self.category, self._random(self.value_range))


class Relation(ABC):
    def __init__(self, field, weight, threshold):
        self.field = field
        self.weight = weight
        self.threshold = threshold

    @abstractmethod
    def is_violated(self, source, result) -> (bool, float):
        """Determine if this relation is violated.

        @param source: source result
        @param result: follow-up result
        @return: the float value denotes the extent to which this relation is violated
        """
        raise NotImplementedError

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.field == other.field and self.weight == other.weight and self.threshold == other.threshold
        else:
            return False


class Invariance(Relation):
    def __init__(self, field, weight=1.0, threshold=0.01):
        super().__init__(field, weight, threshold)

    def is_violated(self, source, result) -> (bool, float):
        diff = abs(result - source)
        return diff > self.threshold, diff


class Decreasing(Relation):
    def __init__(self, field, weight=-1.0, threshold=0.1):
        super().__init__(field, weight, threshold)

    def is_violated(self, source, result) -> (bool, float):
        diff = source - result
        return diff < source * self.threshold, diff


class Increasing(Relation):
    def __init__(self, field, weight=-1.0, threshold=0.1):
        super().__init__(field, weight, threshold)

    def is_violated(self, source, result) -> (bool, float):
        diff = result - source
        return diff < source * self.threshold, diff


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
