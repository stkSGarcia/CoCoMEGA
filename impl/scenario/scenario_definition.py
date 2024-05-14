import math
import random
import uuid
from abc import ABC
from copy import deepcopy
from typing import Dict, List

import numpy as np
from scipy.spatial.distance import cdist

from impl.config import CONFIG
from impl.utils.trajectory import TrajectorySolver


class Boundary:
    REGION = ["left", "focus", "right"]
    _REGION_KEY = "y"

    def __init__(self, boundary: Dict[str, List]):
        self.boundary = boundary

        # Check type consistency and validity.
        for lower, upper in self.boundary.values():
            if type(lower) is not type(upper):
                raise ValueError(f"Unmatched boundary types: [{type(lower)}, {type(upper)}].")
            if not isinstance(lower, float) and not isinstance(lower, int):
                raise ValueError(f"Unsupported boundary type: {type(lower)}.")
            if lower > upper:
                raise ValueError(f"The lower boundary is greater than the upper boundary: {lower} > {upper}.")

        self.dividers = None
        if Boundary._REGION_KEY in self.boundary:
            lower, upper = self.boundary[Boundary._REGION_KEY]
            interval = (upper - lower) / len(Boundary.REGION)
            self.dividers = [lower + interval * i for i in range(1, len(Boundary.REGION))]

    def get(self, field: str):
        return self.boundary[field]

    def get_region(self, value):
        if self.dividers is None: return None
        for i, divider in enumerate(self.dividers):
            if value < divider: return Boundary.REGION[i]
        return Boundary.REGION[-1]

    def random(self, field: str, region=None, none_pb=None):
        lower, upper = self.boundary[field]
        if field == Boundary._REGION_KEY and region is not None:
            i = Boundary.REGION.index(region)
            if i > 0:
                lower = self.dividers[i - 1]
            if i < len(self.dividers):
                upper = self.dividers[i]
        if isinstance(lower, float):
            return None if none_pb and random.random() < none_pb else random.uniform(lower, upper)
        elif isinstance(lower, int):
            return None if none_pb and random.random() < none_pb else random.randint(lower, upper)


def _dist_attrs(this, that, attrs, boundary: Boundary):
    dist = 0
    for attr in attrs:
        lower, upper = boundary.get(attr)
        if isinstance(lower, float):
            dist += pow(abs(getattr(this, attr) - getattr(that, attr)) / (upper - lower), 2) if upper != lower else 0
        elif isinstance(lower, int):
            # TODO: within the same category.
            dist += pow(CONFIG["scenario"]["dist_scaling"] *
                        (0.0 if getattr(this, attr) == getattr(that, attr) else 1.0), 2)
    return dist


def _mate_attrs(this, that, attrs):
    for attr in attrs:
        if random.random() < CONFIG["scenario"]["cxpb"]:
            value = getattr(this, attr)
            setattr(this, attr, getattr(that, attr))
            setattr(that, attr, value)


def _mate_actors(this, that):
    common = min(len(this), len(that))
    for i in range(common):
        if random.random() < CONFIG["scenario"]["cxpb"]:
            this[i], that[i] = that[i], this[i]
    less, more = (this, that) if len(this) < len(that) else (that, this)
    while len(more) > common:
        if random.random() < CONFIG["scenario"]["cxpb"]:
            less.append(more.pop(common))
        else:
            common += 1


def _mutate_attrs(this, attrs, boundary: Boundary):
    for attr in attrs:
        if random.random() >= CONFIG["scenario"]["mutpb"]: continue
        lower, upper = boundary.get(attr)
        if isinstance(lower, float) and lower != upper:  # Polynomial mutation
            x = getattr(this, attr)
            delta_1 = (x - lower) / (upper - lower)
            delta_2 = (upper - x) / (upper - lower)
            rand = random.random()
            mut_pow = 1.0 / (CONFIG["scenario"]["eta"] + 1.)

            if rand < 0.5:
                xy = 1.0 - delta_1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (CONFIG["scenario"]["eta"] + 1)
                delta_q = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta_2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (CONFIG["scenario"]["eta"] + 1)
                delta_q = 1.0 - val ** mut_pow

            x = x + delta_q * (upper - lower)
            x = min(max(x, lower), upper)
            setattr(this, attr, x)
        elif isinstance(lower, int):
            setattr(this, attr, random.randint(lower, upper))


class ScenarioDefinition:
    ATTRIBUTES = ["weather"]
    DYNAMIC = ["vehicle", "walker", "static"]
    _BLUEPRINTS = CONFIG["blueprint"]["scenario"]
    _BOUNDARY = Boundary(CONFIG["boundary"]["env"])
    _TRAJECTORY = CONFIG["trajectory"]

    def __new__(cls, *args):
        if len(args) == 1 and isinstance(args[0], cls): return args[0]
        self = super().__new__(cls)
        self.id_ = uuid.uuid4().hex
        self.town = None
        self.ego_vehicle = None
        self.trajectory = []
        self.vehicles = []
        self.walkers = []
        self.statics = []
        self.weather = None
        return self

    def assign_new_id(self):
        self.id_ = uuid.uuid4().hex

    @classmethod
    def generate_random(cls):
        scenario = cls()
        scenario.ego_vehicle = Vehicle.generate_random()
        trajectory = random.choice(ScenarioDefinition._TRAJECTORY)
        scenario.town = trajectory["town"]
        scenario.trajectory = trajectory["trajectory"]
        scenario.vehicles = ScenarioDefinition._generate_actors(Vehicle, CONFIG["scenario"]["init_pb"]["vehicle"])
        scenario.walkers = ScenarioDefinition._generate_actors(Walker, CONFIG["scenario"]["init_pb"]["walker"])
        scenario.statics = ScenarioDefinition._generate_actors(Static, CONFIG["scenario"]["init_pb"]["static"])
        for attr in ScenarioDefinition.ATTRIBUTES:
            setattr(scenario, attr, ScenarioDefinition._BOUNDARY.random(attr))
        return scenario

    @staticmethod
    def _generate_actors(cls, probability):
        actors = []
        times = 1
        while random.random() < probability ** times:
            actors.append(cls.generate_random())
        return actors

    def get_trigger_position(self):
        return self.trajectory[0]

    def get_other_actors(self):
        return [actor.get_config() for actor in self.vehicles + self.walkers + self.statics]

    def update(self, category: str, operation, value):
        from impl.mr.mr import Operation
        if category in ScenarioDefinition.DYNAMIC:
            actors = getattr(self, f"{category}s")
            if operation == Operation.ADD:
                actors.append(deepcopy(value))
            elif operation == Operation.REMOVE:
                index = ScenarioDefinition._random_pick_actor(actors, value)
                if index >= 0:
                    del actors[index]
            elif operation == Operation.REPLACE:
                index = ScenarioDefinition._random_pick_actor(actors, value[0])
                if index >= 0:
                    actors[index].update(value[1])
            else:
                raise ValueError(f"Unsupported operation: {operation}.")
        elif category in ScenarioDefinition.ATTRIBUTES:
            setattr(self, category, value)
        else:
            raise ValueError(f"Unsupported category: {category}.")

    @staticmethod
    def _random_pick_actor(actors, region):
        index, count = -1, 0
        for i, actor in enumerate(actors):
            if region is None or actor.position == region:
                count += 1
                if random.randint(1, count) == 1:
                    index = i
        return index

    def dist(self, other):
        # if not isinstance(other, self.__class__):
        if str(type(self)) != str(type(other)):
            raise ValueError(f"Unmatched types: [{type(self)}, {type(other)}].")
        dist = _dist_attrs(self, other, ScenarioDefinition.ATTRIBUTES, ScenarioDefinition._BOUNDARY)
        for actors, other_actors in zip([self.vehicles, self.walkers, self.statics],
                                        [other.vehicles, other.walkers, other.statics]):
            if len(actors) == 0 and len(other_actors) == 0:
                continue
            elif len(actors) == 0 or len(other_actors) == 0:
                dist += min(len(actors), len(other_actors)) * pow(CONFIG["scenario"]["dist_scaling"], 2)
            else:
                dist_matrix = cdist(np.array(actors, dtype=object).reshape((-1, 1)),
                                    np.array(other_actors, dtype=object).reshape((-1, 1)),
                                    lambda x, y: x[0].dist(y[0]))
                dist += dist_matrix.min(axis=1 if len(actors) > len(other_actors) else 0).sum()
        return math.sqrt(dist)

    def mate(self, other):
        # if not isinstance(other, self.__class__):
        if str(type(self)) != str(type(other)):
            raise ValueError(f"Unmatched types: [{type(self)}, {type(other)}].")
        _mate_attrs(self, other, ScenarioDefinition.ATTRIBUTES)
        _mate_actors(self.vehicles, other.vehicles)
        _mate_actors(self.walkers, other.walkers)
        _mate_actors(self.statics, other.statics)

    def mutate(self):
        _mutate_attrs(self, ScenarioDefinition.ATTRIBUTES, ScenarioDefinition._BOUNDARY)
        for actor in self.vehicles + self.walkers + self.statics:
            actor.mutate()

    def build_actor_trajectory(self, actor_def):
        spawn_point = actor_def.transform
        yaw_rad = math.radians(spawn_point.yaw)

        if hasattr(actor_def, "speed"):
            total_distance = actor_def.speed * CONFIG['simulation']['scenario_duration']
            source = (
                spawn_point.x,
                spawn_point.y
            )
            destination = (
                spawn_point.x + total_distance * math.cos(yaw_rad),
                spawn_point.y + total_distance * math.sin(yaw_rad)
            )
        else:
            eps = 0.5
            source = (
                spawn_point.x - eps,
                spawn_point.y - eps
            )
            destination = (
                spawn_point.x + eps,
                spawn_point.y + eps,
            )
        return (source, destination)

    def trajectory_collision_score(self):
        score = 0
        for actor in self.vehicles + self.walkers + self.statics:
            actor_traj = self.build_actor_trajectory(actor)
            if TrajectorySolver.solve([(p['x'], p['y']) for p in self.trajectory], actor_traj):
                score += 1
        return score

    @staticmethod
    def _list_eq(this, that):
        if len(this) != len(that): return False
        copy = list(this)
        try:
            for elem in that:
                copy.remove(elem)
        except ValueError:
            return False
        return not copy

    def __eq__(self, other):
        # return (isinstance(other, self.__class__) and
        return (str(type(self)) == str(type(other)) and
                self.town == other.town and
                self.ego_vehicle == other.ego_vehicle and
                self.trajectory == other.trajectory and
                self.weather == other.weather and
                self._list_eq(self.vehicles, other.vehicles) and
                self._list_eq(self.walkers, other.walkers) and
                self._list_eq(self.statics, other.statics))

    def __repr__(self):
        return (f"Scenario(id={self.id_}, "
                f"town={self.town}, "
                f"ego_vehicle={self.ego_vehicle}, "
                f"trajectory={self.trajectory}, "
                f"vehicles={self.vehicles}, "
                f"walkers={self.walkers}, "
                f"statics={self.statics}, "
                f"weather={self.weather})")


class Transform:
    ATTRIBUTES = ["x", "y", "z", "pitch", "yaw", "roll"]

    def __init__(self, x, y, z, pitch, yaw, roll):
        self.x = x
        self.y = y
        self.z = z
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll

    def get_config(self):
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "pitch": self.pitch,
            "yaw": self.yaw,
            "roll": self.roll,
        }

    def __eq__(self, other):
        # return (isinstance(other, self.__class__) and
        return (str(type(self)) == str(type(other)) and
                self.x == other.x and
                self.y == other.y and
                self.z == other.z and
                self.pitch == other.pitch and
                self.yaw == other.yaw and
                self.roll == other.roll)

    def __repr__(self):
        return f"Transform(" + ", ".join(f"{attr}={str(getattr(self, attr))}" for attr in self.ATTRIBUTES) + ")"


class Actor(ABC):
    _ATTRIBUTES = []
    _BOUNDARY = None

    def __init__(self, transform: Transform, *args, **kwargs):
        self.transform = transform
        self.position = None
        self.update_position()

    def update_position(self):
        self.position = self._BOUNDARY.get_region(self.transform.y)

    @classmethod
    def generate_random(cls, region=None, none_pb=None):
        return cls(
            transform=Transform(**{attr: cls._BOUNDARY.random(attr, region, none_pb) for attr in Transform.ATTRIBUTES}),
            **{attr: cls._BOUNDARY.random(attr, none_pb) for attr in cls._ATTRIBUTES}
        )

    def update(self, other):
        for attr in Transform.ATTRIBUTES:
            setattr(self.transform, attr, getattr(other.transform, attr))
        self.update_position()
        for attr in self._ATTRIBUTES:
            setattr(self, attr, getattr(other, attr))

    def dist(self, other):
        # if not isinstance(other, self.__class__):
        if str(type(self)) != str(type(other)):
            raise ValueError(f"Unmatched types: [{type(self)}, {type(other)}].")
        return (_dist_attrs(self.transform, other.transform, Transform.ATTRIBUTES, self._BOUNDARY) +
                _dist_attrs(self, other, self._ATTRIBUTES, self._BOUNDARY))

    def mate(self, other):
        """Mate actors in place."""
        # if not isinstance(other, self.__class__):
        if str(type(self)) != str(type(other)):
            raise ValueError(f"Unmatched types: [{type(self)}, {type(other)}].")
        _mate_attrs(self.transform, other.transform, Transform.ATTRIBUTES)
        self.update_position()
        other.update_position()
        _mate_attrs(self, other, self._ATTRIBUTES)

    def mutate(self):
        """Mutate actors in place."""
        _mutate_attrs(self.transform, Transform.ATTRIBUTES, self._BOUNDARY)
        self.update_position()
        _mutate_attrs(self, self._ATTRIBUTES, self._BOUNDARY)

    def get_config(self):
        return {"spawn_point": self.transform.get_config()}

    def __eq__(self, other):
        # return (isinstance(other, self.__class__) and
        return (str(type(self)) == str(type(other)) and
                self.transform == other.transform)

    def __repr__(self):
        return (f"{self.__class__.__name__}(transform={self.transform}, position={self.position}, " +
                ", ".join(f"{attr}={str(getattr(self, attr))}" for attr in self._ATTRIBUTES) + ")")


class Vehicle(Actor):
    _ATTRIBUTES = ["speed", "model", "autopilot"]
    _BLUEPRINTS = CONFIG["blueprint"]["vehicle"]
    _BOUNDARY = Boundary(CONFIG["boundary"]["vehicle"])

    def __init__(self, transform, speed, model, autopilot):
        super().__init__(transform)
        self.speed = speed
        self.model = model
        self.autopilot = autopilot

    def get_config(self):
        return {
            **super().get_config(),
            "speed": self.speed,
            "model": Vehicle._BLUEPRINTS["model"][self.model],
            "autopilot": bool(self.autopilot),
        }

    def __eq__(self, other):
        return (super().__eq__(other) and
                self.speed == other.speed and
                self.model == other.model and
                self.autopilot == other.autopilot)


class Walker(Actor):
    _ATTRIBUTES = ["speed", "model"]
    _BLUEPRINTS = CONFIG["blueprint"]["walker"]
    _BOUNDARY = Boundary(CONFIG["boundary"]["walker"])

    def __init__(self, transform, speed, model):
        super().__init__(transform)
        self.speed = speed
        self.model = model

    def get_config(self):
        return {
            **super().get_config(),
            "speed": self.speed,
            "model": Walker._BLUEPRINTS["model"][self.model],
        }

    def __eq__(self, other):
        return (super().__eq__(other) and
                self.speed == other.speed and
                self.model == other.model)


class Static(Actor):
    _ATTRIBUTES = ["model"]
    _BLUEPRINTS = CONFIG["blueprint"]["static"]
    _BOUNDARY = Boundary(CONFIG["boundary"]["static"])

    def __init__(self, transform, model):
        super().__init__(transform)
        self.model = model

    def get_config(self):
        return {
            **super().get_config(),
            "model": Static._BLUEPRINTS["model"][self.model],
        }

    def __eq__(self, other):
        return (super().__eq__(other) and
                self.model == other.model)
