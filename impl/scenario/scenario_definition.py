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

    def get(self, field: str):
        return self.boundary[field]

    def random(self, field: str, none_pb=None):
        lower, upper = self.boundary[field]
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
    size = min(len(this), len(that))
    for i in range(size):
        if random.random() < CONFIG["scenario"]["cxpb"]:
            this[i], that[i] = that[i], this[i]
    return this, that


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
    _ATTRIBUTES = ["weather"]
    _DYNAMIC = ["vehicle", "walker", "static"]
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
        scenario.vehicles = [Vehicle.generate_random()]
        scenario.walkers = [Walker.generate_random()]
        scenario.statics = [Static.generate_random()]
        for attr in ScenarioDefinition._ATTRIBUTES:
            setattr(scenario, attr, ScenarioDefinition._BOUNDARY.random(attr))
        return scenario

    def get_trigger_position(self):
        return self.trajectory[0]

    def get_other_actors(self):
        return [actor.get_config() for actor in self.vehicles + self.walkers + self.statics]

    def update(self, category: str, value):
        if category in ScenarioDefinition._DYNAMIC:
            actors = getattr(self, f"{category}s")
            exist = False
            for actor in actors:
                if actor.id_ == value.id_:
                    exist = True
                    actor.update(value)
                    break
            if not exist:
                actors.append(deepcopy(value))
        elif category in ScenarioDefinition._ATTRIBUTES:
            setattr(self, category, value)
        else:
            raise ValueError(f"Unsupported category: {category}.")

    def dist(self, other):
        # if not isinstance(other, self.__class__):
        if str(type(self)) != str(type(other)):
            raise ValueError(f"Unmatched types: [{type(self)}, {type(other)}].")
        dist = _dist_attrs(self, other, ScenarioDefinition._ATTRIBUTES, ScenarioDefinition._BOUNDARY)
        for actors, other_actors in zip([self.vehicles, self.walkers, self.statics],
                                        [other.vehicles, other.walkers, other.statics]):
            dist_matrix = cdist(np.reshape(actors, (-1, 1)),
                                np.reshape(other_actors, (-1, 1)),
                                lambda x, y: x[0].dist(y[0]))
            dist += dist_matrix.min(axis=1 if len(actors) > len(other_actors) else 0).sum()
        return math.sqrt(dist)

    def mate(self, other):
        # if not isinstance(other, self.__class__):
        if str(type(self)) != str(type(other)):
            raise ValueError(f"Unmatched types: [{type(self)}, {type(other)}].")
        _mate_attrs(self, other, ScenarioDefinition._ATTRIBUTES)
        _mate_actors(self.vehicles, other.vehicles)
        _mate_actors(self.walkers, other.walkers)
        _mate_actors(self.statics, other.statics)

    def mutate(self):
        _mutate_attrs(self, ScenarioDefinition._ATTRIBUTES, ScenarioDefinition._BOUNDARY)
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
        self._update_position()

    def _update_position(self):
        lower, upper = self._BOUNDARY.get("y")
        interval = (upper - lower) / len(Boundary.REGION)
        for step in range(1, len(Boundary.REGION)):
            if self.transform.y < lower + step * interval:
                self.position = Boundary.REGION[step - 1]
                return
        self.position = Boundary.REGION[-1]

    @classmethod
    def generate_random(cls, boundary: Boundary = None, none_pb=None):
        if boundary is None:
            boundary = cls._BOUNDARY
        return cls(transform=Transform(**{attr: boundary.random(attr, none_pb) for attr in Transform.ATTRIBUTES}),
                   **{attr: boundary.random(attr, none_pb) for attr in cls._ATTRIBUTES})

    def update(self, other):
        for attr in Transform.ATTRIBUTES:
            setattr(self.transform, attr, getattr(other.transform, attr))
        self._update_position()
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
        self._update_position()
        other._update_position()
        _mate_attrs(self, other, self._ATTRIBUTES)

    def mutate(self):
        """Mutate actors in place."""
        _mutate_attrs(self.transform, Transform.ATTRIBUTES, self._BOUNDARY)
        self._update_position()
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
