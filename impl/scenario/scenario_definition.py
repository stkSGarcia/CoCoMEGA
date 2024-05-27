import math
import random
import uuid
from abc import ABC
from copy import deepcopy
from enum import Enum

import numpy as np
from scipy.spatial.distance import cdist

from impl.config import CONFIG
from impl.scenario.LeaderboardFactory import LeaderBoardFactory

from impl.utils.trajectory import TrajectorySolver



class Boundary(dict):
    class Region(Enum):
        LEFT = CONFIG["boundary"]["region"]["left"]
        FOCUS = CONFIG["boundary"]["region"]["focus"]
        RIGHT = CONFIG["boundary"]["region"]["right"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Check type consistency and validity.
        for lower, upper in self.values():
            if type(lower) is not type(upper):
                raise ValueError(f"Unmatched boundary types: [{type(lower)}, {type(upper)}].")
            if not isinstance(lower, float) and not isinstance(lower, int):
                raise ValueError(f"Unsupported boundary type: {type(lower)}.")
            if lower > upper:
                raise ValueError(f"The lower boundary is greater than the upper boundary: {lower} > {upper}.")

        # Append region boundaries.
        dists, angles = zip(*[(region.value["radius"], region.value["angle"]) for region in Boundary.Region])
        self["radius"] = [np.min(dists), np.max(dists)]
        self["angle"] = [np.min(angles), np.max(angles)]

    def random(self, field: str, region: Region = None, none_pb=None):
        if field in ["radius", "angle"]:
            if region is None:
                region = random.choice(list(Boundary.Region))
            lower, upper = region.value[field] if hasattr(region, "value") else region[field]
        else:
            lower, upper = self[field]
        if isinstance(lower, float):
            return None if none_pb and random.random() < none_pb else random.uniform(lower, upper)
        elif isinstance(lower, int):
            return None if none_pb and random.random() < none_pb else random.randint(lower, upper)

    @staticmethod
    def get_region(angle):
        for region in Boundary.Region:
            lower, upper = region.value["angle"]
            if lower <= angle <= upper:
                return region
        return None


def _dist_attrs(this, that, attrs, boundary: Boundary):
    dist = 0
    for attr in attrs:
        lower, upper = boundary[attr]
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
        lower, upper = boundary[attr]
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
        scenario = cls._generate_empty_scenario()
        scenario.vehicles = ScenarioDefinition._generate_actors(Vehicle, CONFIG["scenario"]["init_pb"]["vehicle"])
        scenario.walkers = ScenarioDefinition._generate_actors(Walker, CONFIG["scenario"]["init_pb"]["walker"])
        scenario.statics = ScenarioDefinition._generate_actors(Static, CONFIG["scenario"]["init_pb"]["static"])
        return scenario

    @classmethod
    def generate_leaderboard_scenario(cls, scenario_type, **kwargs):
        scenario = cls._generate_empty_scenario()
        return LeaderBoardFactory.generate(scenario, scenario_type, **kwargs)

    @classmethod
    def generate_random_or_leaderboard(cls):
        if random.random() < CONFIG["scenario"]["leaderboard_pb"]:
            return cls.generate_leaderboard_scenario(scenario_type="random")
        else:
            return cls.generate_random()

    @classmethod
    def _generate_empty_scenario(cls):
        scenario = cls()
        scenario.ego_vehicle = Vehicle.generate_random()
        trajectory = random.choice(ScenarioDefinition._TRAJECTORY)
        scenario.town = trajectory["town"]
        scenario.trajectory = trajectory["trajectory"]
        for attr in ScenarioDefinition.ATTRIBUTES:
            setattr(scenario, attr, ScenarioDefinition._BOUNDARY.random(attr))
        return scenario

    @staticmethod
    def _generate_actors(cls, probability):
        actors = []
        times = 1
        while len(actors) < CONFIG["scenario"]["max_actors"] and random.random() < probability ** times:
            actors.append(cls.generate_random())
            times += 1
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
    def _random_pick_actor(actors, region: Boundary.Region = None):
        index, count = -1, 0
        for i, actor in enumerate(actors):
            if region is None or actor.region == region:
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
                dist += max(len(actors), len(other_actors)) * pow(CONFIG["scenario"]["dist_scaling"], 2)
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
        if random.random() < CONFIG["scenario"]["mut_del"]:
            times = 1
            while random.random() < CONFIG["scenario"]["mutpb"] ** times:
                actors = random.choice([self.vehicles, self.walkers, self.statics])
                index = ScenarioDefinition._random_pick_actor(actors)
                if index >= 0: del actors[index]
                times += 1
        else:
            for actors, cls in zip([self.vehicles, self.walkers, self.statics], [Vehicle, Walker, Static]):
                actors += ScenarioDefinition._generate_actors(cls, CONFIG["scenario"]["mut_add"])

    def build_actor_trajectory(self, actor_def):
        spawn_point = actor_def.get_config()['spawn_point']
        spawn_point['x'] += self.trajectory[0]['x']
        spawn_point['y'] += self.trajectory[0]['y']
        yaw_rad = math.radians(spawn_point['yaw'])

        total_distance = actor_def.speed * CONFIG['simulation']['scenario_duration'] \
            if hasattr(actor_def, "speed") else 0.5

        source = (
            spawn_point['x'],
            spawn_point['y'],
        )
        destination = (
            spawn_point['x'] + total_distance * math.cos(yaw_rad),
            spawn_point['y'] + total_distance * math.sin(yaw_rad)
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


class Actor(ABC):
    _ATTRIBUTES = ["radius", "angle", "yaw", "model"]
    _BLUEPRINTS = None
    _BOUNDARY = None

    def __init__(self, radius, angle, yaw, model, *args, **kwargs):
        self.radius = radius
        self.angle = angle
        self.yaw = yaw
        self.model = model
        self.region = None
        self.update_region()

    def update_region(self):
        self.region = self._BOUNDARY.get_region(self.angle)

    @classmethod
    def generate_random(cls, region: Boundary.Region = None, none_pb=None):
        return cls(**{attr: cls._BOUNDARY.random(attr, region, none_pb)
                      for attr in Actor._ATTRIBUTES + cls._ATTRIBUTES})

    def update(self, other):
        # if not isinstance(other, self.__class__):
        if str(type(self)) != str(type(other)):
            raise ValueError(f"Unmatched types: [{type(self)}, {type(other)}].")
        for attr in Actor._ATTRIBUTES + self._ATTRIBUTES:
            setattr(self, attr, getattr(other, attr))
        self.update_region()

    def dist(self, other):
        # if not isinstance(other, self.__class__):
        if str(type(self)) != str(type(other)):
            raise ValueError(f"Unmatched types: [{type(self)}, {type(other)}].")
        return _dist_attrs(self, other, Actor._ATTRIBUTES + self._ATTRIBUTES, self._BOUNDARY)

    def mate(self, other):
        """Mate actors in place."""
        # if not isinstance(other, self.__class__):
        if str(type(self)) != str(type(other)):
            raise ValueError(f"Unmatched types: [{type(self)}, {type(other)}].")
        _mate_attrs(self, other, Actor._ATTRIBUTES + self._ATTRIBUTES)
        self.update_region()
        other.update_region()

    def mutate(self):
        """Mutate actors in place."""
        _mutate_attrs(self, Actor._ATTRIBUTES + self._ATTRIBUTES, self._BOUNDARY)
        self.update_region()

    def get_config(self):
        return {
            "role_name": self.region.name.lower(),
            "spawn_point": {
                "x": self.radius * math.cos(math.radians(self.angle)),
                "y": self.radius * math.sin(math.radians(self.angle)),
                "z": 0.0,
                "yaw": self.yaw,
            },
            "model": self._BLUEPRINTS["model"][self.model]
        }

    def __eq__(self, other):
        # return (isinstance(other, self.__class__) and
        return (str(type(self)) == str(type(other)) and
                self.radius == other.radius and
                self.angle == other.angle and
                self.yaw == other.yaw and
                self.model == other.model)

    def __repr__(self):
        return (f"{self.__class__.__name__}(region={self.region}, " +
                ", ".join(f"{attr}={str(getattr(self, attr))}" for attr in Actor._ATTRIBUTES + self._ATTRIBUTES) + ")")


class Vehicle(Actor):
    _ATTRIBUTES = ["speed", "autopilot"]
    _BLUEPRINTS = CONFIG["blueprint"]["vehicle"]
    _BOUNDARY = Boundary(CONFIG["boundary"]["vehicle"])

    def __init__(self, radius, angle, yaw, model, speed, autopilot):
        super().__init__(radius, angle, yaw, model)
        self.speed = speed
        self.autopilot = autopilot

    def get_config(self):
        return {
            **super().get_config(),
            "speed": self.speed,
            "autopilot": bool(self.autopilot),
        }

    def __eq__(self, other):
        return super().__eq__(other) and self.speed == other.speed and self.autopilot == other.autopilot


class Walker(Actor):
    _ATTRIBUTES = ["speed"]
    _BLUEPRINTS = CONFIG["blueprint"]["walker"]
    _BOUNDARY = Boundary(CONFIG["boundary"]["walker"])

    def __init__(self, radius, angle, yaw, model, speed):
        super().__init__(radius, angle, yaw, model)
        self.speed = speed

    def get_config(self):
        return {
            **super().get_config(),
            "speed": self.speed,
        }

    def __eq__(self, other):
        return super().__eq__(other) and self.speed == other.speed


class Static(Actor):
    _ATTRIBUTES = []
    _BLUEPRINTS = CONFIG["blueprint"]["static"]
    _BOUNDARY = Boundary(CONFIG["boundary"]["static"])
