import logging
import math
import random
import uuid
from abc import ABC
from copy import deepcopy
from enum import Enum

import carla
import numpy as np
from scipy.spatial.distance import cdist
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from impl.config import CONFIG
from impl.scenario.LeaderboardFactory import LeaderBoardFactory
from impl.scenario.carla_utils import get_junction_topology, filter_junction_wp_direction, transform_to_dict, \
    get_closest_wp
from impl.scenario.exceptions import InvalidScenarioDefinitionError
from impl.utils.trajectory import TrajectorySolver

logger = logging.getLogger(__name__)


class Boundary(dict):
    class Region(Enum):
        LEFT = CONFIG["boundary"]["region"]["left"]
        FOCUS = CONFIG["boundary"]["region"]["focus"]
        RIGHT = CONFIG["boundary"]["region"]["right"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Check type consistency and validity.
        self._check_type_consistency(self.values())

        # Append region boundaries.
        dists, angles = zip(*[(region.value["radius"], region.value["angle"]) for region in Boundary.Region])
        self["radius"] = [np.min(dists), np.max(dists)]
        self["angle"] = [np.min(angles), np.max(angles)]

    def random(self, field: str, region: Region = None, none_pb=None):
        if field in ["radius", "angle"]:
            if region is None:
                region = random.choice(list(Boundary.Region))
            lower, upper = region.value[field]
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

    def _check_type_consistency(self, values):
        for element in values:
            if isinstance(element, list):
                assert (len(element) == 2)
                lower, upper = element[0], element[1]
                if type(lower) is not type(upper):
                    raise ValueError(f"Unmatched boundary types: [{type(lower)}, {type(upper)}].")
                if not isinstance(lower, float) and not isinstance(lower, int):
                    raise ValueError(f"Unsupported boundary type: {type(lower)}.")
                if lower > upper:
                    raise ValueError(
                        f"The lower boundary is greater than the upper boundary: {lower} > {upper}.")
            elif isinstance(element, dict):
                self._check_type_consistency(element.values())
            else:
                raise ValueError(f"Unsupported type for boundary: '{type(element)}'.")


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
        x = getattr(this, attr)
        if isinstance(lower, float):
            if lower == upper: continue
            if lower <= x <= upper:  # Polynomial mutation
                delta_1 = (x - lower) / (upper - lower)
                delta_2 = (upper - x) / (upper - lower)
                rand = random.random()
                mut_pow = 1.0 / (CONFIG["scenario"]["mut_eta"] + 1.)

                if rand < 0.5:
                    xy = 1.0 - delta_1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (CONFIG["scenario"]["mut_eta"] + 1)
                    delta_q = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta_2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (CONFIG["scenario"]["mut_eta"] + 1)
                    delta_q = 1.0 - val ** mut_pow

                x = x + delta_q * (upper - lower)
                x = min(max(x, lower), upper)
            else:  # Gaussian mutation
                x = random.gauss(x, CONFIG["scenario"]["mut_std"])
            setattr(this, attr, x)
        elif isinstance(lower, int):
            assert lower <= x <= upper
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
    def _load_world(cls, town):
        if CarlaDataProvider.get_world() is None or CarlaDataProvider.get_world().get_map().name != town:
            world = CarlaDataProvider.get_client().load_world(town)
            CarlaDataProvider.set_world(world)

    @classmethod
    def generate_random(cls):
        scenario = cls._generate_empty_scenario()
        scenario.vehicles = Vehicle.generate_random_actors(CONFIG["scenario"]["init_pb"]["vehicle"])
        scenario.walkers = Walker.generate_random_actors(CONFIG["scenario"]["init_pb"]["walker"])
        scenario.statics = Static.generate_random_actors(CONFIG["scenario"]["init_pb"]["static"])
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
        trajectory_def = random.choice(ScenarioDefinition._TRAJECTORY).copy()
        scenario.town = trajectory_def["town"]
        trajectory_def["direction"] = random.choice(trajectory_def.get("direction", [None]))
        cls._load_world(scenario.town)
        trajectory_def["trajectory"] = cls._build_trajectory(trajectory_def)
        scenario.trajectory = trajectory_def

        for attr in ScenarioDefinition.ATTRIBUTES:
            setattr(scenario, attr, ScenarioDefinition._BOUNDARY.random(attr))
        return scenario

    @classmethod
    def _build_trajectory(cls, trajectory_def):
        trajectory = []
        location = carla.Location(x=trajectory_def["start"]["x"], y=trajectory_def["start"]["y"], z=0)
        waypoint = CarlaDataProvider.get_map().get_waypoint(location)
        trajectory.append(transform_to_dict(waypoint.transform))

        # Find the nearest junction
        while not waypoint.is_junction:
            waypoint = waypoint.next(1.0)[0]

        if trajectory_def["direction"] is None:
            trajectory.append(transform_to_dict(waypoint.transform))
            return trajectory

        trajectory.append(transform_to_dict(waypoint.transform))
        junction = waypoint.get_junction()
        _, exit_wps = get_junction_topology(junction)

        # Filter waypoints for the target lane direction
        direction_mapping = {
            'left': 'right',
            'right': 'left',
            'forward': 'ref',
        }
        target_exit_wps = filter_junction_wp_direction(waypoint, exit_wps,
                                                       direction_mapping[trajectory_def["direction"]])

        if not target_exit_wps:
            raise InvalidScenarioDefinitionError(f"No lane found in the '{trajectory_def['direction']}' direction!")

        target_wp = get_closest_wp(wp_list=target_exit_wps, reference_wp=waypoint)
        for i in range(5):
            trajectory.append(transform_to_dict(target_wp.transform))
            target_wp = target_wp.next(10)[0]

        return trajectory

    def get_trigger_position(self):
        return self.trajectory["start"]

    def get_other_actors(self):
        return [actor.get_config() for actor in self.vehicles + self.walkers + self.statics]

    def add_actor(self, category: str, new_actor):
        actors = getattr(self, f"{category}s")
        actors.append(deepcopy(new_actor))

    def remove_actor(self, category: str, region):
        actors = getattr(self, f"{category}s")
        index = ScenarioDefinition._random_pick_actor(actors, region)
        if index >= 0:
            del actors[index]

    def replace_actor(self, category: str, region, new_actor):
        actors = getattr(self, f"{category}s")
        index = ScenarioDefinition._random_pick_actor(actors, region)
        if index >= 0:
            actors[index].update(new_actor)

    def update_attribute(self, category: str, value):
        setattr(self, category, value)

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
        if not isinstance(other, self.__class__):
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
        if not isinstance(other, self.__class__):
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
                actors += cls.generate_random_actors(CONFIG["scenario"]["mut_add"])

    def build_actor_trajectory(self, actor_def):
        spawn_point = actor_def.get_config()['spawn_point']
        spawn_point['x'] += self.trajectory["start"]["x"]
        spawn_point['y'] += self.trajectory["start"]["y"]
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
            ego_traj = [(t["x"], t["y"]) for t in self.trajectory["trajectory"]]
            if TrajectorySolver.solve(ego_traj, actor_traj):
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

    @classmethod
    def generate_random_actors(cls, probability):
        actors = []
        times = 1
        while len(actors) < CONFIG["scenario"]["max_actors"] and random.random() < probability ** times:
            actors.append(cls.generate_random())
            times += 1
        return actors

    def update(self, other):
        if not isinstance(other, self.__class__):
            raise ValueError(f"Unmatched types: [{type(self)}, {type(other)}].")
        for attr in Actor._ATTRIBUTES + self._ATTRIBUTES:
            setattr(self, attr, getattr(other, attr))
        self.update_region()

    def dist(self, other):
        if not isinstance(other, self.__class__):
            raise ValueError(f"Unmatched types: [{type(self)}, {type(other)}].")
        return _dist_attrs(self, other, Actor._ATTRIBUTES + self._ATTRIBUTES, self._BOUNDARY)

    def mate(self, other):
        """Mate actors in place."""
        if not isinstance(other, self.__class__):
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
            "role_name": self.region.name.lower() if self.region else "others",
            "spawn_point": {
                "x": self.radius * math.cos(math.radians(self.angle)),
                "y": self.radius * math.sin(math.radians(self.angle)),
                "z": 0.0,
                "yaw": self.yaw,
            },
            "model": self._BLUEPRINTS["model"][self.model]
        }

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                self.radius == other.radius and
                self.angle == other.angle and
                self.yaw == other.yaw and
                self.model == other.model)

    def __repr__(self):
        return (f"{self.__class__.__name__}(region={self.region}, " +
                ", ".join(f"{attr}={str(getattr(self, attr))}" for attr in Actor._ATTRIBUTES + self._ATTRIBUTES) + ")")


class Vehicle(Actor):
    _ATTRIBUTES = ["speed"]
    _BLUEPRINTS = CONFIG["blueprint"]["vehicle"]
    _BOUNDARY = Boundary(CONFIG["boundary"]["vehicle"])

    def __init__(self, radius, angle, yaw, model, speed):
        super().__init__(radius, angle, yaw, model)
        self.speed = speed
        self.autopilot = True

    def get_config(self):
        return {
            **super().get_config(),
            "speed": self.speed,
            "autopilot": self.autopilot,
        }

    @classmethod
    def generate_random(cls, region: Boundary.Region = None, none_pb=None, **filters):
        vehicle = super().generate_random(region, none_pb)
        if "base_model" in filters:
            if isinstance(filters["base_model"], list):
                weights = [cls._BOUNDARY["base_model"][bm][1] - cls._BOUNDARY["base_model"][bm][0] + 1 for bm in
                           filters["base_model"]]
                base_model = random.choices(filters["base_model"], weights=weights, k=1)[0]
            else:
                base_model = filters["base_model"]
            lower, upper = cls._BOUNDARY["base_model"][base_model]
            vehicle.model = random.randint(lower, upper)
        return vehicle

    def __eq__(self, other):
        return super().__eq__(other) and self.speed == other.speed


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
