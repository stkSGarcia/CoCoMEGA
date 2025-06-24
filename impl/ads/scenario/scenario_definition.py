import logging
import math
import random
import sys
from abc import ABC
from copy import deepcopy
from enum import Enum
from itertools import groupby

import carla
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from impl import config as cfg
from impl.ads.evaluation.exceptions import InvalidScenarioDefinitionError
from impl.ads.scenario.Leaderboard_factory import LeaderBoardFactory
from impl.ads.utils.carla_utils import load_world, trajectory_interpolation, get_available_directions, get_junction, \
    location_to_dict, dict_to_location, group_junction_directions, get_closest_wp
from impl.ads.utils.trajectory import rotate_vector, single_trajectory_score
from impl.core.scenario.base_scenario import AbstractScenarioDefinition

logger = logging.getLogger(__name__)


class Boundary(dict):
    """Boundary definition for actor placement and parameters."""

    class Region(Enum):
        LEFT = cfg.CONFIG["boundary"]["region"]["left"]
        FOCUS = cfg.CONFIG["boundary"]["region"]["focus"]
        RIGHT = cfg.CONFIG["boundary"]["region"]["right"]

        def __repr__(self):
            return self.name

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Check type consistency and validity.
        self._check_type_consistency(self.values())

        # Append region boundaries.
        dists, angles = zip(*[(region.value["radius"], region.value["angle"]) for region in Boundary.Region])
        self["radius"] = [np.min(dists), np.max(dists)]
        self["angle"] = [np.min(angles), np.max(angles)]

    def random(self, field: str, region: Region = None, none_pb=None):
        """Randomly sample a value within the boundary."""
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
        return None

    @staticmethod
    def get_region(angle):
        """Return the region associated with a given angle."""
        for region in Boundary.Region:
            lower, upper = region.value["angle"]
            if lower <= angle <= upper:
                return region
        return None

    def _check_type_consistency(self, values):
        """Validate the consistency of boundary type values."""
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


def _dist_attrs(this, that, attrs, boundary: Boundary, scaling):
    """Calculate the distance between global attributes of two given scenarios."""
    dist = 0
    for attr in attrs:
        lower, upper = boundary[attr]
        if isinstance(lower, float):
            dist += pow(abs(getattr(this, attr) - getattr(that, attr)) / (upper - lower), 2) if upper != lower else 0
        elif isinstance(lower, int):
            if hasattr(this, attr) and hasattr(that, attr):
                dist += pow(scaling * (0.0 if getattr(this, attr) == getattr(that, attr) else 1.0), 2)
        else:
            raise ValueError(f"Unsupported boundaries: [{lower}, {upper}].")
    return dist


def _mate_attrs(this, that, attrs, cxpb):
    """Perform uniform crossover on the global attributes of two given scenarios."""
    for attr in attrs:
        if random.random() < cxpb:
            value = getattr(this, attr)
            setattr(this, attr, getattr(that, attr))
            setattr(that, attr, value)


def _mate_actors(this, that, cxpb):
    """Perform uniform crossover on the actors of two given scenarios."""
    common = min(len(this), len(that))
    for i in range(common):
        if random.random() < cxpb:
            this[i], that[i] = that[i], this[i]
    less, more = (this, that) if len(this) < len(that) else (that, this)
    while len(more) > common:
        if random.random() < cxpb:
            less.append(more.pop(common))
        else:
            common += 1


def _mutate_attrs(this, attrs, boundary: Boundary, mutpb, eta, std):
    """Perform mutation on the global attributes of a given scenario."""
    for attr in attrs:
        if random.random() >= mutpb: continue
        lower, upper = boundary[attr]
        x = getattr(this, attr)
        if isinstance(lower, float):
            if lower == upper: continue
            if lower <= x <= upper:  # Polynomial mutation.
                delta_1 = (x - lower) / (upper - lower)
                delta_2 = (upper - x) / (upper - lower)
                rand = random.random()
                mut_pow = 1.0 / (eta + 1.)

                if rand < 0.5:
                    xy = 1.0 - delta_1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta + 1)
                    delta_q = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta_2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta + 1)
                    delta_q = 1.0 - val ** mut_pow

                x = x + delta_q * (upper - lower)
                x = min(max(x, lower), upper)
            else:  # Gaussian mutation.
                x = random.gauss(x, std)
            setattr(this, attr, x)
        elif isinstance(lower, int):
            assert lower <= x <= upper
            setattr(this, attr, random.randint(lower, upper))


class ScenarioDefinition(AbstractScenarioDefinition):
    """Defines a scenario for simulation including ego, actors, weather, and trajectory."""

    ATTRIBUTES = ["weather", "brightness", "stop_sign_est", "red_light_est", "is_junction_est"]
    DYNAMIC = ["vehicle", "walker", "static"]
    _BLUEPRINTS = cfg.CONFIG["blueprint"]["scenario"]
    _BOUNDARY = Boundary(cfg.CONFIG["boundary"]["env"])
    _TRAJECTORY = cfg.CONFIG["trajectory"]["predefined"]

    def __init__(self, instance=None):
        super().__init__(instance)
        if instance is None:
            self.town = None
            self.ego_vehicle = None
            self.trajectory = []
            self.vehicles = []
            self.walkers = []
            self.statics = []
            self.weather = None
            self.brightness = None
            self.stop_sign_est = None
            self.red_light_est = None
            self.is_junction_est = None
        elif isinstance(instance, ScenarioDefinition):
            self.town = instance.town
            self.ego_vehicle = instance.ego_vehicle
            self.trajectory = instance.trajectory
            self.vehicles = instance.vehicles
            self.walkers = instance.walkers
            self.statics = instance.statics
            self.weather = instance.weather
            self.brightness = instance.brightness
            self.stop_sign_est = instance.stop_sign_est
            self.red_light_est = instance.red_light_est
            self.is_junction_est = instance.is_junction_est

    def dist(self, other, **kwargs):
        """
        Compute a distance measure between two scenarios.
        The distance accounts for both scenario attributes and actor configurations.

        :param other: Another :class:`ScenarioDefinition` instance to compare against.
        :return: Computed distance between two scenarios.
        :raises ValueError: If `other` is not a :class:`ScenarioDefinition`.
        """
        scaling = kwargs.get("scaling", cfg.CONFIG["scenario"]["dist_scaling"])
        if str(type(other)) != str(type(self)):
            raise ValueError(f"Unmatched types: [{type(self)}, {type(other)}].")
        dist = _dist_attrs(self, other, ScenarioDefinition.ATTRIBUTES, ScenarioDefinition._BOUNDARY, scaling=scaling)
        for actors, other_actors in zip([self.vehicles, self.walkers, self.statics],
                                        [other.vehicles, other.walkers, other.statics]):
            if len(actors) == 0 and len(other_actors) == 0:
                continue
            elif len(actors) == 0 or len(other_actors) == 0:
                dist += max(len(actors), len(other_actors)) * pow(cfg.CONFIG["scenario"]["dist_scaling"], 2)
            else:
                dist_matrix = cdist(np.array(actors, dtype=object).reshape((-1, 1)),
                                    np.array(other_actors, dtype=object).reshape((-1, 1)),
                                    lambda x, y: x[0].dist(y[0]))
                dist += np.power(dist_matrix.min(axis=1 if len(actors) > len(other_actors) else 0), 2).sum()
        return math.sqrt(dist)

    def mate(self, other, cxpb=cfg.CONFIG["scenario"]["cxpb"], **kwargs):
        """
        Apply uniform crossover on two scenarios by swapping their attributes and actors.

        :param other: Another :class:`ScenarioDefinition` instance to mate with.
        :param cxpb: Crossover probability (default from config).
        :raises ValueError: If `other` is not a :class:`ScenarioDefinition`.
        """
        if not isinstance(other, self.__class__):
            raise ValueError(f"Unmatched types: [{type(self)}, {type(other)}].")
        _mate_attrs(self, other, ScenarioDefinition.ATTRIBUTES, cxpb=cxpb)
        _mate_actors(self.vehicles, other.vehicles, cxpb=cxpb)
        _mate_actors(self.walkers, other.walkers, cxpb=cxpb)
        _mate_actors(self.statics, other.statics, cxpb=cxpb)

    def mutate(self, mutpb=cfg.CONFIG["scenario"]["mutpb"], **kwargs):
        """
        Mutate the scenario's attributes and actors. Mutation can also randomly add or delete actors.

        :param mutpb: Probability of mutation per attribute (default from config).
        :param eta: Distribution index for polynomial mutation.
        :param std: Standard deviation for Gaussian mutation.
        """
        eta = kwargs.get("eta", cfg.CONFIG["scenario"]["mut_eta"])
        std = kwargs.get("std", cfg.CONFIG["scenario"]["mut_std"])
        _mutate_attrs(self, ScenarioDefinition.ATTRIBUTES, ScenarioDefinition._BOUNDARY, mutpb=mutpb, eta=eta, std=std)
        for actor in self.vehicles + self.walkers + self.statics:
            actor.mutate(mutpb=mutpb, eta=eta, std=std)
        if random.random() < cfg.CONFIG["scenario"]["mut_del"]:
            times = 1
            while random.random() < cfg.CONFIG["scenario"]["mutpb"] ** times:
                actors = random.choice([self.vehicles, self.walkers, self.statics])
                index = ScenarioDefinition._random_pick_actor(actors)
                if index >= 0: del actors[index]
                times += 1
        else:
            for actors, cls in zip([self.vehicles, self.walkers, self.statics], [Vehicle, Walker, Static]):
                limit = cfg.CONFIG["scenario"]["max_actors"] - len(actors)
                if limit > 0: actors += cls.generate_random_actors(cfg.CONFIG["scenario"]["mut_add"], limit)

    @classmethod
    def generate_random(cls, predefined_trajectory=False):
        """Generate a random scenario.

        :param predefined_trajectory: Determines whether to select a predefined trajectory or generate a random one.
        :return: The generated scenario.
        """
        scenario = cls._generate_empty_scenario()
        if predefined_trajectory:
            scenario.set_trajectory(cls._random_predefined_trajectory())
        else:
            scenario.set_trajectory(cls._random_trajectory())
        scenario.vehicles = Vehicle.generate_random_actors(cfg.CONFIG["scenario"]["init_pb"]["vehicle"])
        scenario.walkers = Walker.generate_random_actors(cfg.CONFIG["scenario"]["init_pb"]["walker"])
        scenario.statics = Static.generate_random_actors(cfg.CONFIG["scenario"]["init_pb"]["static"])
        return scenario

    @classmethod
    def generate_leaderboard_scenario(cls, scenario_type, **kwargs):
        """
        Generate a scenario based on a predefined leaderboard style.

        :param scenario_type: Type of leaderboard scenario (e.g., :data:`crossing_negotiation`, :data:`pedestrian_emerging`).
        :type scenario_type: str
        :param kwargs: Additional parameters passed to the :class:`LeaderBoardFactory`.
        :return: A leaderboard scenario instance.
        """
        scenario = cls._generate_empty_scenario()
        scenario.set_trajectory(cls._random_predefined_trajectory())

        return LeaderBoardFactory.generate(scenario, scenario_type, **kwargs)

    @classmethod
    def generate_random_or_leaderboard(cls):
        """
        Randomly decide whether to generate a random scenario or a leaderboard scenario.

        :return: A randomly generated or leaderboard scenario instance.
        """
        if random.random() < cfg.CONFIG["scenario"]["leaderboard_pb"]:
            return cls.generate_leaderboard_scenario(scenario_type="random")
        else:
            return cls.generate_random()

    @classmethod
    def generate_random_with_marked_actors(cls):
        """
        Generate a random scenario with a marked actor in front of the ego vehicle.

        :return: The generated scenario.
        """
        scenario = cls.generate_random()
        category = random.choice(("vehicle", "walker"))
        actors = getattr(scenario, f"{category}s")
        actor_cls = getattr(sys.modules[__name__], category.capitalize())
        marked_actor = actor_cls.generate_random(region=Boundary.Region.FOCUS)
        marked_actor.mark = True
        actors.append(marked_actor)
        return scenario

    @classmethod
    def _generate_empty_scenario(cls):
        """
        Create an empty scenario template including an ego vehicle, its trajectory, and basic environment settings.

        :return: An empty :class:`ScenarioDefinition` instance with initialized fields.
        """
        scenario = cls()
        scenario.ego_vehicle = Vehicle.generate_random()
        scenario.ego_vehicle.angle = 0
        scenario.ego_vehicle.radius = 0
        scenario.ego_vehicle.region = None
        scenario.stop_sign_est = 0
        scenario.red_light_est = 0
        scenario.is_junction_est = 0

        for attr in ScenarioDefinition.ATTRIBUTES:
            setattr(scenario, attr, ScenarioDefinition._BOUNDARY.random(attr))
        return scenario

    @classmethod
    def _random_predefined_trajectory(cls):
        """
        Select a random trajectory definition from the list of predefined trajectories given in `config.yaml`.

        :return: The selected trajectory definition.
        """
        trajectory_def = random.choice(cls._TRAJECTORY).copy()
        trajectory_def["direction"] = random.choice(trajectory_def.get("direction", [None]))
        return trajectory_def

    @classmethod
    def _random_trajectory(cls):
        """
        Generate a random trajectory definition.

        :return: The generated trajectory definition.
        """
        while True:
            town = random.choice(cfg.CONFIG["trajectory"]["towns"])
            load_world(town)
            initial_transform = CarlaDataProvider._rng.choice(CarlaDataProvider._spawn_points)
            wp = CarlaDataProvider.get_map().get_waypoint(initial_transform.location)
            if wp.is_junction:
                continue
            start_transform = initial_transform
            start_location = start_transform.location
            start_rotation = start_transform.rotation
            initial_speed = random.uniform(0, cfg.CONFIG["trajectory"]["initial_speed_limit"])
            direction, exit_waypoint = random.choice(get_available_directions(
                start_transform,
                distance_limit=cfg.CONFIG["trajectory"]["junction_distance_limit"]
            ))
            junction_exit = location_to_dict(exit_waypoint.transform.location) if exit_waypoint is not None else None
            trajectory_def = {
                "town": town,
                "start": {
                    "x": start_location.x,
                    "y": start_location.y,
                    "z": start_location.z,
                    "yaw": start_rotation.yaw,
                    "speed": initial_speed,
                },
                "direction": direction,
                "junction_exit": junction_exit,
            }
            return trajectory_def

    @classmethod
    def _build_trajectory(cls, trajectory_def):
        """
        Construct a complete trajectory for the ego vehicle, based on the starting point and direction.

        :param trajectory_def: Dictionary containing `start`, `direction`, and optionally `junction_exit`.
        :return: A tuple of trajectory waypoints, GPS route, CARLA route, and junction status.
        :raises InvalidScenarioDefinitionError: If the trajectory direction is not valid given the starting location.
        """
        initial_location = carla.Location(x=trajectory_def["start"]["x"], y=trajectory_def["start"]["y"],
                                          z=trajectory_def["start"]["z"])
        trajectory, junction = get_junction(initial_location,
                                            distance_limit=cfg.CONFIG["trajectory"]["junction_distance_limit"])
        is_junction = (junction is not None)
        if trajectory_def["direction"] is None:
            waypoint = CarlaDataProvider.get_map().get_waypoint(trajectory[-1].location)
            for i in range(cfg.CONFIG["trajectory"]["junction_distance_limit"]):
                waypoint = waypoint.next(1)[0]
                trajectory.append(waypoint.transform)
        else:
            if junction is None:
                if trajectory_def["direction"] != "forward":
                    raise InvalidScenarioDefinitionError(
                        f"The trajectory direction is '{trajectory_def['direction']}' but no junction found!")
            else:
                if "junction_exit" not in trajectory_def:
                    wp_dict = group_junction_directions(junction, reference_yaw=trajectory[-1].rotation.yaw)
                    if trajectory_def["direction"] not in wp_dict:
                        raise InvalidScenarioDefinitionError(
                            f"No '{trajectory_def['direction']}' direction found in the junction!")
                    closest_wp = get_closest_wp(wp_dict[trajectory_def["direction"]],
                                                reference_loc=trajectory[-1].location)
                    trajectory_def["junction_exit"] = location_to_dict(closest_wp.transform.location)

                assert trajectory_def["junction_exit"] is not None
                waypoint = CarlaDataProvider.get_map().get_waypoint(dict_to_location(trajectory_def["junction_exit"]))
                for i in range(cfg.CONFIG["trajectory"]["junction_distance_limit"]):
                    trajectory.append(waypoint.transform)
                    waypoint = waypoint.next(1)[0]

        trajectory, gps_route, route = trajectory_interpolation([t.location for t in trajectory])

        return trajectory, gps_route, route, is_junction

    def set_trajectory(self, trajectory_def):
        """
        Set the ego vehicle's trajectory using a trajectory definition.

        :param trajectory_def: Dictionary describing the trajectory setup (start, direction, etc.).
        """
        self.ego_vehicle.yaw = trajectory_def["start"]["yaw"]
        self.ego_vehicle.speed = trajectory_def["start"]["speed"]
        load_world(trajectory_def["town"])
        trajectory_def["trajectory"], trajectory_def["gps_route"], trajectory_def[
            "route"], trajectory_def["start"]["is_junction"] = ScenarioDefinition._build_trajectory(
            trajectory_def)
        self.trajectory = trajectory_def
        self.town = trajectory_def["town"]

    def get_trigger_position(self):
        """
        Return the ego vehicle's starting position.

        :return: Dictionary containing the `x`, `y`, `z`, `yaw`, and `speed` values.
        """
        return self.trajectory["start"]

    def get_other_actors(self):
        """
        Get the configuration of all non-ego actors (vehicles, walkers, statics).

        :return: List of actor configuration dictionaries.
        """
        return [actor.get_config() for actor in self.vehicles + self.walkers + self.statics]

    def get_weather(self):
        """
        Get the weather settings for the scenario.

        :return: Dictionary with weather parameters.
        """
        weather_parameters = ScenarioDefinition._BLUEPRINTS["weather"][self.weather]
        weather_parameters["sun_altitude_angle"] = ScenarioDefinition._BLUEPRINTS["brightness"][self.brightness]
        return weather_parameters

    def set_brightness(self, brightness_value):
        """
        Update the brightness level to the nearest available preset.

        :param brightness_value: Desired sun brightness value.
        """
        self.brightness = self._find_closest_brightness_index(brightness_value)

    def _find_closest_brightness_index(self, brightness_value):
        """
        Find the index of the brightness preset closest to the given value.

        :param brightness_value: Desired brightness value.
        :return: Index of the closest brightness level.
        """
        ret = 0
        dist = None
        for idx, value in enumerate(ScenarioDefinition._BLUEPRINTS["brightness"]):
            if not dist or dist > abs(value - brightness_value):
                dist = abs(value - brightness_value)
                ret = idx
        return ret

    def add_actor(self, category: str, new_actor, mark=False, tilt_dir=None):
        """
        Add a new actor to the scenario.

        :param category: Type of the actor to add (:data:`vehicle`, :data:`walker`, or :data:`static`).
        :param new_actor: The actor instance to add.
        :param mark: Whether to mark this actor as special (default: :data:`False`).
        :param tilt_dir: Apply a tilt to the actor's position (:data:`left`, :data:`right`, or :data:`None`).
        """
        actors = getattr(self, f"{category}s")
        new_actor_dc = deepcopy(new_actor)
        new_actor_dc.mark = mark
        new_actor_dc.tilt(tilt_dir)
        actors.append(new_actor_dc)

    def remove_actor(self, category: str, region):
        """
        Remove the nearest actor of the specified category and region.

        :param category: Actor type (:data:`vehicle`, :data:`walker`, or :data:`static`).
        :param region: Region constraint for selecting the actor to remove.
        """
        actors = getattr(self, f"{category}s")
        index = ScenarioDefinition._pick_nearest_actor(actors, region)
        if index >= 0:
            del actors[index]

    def replace_actor(self, category: str, region, new_actor, mark=False, tilt_dir=None):
        """
        Replace the nearest actor of the given category and region with a new actor.

        :param category: Actor type (:data:`vehicle`, :data:`walker`, or :data:`static`).
        :param region: Region constraint for selecting the actor to replace.
        :param new_actor: New actor instance to insert.
        :param mark: Whether to mark the new actor (default: :data:`False`).
        :param tilt_dir: Tilt direction to apply (:data:`left`, :data:`right`, or :data:`None`).
        """
        actors = getattr(self, f"{category}s")
        index = ScenarioDefinition._pick_nearest_actor(actors, region)
        if index >= 0:
            del actors[index]
            new_actor_dc = deepcopy(new_actor)
            new_actor_dc.mark = mark
            new_actor_dc.tilt(tilt_dir)
            actors.append(new_actor_dc)

    def update_attribute(self, category: str, value):
        """
        Update a global scenario attribute.

        :param category: Attribute name to update (e.g., :data:`weather`, :data:`brightness`).
        :param value: New value to assign to the attribute.
        """
        setattr(self, category, value)

    def update_ego(self, category: str, value):
        """
        Update an attribute of the ego vehicle.

        :param category: Attribute to update (:data:`position`, :data:`yaw`, :data:`speed`, etc.).
        :param value: New value for the attribute.
        """
        if category == "position":
            original = self.get_trigger_position()
            load_world(self.town)
            x, y, z = self._next_waypoint(original["x"], original["y"], original["z"], value)
            if x is not None and y is not None and z is not None:
                try:
                    (self.trajectory["trajectory"],
                     self.trajectory["gps_route"],
                     self.trajectory["route"],
                     self.trajectory["start"]["is_junction"]
                     ) = self._build_trajectory({
                        "start": {"x": x, "y": y, "z": z},
                        "direction": self.trajectory["direction"]}
                    )
                    self.trajectory["start"]["x"], self.trajectory["start"]["y"], self.trajectory["start"][
                        "z"] = x, y, z
                except InvalidScenarioDefinitionError:
                    logger.warning(f"Unable to change the starting position.")
            else:
                logger.warning(f"Unable to change the starting position.")
        else:
            self.ego_vehicle.update_attribute(category, value)

    @staticmethod
    def _next_waypoint(x, y, z, interval):
        """
        Find the next waypoint at a certain distance from a given location.

        :param x: X-coordinate of the current location.
        :param y: Y-coordinate of the current location.
        :param z: Z-coordinate of the current location.
        :param interval: Distance to move along the road.
        :return: New (x, y) coordinates, or (None, None) if no waypoint is found.
        """
        waypoint = CarlaDataProvider.get_map().get_waypoint(carla.Location(x=x, y=y, z=z))
        new_waypoints = waypoint.next(interval)
        if len(new_waypoints) > 0:
            new_location = new_waypoints[0].transform.location
            return new_location.x, new_location.y, new_location.z
        else:
            return None, None, None

    def update_actor(self, category: str, attribute: str, value):
        """
        Update an attribute for all marked actors of a specific category.

        :param category: Actor type (:data:`vehicle`, :data:`walker`, or :data:`static`).
        :param attribute: Attribute name to update (e.g., :data:`speed`, :data:`yaw`).
        :param value: New value to assign to the attribute.
        """
        actors = getattr(self, f"{category}s")
        is_changed = False
        for actor in actors:
            if actor.mark:
                is_changed = True
                actor.update_attribute(attribute, value)
        if not is_changed:
            logger.warning(f"No marked actor in {category}.")

    @staticmethod
    def _random_pick_actor(actors, region: Boundary.Region = None):
        """
        Randomly pick an unmarked actor from a given list, optionally filtering by region.

        :param actors: List of actor instances to select from.
        :param region: Region to filter actors by (optional).
        :return: Index of the randomly selected actor, or :data:`-1` if :data:`None` found.
        """
        index, count = -1, 0
        for i, actor in enumerate(actors):
            if not actor.mark and (region is None or actor.region == region):
                count += 1
                if random.randint(1, count) == 1:
                    index = i
        return index

    @staticmethod
    def _pick_nearest_actor(actors, region: Boundary.Region = None):
        """
        Pick the nearest unmarked actor to the origin based on radius.

        :param actors: List of actor instances to choose from.
        :param region: Region to filter actors by (optional).
        :return: Index of the actor with minimum radius, or :data:`-1` if no actor satisfies conditions.
        """
        candidates = [(i, actor) for i, actor in enumerate(actors)
                      if not actor.mark and (region is None or actor.region == region)]
        return sorted(candidates, key=lambda x: x[1].radius)[0][0] if candidates else -1

    def build_actor_trajectory(self, actor_def, scenario_duration=cfg.CONFIG["simulation"]["scenario_duration"]):
        """
        Build a trajectory for an actor based on its spawn point and direction.

        :param actor_def: Actor definition containing spawn information.
        :param scenario_duration: Total duration of the scenario to compute the trajectory (in seconds).
        :return: List of trajectory points as dictionaries with `x` and `y` keys.
        """
        num_trajectory_points = 30
        spawn_point = actor_def.get_config()['spawn_point']

        loc = rotate_vector(spawn_point, self.trajectory["start"]["yaw"])

        loc["x"] += self.trajectory["start"]["x"]
        loc["y"] += self.trajectory["start"]["y"]
        yaw_rad = math.radians(spawn_point['yaw'])

        total_distance = actor_def.speed * scenario_duration \
            if hasattr(actor_def, "speed") else 0.5
        trajectory = []

        for i in range(num_trajectory_points):
            trajectory.append(
                {"x": loc["x"] + total_distance * math.cos(yaw_rad) * i / num_trajectory_points,
                 "y": loc["y"] + total_distance * math.sin(yaw_rad) * i / num_trajectory_points})

        return trajectory

    def score(self):
        """
        Compute an average trajectory score for all actors relative to the ego route.

        :return: Average trajectory score across all vehicles, walkers, and statics.
        """
        score = 0
        count = 0
        for actor in self.vehicles + self.walkers + self.statics:
            actor_traj = self.build_actor_trajectory(actor)
            score += single_trajectory_score([r[0] for r in self.trajectory["route"]], actor_traj)
            count += 1
        return score / count if count > 0 else 0

    def clear_marks(self):
        """
        Clear the :attr:`mark` attribute for all vehicles, walkers, and statics.
        """
        for actor in self.vehicles + self.walkers + self.statics:
            actor.mark = False

    def vectorize(self, max_actors: int, prefix: str, mode="stats"):
        """Vectorize the scenario.

        :param max_actors: Maximum number of actors.
            It should be greater than the value configured in `scenario:max_actors`.
        :param prefix: The string added before the feature names.
        :param mode: The way to encode actors. Options are :data:`stats` for encoding statistics
            or :data:`padding` for padding shorter lists of actors (default: :data:`stats`).
        :return: A :class:`DataFrame` representing the vector.
        """
        # Vectorize global attributes.
        df = pd.DataFrame({f"{prefix}_town": [self.town]}).join(
            pd.DataFrame({f"{prefix}_{attr}": [getattr(self, attr, None)]
                          for attr in self.ATTRIBUTES
                          if attr not in ("stop_sign_est", "red_light_est", "is_junction_est")})
        )

        # Vectorize the ego vehicle.
        df = df.join(self.ego_vehicle.vectorize(prefix=f"{prefix}_ego"))

        # Vectorize the trajectory.
        df = df.join(pd.DataFrame({f"{prefix}_traj_direction": [self.trajectory["direction"]]}))
        waypoints = self.trajectory["route"]
        n_wps = len(waypoints)
        assert n_wps >= 4
        percentiles = {
            "start": waypoints[0],
            "1q": waypoints[int((n_wps - 1) * 0.25)],
            "middle": waypoints[int((n_wps - 1) * 0.5)],
            "3q": waypoints[int((n_wps - 1) * 0.75)],
            "end": waypoints[-1],
        }
        df = df.join(pd.DataFrame({f"{prefix}_traj_{i}_{k}": [v]
                                   for i, wp in percentiles.items()
                                   for k, v in wp[0].items()}))

        # Vectorize actors.
        if mode == "padding":
            for category in self.DYNAMIC:
                actors = getattr(self, f"{category}s")
                if len(actors) > max_actors:
                    raise ValueError(f"The number of actors ({len(actors)}) should be less than or equal to "
                                     f"the maximum number of actors ({max_actors}).")
                df = df.join(pd.DataFrame({f"{prefix}_num_{category}": [len(actors)]}))
                actors = sorted(actors, key=lambda x: x.radius)
                for i, actor in enumerate(actors):
                    df = df.join(actor.vectorize(prefix=f"{prefix}_{category}_{i}"))
                i = len(actors)
                while i < max_actors:
                    cls = getattr(sys.modules[__name__], category.capitalize())
                    df = df.join(cls.vectorize_padding(prefix=f"{prefix}_{category}_{i}"))
                    i += 1
        else:
            for category in self.DYNAMIC:
                actors = getattr(self, f"{category}s")
                df = df.join(pd.DataFrame({f"{prefix}_num_{category}": [len(actors)]}))
                regions = {k: list(v) for k, v in groupby(actors, lambda x: x.region)}
                for region in list(Boundary.Region) + [None]:
                    region_name = region.name.lower() if region else None
                    actor_list = regions.get(region_name, [])
                    if len(actor_list) > 0:
                        actor_df = pd.concat([actor.vectorize(prefix=f"{prefix}_{category}_{region_name}")
                                              for actor in actor_list])
                        stats = {}
                        for col in actor_df.columns:
                            if "model" in col: continue
                            stats.update({
                                f"{col}_min": [actor_df[col].min()],
                                f"{col}_max": [actor_df[col].max()],
                                f"{col}_mean": [actor_df[col].mean()],
                                f"{col}_median": [actor_df[col].median()],
                            })
                        df = df.join(pd.DataFrame(stats))
                    else:
                        cls = getattr(sys.modules[__name__], category.capitalize())
                        empty_df = cls.vectorize_padding(prefix=f"{prefix}_{category}_{region_name}")
                        stats = {}
                        for col in empty_df.columns:
                            if "model" in col: continue
                            stats.update({
                                f"{col}_min": [0.0],
                                f"{col}_max": [0.0],
                                f"{col}_mean": [0.0],
                                f"{col}_median": [0.0],
                            })
                        df = df.join(pd.DataFrame(stats))

        return df

    @staticmethod
    def _list_eq(this, that):
        """
        Check if two lists contain the same elements regardless of order.

        :param this: First list to compare.
        :param that: Second list to compare.
        :return: :data:`True` if lists are equivalent, :data:`False` otherwise.
        """
        if len(this) != len(that): return False
        copy = list(this)
        try:
            for elem in that:
                copy.remove(elem)
        except ValueError:
            return False
        return not copy

    def __eq__(self, other):
        """
        Check if two ScenarioDefinition instances are equivalent.

        :param other: Another ScenarioDefinition to compare against.
        :return: :data:`True` if scenarios are equivalent, :data:`False` otherwise.
        """
        # return (isinstance(other, self.__class__) and
        return (str(type(self)) == str(type(other)) and
                self.town == other.town and
                self.ego_vehicle == other.ego_vehicle and
                self.trajectory == other.trajectory and
                self.weather == other.weather and
                self.brightness == other.brightness and
                self._list_eq(self.vehicles, other.vehicles) and
                self._list_eq(self.walkers, other.walkers) and
                self._list_eq(self.statics, other.statics))

    def __repr__(self):
        """
        Generate a string representation of the :class:`ScenarioDefinition`.

        :return: Readable string showing the scenario's key attributes.
        """
        return (f"Scenario(id={self.id_}, "
                f"town={self.town}, "
                f"ego_vehicle={self.ego_vehicle}, "
                f"trajectory={self.trajectory}, "
                f"vehicles={self.vehicles}, "
                f"walkers={self.walkers}, "
                f"statics={self.statics}, "
                f"weather={self.weather}, "
                f"brightness={self.brightness})")


class Actor(ABC):
    """
    Abstract base class representing a generic actor (vehicle, walker, or static) in the scenario.
    """

    _ATTRIBUTES = ["radius", "angle", "yaw", "model"]
    _BLUEPRINTS = None
    _BOUNDARY = None

    def __init__(self, radius, angle, yaw, model, *args, **kwargs):
        """
        Initialize a new :class:`Actor` instance.

        :param radius: Distance from the ego vehicle.
        :param angle: Angular direction around the ego vehicle.
        :param yaw: Actor orientation.
        :param model: Index of the blueprint model.
        """
        self.radius = radius
        self.angle = angle
        self.yaw = yaw
        self.model = model
        self.region = None
        self.update_region()
        self.mark = False

    def update_region(self):
        """
        Update the region (:data:`LEFT`, :data:`FOCUS`, :data:`RIGHT`) of the actor based on its angle.
        """
        self.region = self._BOUNDARY.get_region(self.angle)

    @classmethod
    def get_actor_index(cls, actor_blueprint):
        """
        Get the index of a blueprint model from its name.

        :param actor_blueprint: Blueprint name of the actor.
        :return: Index of the blueprint in the model list.
        """
        try:
            return cls._BLUEPRINTS["model"].index(actor_blueprint)
        except ValueError:
            logger.warning(f"No blueprint index found for '{actor_blueprint}', defaulting to 0.")
            return 0

    @classmethod
    def generate_random(cls, region: Boundary.Region = None, none_pb=None):
        """
        Generate a random actor with attributes sampled within boundaries.

        :param region: Optional region constraint for actor placement.
        :param none_pb: Probability of returning :data:`None` for a given attribute.
        :return: Randomly generated actor.
        """
        return cls(**{attr: cls._BOUNDARY.random(attr, region, none_pb)
                      for attr in Actor._ATTRIBUTES + cls._ATTRIBUTES})

    @classmethod
    def generate_random_actors(cls, probability, limit=cfg.CONFIG["scenario"]["max_actors"]):
        """
        Generate a list of random actors based on a sampling probability.

        :param probability: Probability of adding each actor.
        :param limit: Maximum number of actors to generate.
        :return: List of randomly generated actors.
        """
        actors = []
        times = 1
        while len(actors) < limit and random.random() < probability ** times:
            actors.append(cls.generate_random())
            times += 1
        return actors

    def dist(self, other, scaling=cfg.CONFIG["scenario"]["dist_scaling"]):
        """
        Compute a distance metric between this actor and another based on attribute differences.

        :param other: Another actor to compare.
        :param scaling: Scaling factor for categorical attributes.
        :return: Distance score.
        """
        if not isinstance(other, self.__class__):
            raise ValueError(f"Unmatched types: [{type(self)}, {type(other)}].")
        return _dist_attrs(self, other, Actor._ATTRIBUTES + self._ATTRIBUTES, self._BOUNDARY, scaling=scaling)

    def mate(self, other, cxpb=cfg.CONFIG["scenario"]["cxpb"]):
        """
        Perform uniform crossover (attribute swapping) between two actors.

        :param other: Another actor.
        :param cxpb: Probability of swapping each attribute.
        """
        if not isinstance(other, self.__class__):
            raise ValueError(f"Unmatched types: [{type(self)}, {type(other)}].")
        _mate_attrs(self, other, Actor._ATTRIBUTES + self._ATTRIBUTES, cxpb=cxpb)
        self.update_region()
        other.update_region()

    def mutate(self, mutpb=cfg.CONFIG["scenario"]["mutpb"],
               eta=cfg.CONFIG["scenario"]["mut_eta"],
               std=cfg.CONFIG["scenario"]["mut_std"]):
        """
        Apply mutation to actor attributes.

        :param mutpb: Mutation probability.
        :param eta: Crowding degree of the mutation (polynomial mutation parameter).
        :param std: Standard deviation for Gaussian mutation fallback.
        """
        _mutate_attrs(self, Actor._ATTRIBUTES + self._ATTRIBUTES, self._BOUNDARY, mutpb=mutpb, eta=eta, std=std)
        self.update_region()

    def tilt(self, tilt_dir):
        """
        Adjust the actor's angle slightly to left or right.

        :param tilt_dir: Direction of tilt (:data:`left` or :data:`right`).
        """
        coef = -1 if tilt_dir == "left" else (1 if tilt_dir == "right" else 0)
        self.angle += coef * cfg.CONFIG["boundary"]["tilt_degrees"]

    def update_attribute(self, category: str, value):
        """
        Update a specific attribute of the actor.

        :param category: Attribute name.
        :param value: New value for the attribute.
        """
        old_value = getattr(self, category)
        if value == old_value:
            logger.warning(f"The new {category} value is identical to the original.")
        setattr(self, category, value)

    def get_config(self):
        """
        Export the actor's configuration dictionary for spawning in simulation.

        :return: Configuration dictionary.
        """
        return {
            "role_name": f"{self.region.name.lower() if self.region else 'others'}{'-mark' if self.mark else ''}",
            "spawn_point": {
                "x": self.radius * math.cos(math.radians(self.angle)),
                "y": self.radius * math.sin(math.radians(self.angle)),
                "z": 0.0,
                "yaw": self.yaw,
            },
            "model": self._BLUEPRINTS["model"][self.model]
        }

    def vectorize(self, prefix: str):
        """Vectorize the actor.

        :param prefix: The string added before the actor's attribute name.
        :return: A :class:`DataFrame` representing the vector.
        """
        return pd.DataFrame({f"{prefix}_{attr}": [getattr(self, attr, None)]
                             for attr in Actor._ATTRIBUTES + self._ATTRIBUTES})

    @classmethod
    def vectorize_padding(cls, prefix: str):
        """Generate a padding vector for an actor.

        :param prefix: The string added before the actor's attribute name.
        :return: A :class:`DataFrame` representing the vector.
        """
        return pd.DataFrame({f"{prefix}_{attr}": [None]
                             for attr in Actor._ATTRIBUTES + cls._ATTRIBUTES})

    def __eq__(self, other):
        """
        Compare two actors for equality based on key attributes.

        :param other: Another actor.
        :return: :data:`True` if the actors are equal, :data:`False` otherwise.
        """
        return (isinstance(other, self.__class__) and
                self.radius == other.radius and
                self.angle == other.angle and
                self.yaw == other.yaw and
                self.model == other.model)

    def __repr__(self):
        """
        Return a human-readable string representation of the actor.

        :return: String showing actor class, attributes, and marking.
        """
        return (f"{self.__class__.__name__}{'*' if self.mark else ''}(region={self.region}, " +
                ", ".join(f"{attr}={str(getattr(self, attr))}" for attr in Actor._ATTRIBUTES + self._ATTRIBUTES) + ")")


class Vehicle(Actor):
    """
    :class:`Vehicle` class extending :class:`Actor`, representing a dynamic vehicle in the simulation.
    """
    _ATTRIBUTES = ["speed"]
    _BLUEPRINTS = cfg.CONFIG["blueprint"]["vehicle"]
    _BOUNDARY = Boundary(cfg.CONFIG["boundary"]["vehicle"])

    def __init__(self, radius, angle, yaw, model, speed):
        """
        Initialize a :class:`Vehicle` instance.

        :param radius: Distance from the ego vehicle.
        :param angle: Direction angle relative to ego vehicle.
        :param yaw: Orientation of the vehicle.
        :param model: Model index from blueprint.
        :param speed: initial speed of the vehicle.
        """
        super().__init__(radius, angle, yaw, model)
        self.speed = speed
        self.autopilot = True

    def get_config(self):
        """
        Return the vehicle configuration dictionary used for spawning.

        :return: Configuration dictionary including speed and autopilot status.
        """
        return {
            **super().get_config(),
            "speed": self.speed,
            "autopilot": self.autopilot,
        }

    @classmethod
    def generate_random(cls, region: Boundary.Region = None, none_pb=None, **filters):
        """
        Generate a random vehicle, optionally filtering by base model.

        :param region: Optional boundary region constraint.
        :param none_pb: Probability of returning :data:`None` for an attribute.
        :param filters: Filters to control base model selection.
        :return: Randomly generated vehicle.
        """
        vehicle = super().generate_random(region, none_pb)
        if "base_model" in filters:
            if isinstance(filters["base_model"], list):
                weights = [cls._BOUNDARY["base_model"][bm][1] - cls._BOUNDARY["base_model"][bm][0] + 1
                           for bm in filters["base_model"]]
                base_model = random.choices(filters["base_model"], weights=weights, k=1)[0]
            else:
                base_model = filters["base_model"]
            lower, upper = cls._BOUNDARY["base_model"][base_model]
            vehicle.model = random.randint(lower, upper)
        return vehicle

    def __eq__(self, other):
        """
        Compare two vehicles for equality including speed.

        :param other: Another :class:`Vehicle` object.
        :return: :data:`True` if equal, :data:`False` otherwise.
        """
        return super().__eq__(other) and self.speed == other.speed


class Walker(Actor):
    """
    :class:`Walker` class extending :class:`Actor`, representing a pedestrian in the simulation.
    """
    _ATTRIBUTES = ["speed"]
    _BLUEPRINTS = cfg.CONFIG["blueprint"]["walker"]
    _BOUNDARY = Boundary(cfg.CONFIG["boundary"]["walker"])

    def __init__(self, radius, angle, yaw, model, speed):
        """
        Initialize a :class:`Walker` instance.

        :param radius: Distance from the ego vehicle.
        :param angle: Direction angle relative to ego vehicle.
        :param yaw: Orientation of the walker.
        :param model: Model index from blueprint.
        :param speed: Walking speed.
        """
        super().__init__(radius, angle, yaw, model)
        self.speed = speed

    def get_config(self):
        """
        Return the walker configuration dictionary used for spawning.

        :return: Configuration dictionary including speed.
        """
        return {
            **super().get_config(),
            "speed": self.speed,
        }

    def __eq__(self, other):
        """
        Compare two walkers for equality including speed.

        :param other: Another :class:`Walker` object.
        :return: :data:`True` if equal, :data:`False` otherwise.
        """
        return super().__eq__(other) and self.speed == other.speed


class Static(Actor):
    """
    :class:`Static` class extending :class:`Actor`, representing a static object (e.g., prop, obstacle) in the simulation.
    """
    _ATTRIBUTES = []
    _BLUEPRINTS = cfg.CONFIG["blueprint"]["static"]
    _BOUNDARY = Boundary(cfg.CONFIG["boundary"]["static"])
