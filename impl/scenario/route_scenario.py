#!/usr/bin/env python

# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides Challenge routes as standalone scenarios
"""

from __future__ import print_function
import logging
import math
import xml.etree.ElementTree as ET
import numpy.random as random

import py_trees

import carla
from carla.libcarla import Location

from agents.navigation.local_planner import RoadOption

from impl.scenario.carla_utils import dict_to_location
from impl.scenario.criterions import VehicleMeasurementTest
from impl.scenario.exceptions import InvalidScenarioDefinitionError

# pylint: disable=line-too-long
from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration, ActorConfigurationData
# pylint: enable=line-too-long

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest

from srunner.scenariomanager.scenarioatomics.atomic_behaviors import AccelerateToVelocity

from leaderboard.utils.route_parser import RouteParser, TRIGGER_THRESHOLD, TRIGGER_ANGLE_THRESHOLD
from leaderboard.utils.route_manipulation import interpolate_trajectory
from impl.config import CONFIG
from impl.scenario.scenario_definition import Walker

logger = logging.getLogger(__name__)

ROUTESCENARIO = ["RouteScenario"]

SECONDS_GIVEN_PER_METERS = 0.8  # for timeout
INITIAL_SECONDS_DELAY = 5.0


def oneshot_behavior(name, variable_name, behaviour):
    """
    This is taken from py_trees.idiom.oneshot.
    """
    # Initialize the variables
    blackboard = py_trees.blackboard.Blackboard()
    _ = blackboard.set(variable_name, False)

    # Wait until the scenario has ended
    subtree_root = py_trees.composites.Selector(name=name)
    check_flag = py_trees.blackboard.CheckBlackboardVariable(
        name=variable_name + " Done?",
        variable_name=variable_name,
        expected_value=True,
        clearing_policy=py_trees.common.ClearingPolicy.ON_INITIALISE
    )
    set_flag = py_trees.blackboard.SetBlackboardVariable(
        name="Mark Done",
        variable_name=variable_name,
        variable_value=True
    )
    # If it's a sequence, don't double-nest it in a redundant manner
    if isinstance(behaviour, py_trees.composites.Sequence):
        behaviour.add_child(set_flag)
        sequence = behaviour
    else:
        sequence = py_trees.composites.Sequence(name="OneShot")
        sequence.add_children([behaviour, set_flag])

    subtree_root.add_children([check_flag, sequence])
    return subtree_root


def convert_json_to_transform(actor_dict):
    """
    Convert a JSON string to a CARLA transform
    """
    return carla.Transform(location=carla.Location(x=float(actor_dict['x']), y=float(actor_dict['y']),
                                                   z=float(actor_dict['z'])),
                           rotation=carla.Rotation(roll=0.0, pitch=0.0, yaw=float(actor_dict['yaw'])))


def convert_json_to_actor(actor_dict):
    """
    Convert a JSON string to an ActorConfigurationData dictionary
    """
    node = ET.Element('waypoint')
    sp = actor_dict['spawn_point']
    node.set('x', sp['x'])
    node.set('y', sp['y'])
    node.set('z', sp['z'])
    node.set('yaw', sp['yaw'])
    node.set('model', actor_dict.get('model', 'vehicle.*'))
    node.set('speed', actor_dict.get('speed', 0))
    node.set('color', actor_dict.get('color', None))
    if actor_dict.get('autopilot', False):
        node.set('autopilot', 'true')
    if actor_dict.get('random_location', False):
        node.set('random_location', 'true')

    return ActorConfigurationData.parse_from_node(node, actor_dict.get('role_name', 'simulation'))


def convert_transform_to_location(transform_vec):
    """
    Convert a vector of transforms to a vector of locations
    """
    location_vec = []
    for transform_tuple in transform_vec:
        location_vec.append((transform_tuple[0].location, transform_tuple[1]))

    return location_vec


def convert_polar_to_cartesian(radius, angle_degrees):
    x = radius * math.cos(math.radians(angle_degrees))
    y = radius * math.sin(math.radians(angle_degrees))
    return x, y


def compare_scenarios(scenario_choice, existent_scenario):
    """
    Compare function for scenarios based on distance of the scenario start position
    """

    def transform_to_pos_vec(scenario):
        """
        Convert left/right/front to a meaningful CARLA position
        """
        position_vec = [scenario['trigger_position']]
        if scenario['other_actors'] is not None:
            if 'left' in scenario['other_actors']:
                position_vec += scenario['other_actors']['left']
            if 'front' in scenario['other_actors']:
                position_vec += scenario['other_actors']['front']
            if 'right' in scenario['other_actors']:
                position_vec += scenario['other_actors']['right']

        return position_vec

    # put the positions of the scenario choice into a vec of positions to be able to compare

    choice_vec = transform_to_pos_vec(scenario_choice)
    existent_vec = transform_to_pos_vec(existent_scenario)
    for pos_choice in choice_vec:
        for pos_existent in existent_vec:

            dx = float(pos_choice['x']) - float(pos_existent['x'])
            dy = float(pos_choice['y']) - float(pos_existent['y'])
            dz = float(pos_choice['z']) - float(pos_existent['z'])
            dist_position = math.sqrt(dx * dx + dy * dy + dz * dz)
            dyaw = float(pos_choice['yaw']) - float(pos_choice['yaw'])
            dist_angle = math.sqrt(dyaw * dyaw)
            if dist_position < TRIGGER_THRESHOLD and dist_angle < TRIGGER_ANGLE_THRESHOLD:
                return True

    return False


def request_new_actor(model, spawn_point, anchor, rolename='scenario', autopilot=False,
                      random_location=False, color=None, actor_category="car", speed=10):
    """
    This method tries to create a new actor, returning it if successful (raises InvalidScenarioConfError otherwise).
    """
    extra_height = 0.0
    if model.startswith('static'):
        wp = CarlaDataProvider.get_map().get_waypoint(spawn_point.location)
        if wp is not None:
            extra_height = wp.transform.location.z
    while True:
        try:
            blueprint = CarlaDataProvider.create_blueprint(model, rolename, color, actor_category)

            if random_location:
                actor = None
                while not actor:
                    _spawn_point = CarlaDataProvider._rng.choice(CarlaDataProvider._spawn_points)
                    actor = CarlaDataProvider._world.spawn_actor(blueprint, _spawn_point)

            else:
                # Incrementally lift the actor to avoid collisions with ground when spawning the actor
                # DO NOT USE spawn_point directly, as this will modify spawn_point permanently
                spawn_location = carla.Location(
                    x=spawn_point.location.x,
                    y=spawn_point.location.y,
                    z=spawn_point.location.z + extra_height,
                )
                _spawn_point = carla.Transform(anchor.transform(spawn_location), spawn_point.rotation) \
                    if anchor else carla.Transform(spawn_location, spawn_point.rotation)
                actor = CarlaDataProvider._world.spawn_actor(blueprint, _spawn_point)

            # wait for the actor to be spawned properly before we do anything
            if CarlaDataProvider.is_sync_mode():
                CarlaDataProvider._world.tick()
            else:
                CarlaDataProvider._world.wait_for_tick()

            if CONFIG["simulation"]["autopilot"] and autopilot and isinstance(actor, carla.Vehicle):
                actor.set_autopilot(autopilot, CarlaDataProvider._traffic_manager_port)
                speed_limit = actor.get_speed_limit() / 3.6
                pct = (speed_limit - speed) / speed_limit * 100
                tm = CarlaDataProvider._client.get_trafficmanager(CarlaDataProvider._traffic_manager_port)
                tm.vehicle_percentage_speed_difference(actor, pct)

            if actor is None:
                return None

            CarlaDataProvider._carla_actor_pool[actor.id] = actor
            CarlaDataProvider.register_actor(actor)
            return actor
        except Exception as e:
            if extra_height > 0.4:
                logger.error(f"Error has occurred while trying to spawn actor {model} on location {spawn_point}: {e}")
                raise InvalidScenarioDefinitionError(
                    f"An error has occurred while trying to spawn actor {model} on location {spawn_point}: {e}"
                )
            logger.debug("Increasing spawn height to avoid collision with the ground")
            extra_height += 0.1


class RouteScenario(BasicScenario):
    """
    Implementation of a RouteScenario, i.e. a scenario that consists of driving along a pre-defined route,
    along which several smaller scenarios are triggered
    """

    category = "RouteScenario"

    def __init__(self, world, scenario_definition, agent_instance, debug_mode=0, criteria_enable=True):
        """
        Setup all relevant parameters and create scenarios along route
        """
        # self.config = config
        self.route = None
        self.scenario_definition = scenario_definition
        self.agent_instance = agent_instance
        self.timeout = CONFIG['simulation']['scenario_duration']
        self._update_route(world, debug_mode > 0)
        self._vehicle_lights = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam
        self.weather_preset, self.weather_preset_name = \
            CarlaDataProvider.find_weather_presets()[self.scenario_definition.weather]
        ego_vehicle = self._update_ego_vehicle()
        scenario_config = self._build_scenario_configuration(scenario_definition, ego_vehicle, agent_instance)

        super(RouteScenario, self).__init__(name=scenario_config.name,
                                            ego_vehicles=[ego_vehicle],
                                            config=scenario_config,
                                            world=world,
                                            debug_mode=debug_mode > 1,
                                            terminate_on_failure=False,
                                            criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        initialization of other actors.
        """
        anchor = carla.Transform(
            carla.Location(
                x=self.route[0][0].location.x,
                y=self.route[0][0].location.y,
                z=self.route[0][0].location.z,
            ),
            carla.Rotation(
                pitch=self.route[0][0].rotation.pitch,
                yaw=self.route[0][0].rotation.yaw,
                roll=self.route[0][0].rotation.roll,
            )
        )
        if CONFIG["debug"]:
            self._draw_boundary(Walker._BOUNDARY.Region.FOCUS, anchor)
            # self._draw_boundary(Walker._BOUNDARY.Region.LEFT, anchor)
            # self._draw_boundary(Walker._BOUNDARY.Region.RIGHT, anchor)
            self._draw_route(self.route)

        if config.other_actors:
            for actor_conf in config.other_actors:
                new_actor = request_new_actor(
                    model=actor_conf.model,
                    spawn_point=actor_conf.transform,
                    anchor=anchor,
                    rolename=actor_conf.rolename,
                    autopilot=actor_conf.autopilot,
                    random_location=actor_conf.random_location,
                    color=actor_conf.color,
                    actor_category=actor_conf.category,
                    speed=actor_conf.speed,
                )
                self.other_actors.append(new_actor)

    def _update_route(self, world, debug_mode):
        """
        Update the input route, i.e. refine waypoint list, and extract possible scenario locations

        Parameters:
        - world: CARLA world
        - config: Scenario configuration (RouteConfiguration)
        """

        # prepare route's trajectory (build, interpolate and add the GPS route)
        trajectory = [dict_to_location(t) for t in self.scenario_definition.trajectory["trajectory"]]
        gps_route, route = interpolate_trajectory(world, trajectory)

        self.route = route
        CarlaDataProvider.set_ego_vehicle_route(convert_transform_to_location(self.route))

        self.agent_instance.set_global_plan(gps_route, self.route)

    def _update_ego_vehicle(self):
        """
        Set/Update the start position of the ego_vehicle
        """
        # move ego to correct position
        # elevate_transform = self.route[0][0]
        # elevate_transform.location.z += 0.5

        ego_vehicle = request_new_actor(
            model='vehicle.lincoln.mkz2017',
            spawn_point=self.route[0][0],
            anchor=None,
            rolename='hero',
            autopilot=False,
            random_location=False,
            color=None,
            actor_category="car",
        )

        spectator = CarlaDataProvider.get_world().get_spectator()
        ego_trans = ego_vehicle.get_transform()
        spectator.set_transform(carla.Transform(ego_trans.location + carla.Location(z=50),
                                                carla.Rotation(pitch=-90)))

        if self.weather_preset.sun_altitude_angle < 0.0:
            ego_vehicle.set_light_state(carla.VehicleLightState(self._vehicle_lights))

        return ego_vehicle

    def _estimate_route_timeout(self):
        """
        Estimate the duration of the route
        """
        route_length = 0.0  # in meters

        prev_point = self.route[0][0]
        for current_point, _ in self.route[1:]:
            dist = current_point.location.distance(prev_point.location)
            route_length += dist
            prev_point = current_point

        return int(SECONDS_GIVEN_PER_METERS * route_length + INITIAL_SECONDS_DELAY)

    # pylint: disable=no-self-use
    def _draw_waypoints(self, world, waypoints, vertical_shift, persistency=-1):
        """
        Draw a list of waypoints at a certain height given in vertical_shift.
        """
        for w in waypoints:
            wp = w[0].location + carla.Location(z=vertical_shift)

            size = 0.2
            if w[1] == RoadOption.LEFT:  # Yellow
                color = carla.Color(255, 255, 0)
            elif w[1] == RoadOption.RIGHT:  # Cyan
                color = carla.Color(0, 255, 255)
            elif w[1] == RoadOption.CHANGELANELEFT:  # Orange
                color = carla.Color(255, 64, 0)
            elif w[1] == RoadOption.CHANGELANERIGHT:  # Dark Cyan
                color = carla.Color(0, 64, 255)
            elif w[1] == RoadOption.STRAIGHT:  # Gray
                color = carla.Color(128, 128, 128)
            else:  # LANEFOLLOW
                color = carla.Color(0, 255, 0)  # Green
                size = 0.1

            world.debug.draw_point(wp, size=size, color=color, life_time=persistency)

        world.debug.draw_point(waypoints[0][0].location + carla.Location(z=vertical_shift), size=0.2,
                               color=carla.Color(0, 0, 255), life_time=persistency)
        world.debug.draw_point(waypoints[-1][0].location + carla.Location(z=vertical_shift), size=0.2,
                               color=carla.Color(255, 0, 0), life_time=persistency)

    def _draw_boundary(self, boundary, anchor, z=0.1):
        rads, angs = boundary.value["radius"], boundary.value["angle"]
        point_indices = [(0, 0), (0, 1), (1, 1), (1, 0)]
        points = [convert_polar_to_cartesian(rads[i], angs[j]) for i, j in point_indices]
        for i in range(len(points)):
            start_point = anchor.transform(carla.Location(x=points[i][0], y=points[i][1], z=z))
            end_point = anchor.transform(
                carla.Location(
                    x=points[(i + 1) % len(points)][0],
                    y=points[(i + 1) % len(points)][1],
                    z=z,
                )
            )
            CarlaDataProvider._world.debug.draw_line(start_point, end_point, thickness=0.15,
                                                     color=carla.Color(225, 10, 10),
                                                     life_time=0)

    def _draw_route(self, trajectory, z=0.1):
        for i in range(len(trajectory) - 1):
            # for i in range(40):
            start_point = carla.Location(
                x=trajectory[i][0].location.x,
                y=trajectory[i][0].location.y,
                z=z
            )
            end_point = carla.Location(
                x=trajectory[i + 1][0].location.x,
                y=trajectory[i + 1][0].location.y,
                z=z
            )
            CarlaDataProvider._world.debug.draw_line(start_point, end_point, thickness=0.15,
                                                     color=carla.Color(10, 10, 225),
                                                     life_time=0)
    def _scenario_sampling(self, potential_scenarios_definitions, random_seed=0):
        """
        The function used to sample the scenarios that are going to happen for this route.
        """

        # fix the random seed for reproducibility
        rgn = random.RandomState(random_seed)

        def position_sampled(scenario_choice, sampled_scenarios):
            """
            Check if a position was already sampled, i.e. used for another scenario
            """
            for existent_scenario in sampled_scenarios:
                # If the scenarios have equal positions then it is true.
                if compare_scenarios(scenario_choice, existent_scenario):
                    return True

            return False

        def select_scenario(list_scenarios):
            # priority to the scenarios with higher number: 10 has priority over 9, etc.
            higher_id = -1
            selected_scenario = None
            for scenario in list_scenarios:
                try:
                    scenario_number = int(scenario['name'].split('Scenario')[1])
                except:
                    scenario_number = -1

                if scenario_number >= higher_id:
                    higher_id = scenario_number
                    selected_scenario = scenario

            return selected_scenario

        def select_scenario_randomly(list_scenarios):
            # randomly select a scenario
            return rgn.choice(list_scenarios)

        # The idea is to randomly sample a scenario per trigger position.
        sampled_scenarios = []
        for trigger in potential_scenarios_definitions.keys():
            possible_scenarios = potential_scenarios_definitions[trigger]

            # scenario_choice = select_scenario(possible_scenarios) # original prioritized sampling
            scenario_choice = select_scenario_randomly(possible_scenarios)  # random sampling
            del possible_scenarios[possible_scenarios.index(scenario_choice)]
            # We keep sampling and testing if this position is present on any of the scenarios.
            while position_sampled(scenario_choice, sampled_scenarios):
                if possible_scenarios is None or not possible_scenarios:
                    scenario_choice = None
                    break
                scenario_choice = rgn.choice(possible_scenarios)
                del possible_scenarios[possible_scenarios.index(scenario_choice)]

            if scenario_choice is not None:
                sampled_scenarios.append(scenario_choice)

        return sampled_scenarios

    def _build_scenario_configuration(self, scenario_def, ego_vehicle, agent_instance):
        list_of_actor_conf_instances = self._get_actors_instances(scenario_def.get_other_actors())
        # Create an actor configuration for the ego-vehicle trigger position

        egoactor_trigger_position = convert_json_to_transform(scenario_def.get_trigger_position())
        scenario_config = ScenarioConfiguration()
        scenario_config.other_actors = list_of_actor_conf_instances
        scenario_config.trigger_points = [egoactor_trigger_position]
        scenario_config.name = 'Scenariotest'
        scenario_config.ego_vehicles = [ActorConfigurationData('vehicle.lincoln.mkz2017',
                                                               ego_vehicle.get_transform(),
                                                               'hero')]
        scenario_config.agent = agent_instance

        # Set weather
        scenario_config.weather = self.weather_preset

        return scenario_config

    def _get_actors_instances(self, list_of_antagonist_actors):
        """
        Get the full list of actor instances.
        Receives a list of actor definitions and creates an actual list of ActorConfigurationObjects
        """

        list_of_actors = []
        for actor_def in list_of_antagonist_actors:
            list_of_actors.append(convert_json_to_actor(actor_def))

        return list_of_actors

    def _create_behavior(self):
        """
        Basic behavior do nothing, i.e. Idle
        """

        behavior = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
        behavior.add_child(py_trees.behaviours.Running(name="Keep Running"))
        actor_definitions = self.scenario_definition.get_other_actors()
        for i, other_actor in enumerate(self.other_actors):
            if other_actor is not None and 'speed' in actor_definitions[i]:
                behavior.add_child(
                    AccelerateToVelocity(other_actor, throttle_value=1, target_velocity=actor_definitions[i]['speed']))

        return behavior

    def _create_test_criteria(self):
        """
        """
        criteria = []
        # route = convert_transform_to_location(self.route)

        vehicle_measurement = VehicleMeasurementTest(
            actor=self.ego_vehicles[0],
            other_actors=self.other_actors,
            measures=['brake', 'throttle', 'steer', 'velocity'],
            measurement_interval=10,
            scenario_def_id=self.scenario_definition.id_,
        )

        collision_criterion = CollisionTest(self.ego_vehicles[0], terminate_on_failure=True)

        # route_criterion = InRouteTest(self.ego_vehicles[0],
        #                               route=route,
        #                               offroad_max=30,
        #                               terminate_on_failure=True)

        # completion_criterion = RouteCompletionTest(self.ego_vehicles[0], route=route)

        # outsidelane_criterion = OutsideRouteLanesTest(self.ego_vehicles[0], route=route)

        # red_light_criterion = RunningRedLightTest(self.ego_vehicles[0])

        # stop_criterion = RunningStopTest(self.ego_vehicles[0])

        # blocked_criterion = ActorSpeedAboveThresholdTest(self.ego_vehicles[0],
        #                                                  speed_threshold=0.1,
        #                                                  below_threshold_max_time=180.0,
        #                                                  terminate_on_failure=True,
        #                                                  name="AgentBlockedTest")
        criteria.append(vehicle_measurement)
        # criteria.append(completion_criterion)
        # criteria.append(outsidelane_criterion)
        criteria.append(collision_criterion)
        # criteria.append(red_light_criterion)
        # criteria.append(stop_criterion)
        # criteria.append(route_criterion)
        # criteria.append(blocked_criterion)

        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
