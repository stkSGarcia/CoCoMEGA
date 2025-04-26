import inspect
import logging
import math
import random

import carla
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from impl.utils.carla_utils import get_junction_topology, filter_junction_wp_direction, get_junction, \
    dict_to_location
from impl.utils.math_utils import cartesian_to_polar

# Set up logging
logger = logging.getLogger(__name__)

crossings = {
    "left": ["left", "opposite", "right"],
    "forward": ["left", "right"],
    "right": ["left"],
    None: ["ref"]
}


class LeaderBoardFactory:
    """
    Factory class for generating predefined types of driving scenarios in CARLA.
    """

    @classmethod
    def generate(cls, scenario, scenario_type, **kwargs):
        """
        Generate a scenario based on the given scenario type.

        :param scenario: The scenario object to modify.
        :param scenario_type: A string indicating the type of scenario to generate.
        :param kwargs: Additional keyword arguments for the generation function.
        :return: Modified scenario object.
        :raises ValueError: If `scenario_type` is unknown.
        """
        scenario_types = [name[9:] for name, _ in
                          inspect.getmembers(LeaderBoardFactory, predicate=inspect.isfunction) if
                          name.startswith("generate_")]

        if scenario_type == "random":
            scenario_type = random.choice(scenario_types)

        factory_method_name = f"generate_{scenario_type}"
        factory_method = getattr(cls, factory_method_name, None)

        if not factory_method:
            raise ValueError(f"Unknown scenario type: '{scenario_type}'. Choices are: {','.join(scenario_types)}")

        return factory_method(scenario, **kwargs)

    @staticmethod
    def generate_random_lane_vehicles(scenario, num_vehicles=4, distance_between=15, speed=10.0, vehicle_types=None,
                                      force_crossing=False):
        """
        Generate a random lane vehicle scenario.

        :param scenario: The scenario object to modify.
        :param num_vehicles: Number of vehicles to generate. Default is :data:`4`.
        :param distance_between: Distance between vehicles. Default is :data:`15`.
        :param speed: Speed of the vehicles. Default is :data:`10.0`.
        :param vehicle_types: List of vehicle types. Default is :data:`['car', 'truck', 'van']`.
        :param force_crossing: Whether to force crossing vehicles. Default is :data:`False`.
        :return: Modified scenario object.
        """
        if vehicle_types is None:
            vehicle_types = ['car', 'truck', 'van']
        lane_dir = random.choice(
            crossings[scenario.trajectory["direction"]] if force_crossing else ["ref", "left", "right", "opposite"]
        )
        scenario = LeaderBoardFactory._generate_lane_vehicles(
            scenario, lane_dir, num_vehicles, distance_between, speed, vehicle_types, autopilot=False)
        return scenario

    @staticmethod
    def generate_crossing_negotiation(scenario, min_vehicles=1, max_vehicles=2, distance_between=15, speed=10.0,
                                      vehicle_types=None):
        """
        Generate a crossing negotiation scenario with multiple lane directions.

        :param scenario: The scenario object to modify.
        :param min_vehicles: Minimum number of vehicles on each lane. Default is :data:`1`.
        :param max_vehicles: Maximum number of vehicles on each lane. Default is :data:`2`.
        :param distance_between: Distance between vehicles on each lane. Default is :data:`15`.
        :param speed: Speed of the vehicles. Default is :data:`10.0`.
        :param vehicle_types: List of vehicle types. Default is :data:`['car', 'truck', 'van']`.
        :return: Modified scenario object.
        """
        if vehicle_types is None:
            vehicle_types = ['car', 'truck', 'van']
        for lane_dir in ['left', 'right', 'opposite']:
            num_vehicles = random.randint(min_vehicles, max_vehicles)
            scenario = LeaderBoardFactory._generate_lane_vehicles(
                scenario, lane_dir, num_vehicles, distance_between, speed, vehicle_types, autopilot=True)
        return scenario

    @staticmethod
    def generate_motorcycles_crossing(scenario, num_vehicles=4, distance_between=10, speed=12.0):
        """
        Generate a random lane motorcycle crossing scenario.

        :param scenario: The scenario object to modify.
        :param num_vehicles: Number of vehicles to generate. Default is :data:`4`.
        :param distance_between: Distance between vehicles. Default is :data:`10`.
        :param speed: Speed of the vehicles. Default is :data:`12.0`.
        :return: Modified scenario object.
        """
        return LeaderBoardFactory.generate_random_lane_vehicles(
            scenario, num_vehicles, distance_between, speed, vehicle_types=["motorcycle"], force_crossing=True)

    @staticmethod
    def generate_bicycles_crossing(scenario, num_vehicles=4, distance_between=8, speed=5.0):
        """
        Generate a random lane bicycle crossing scenario.

        :param scenario: The scenario object to modify.
        :param num_vehicles: Number of vehicles to generate. Default is :data:`4`.
        :param distance_between: Distance between vehicles. Default is :data:`8`.
        :param speed: Speed of the vehicles. Default is :data:`5.0`.
        :return: Modified scenario object.
        """
        return LeaderBoardFactory.generate_random_lane_vehicles(
            scenario, num_vehicles, distance_between, speed, vehicle_types=["bicycle"], force_crossing=True)

    @staticmethod
    def generate_obstacle_avoidance(scenario, distance_from_junction=0):
        """
        Generate an obstacle avoidance scenario.

        :param scenario: The scenario object to modify.
        :param distance_from_junction: Distance from the junction to place the obstacle. Default is :data:`0`.
        :return: Modified scenario object.
        """
        return LeaderBoardFactory._generate_obstacle_vehicle(scenario,
                                                             ["car", "truck", "van", "bicycle", "motorcycle"],
                                                             distance_from_junction=distance_from_junction)

    @staticmethod
    def generate_slow_moving_hazard(scenario, distance=15, speed=1.0):
        """
        Generate a slow moving hazard scenario.

        :param scenario: The scenario object to modify.
        :param distance: Distance from the ego vehicle to generate the hazard. Default is :data:`15`.
        :param speed: Speed of the vehicles. Default is :data:`1.0`.
        :return: Modified scenario object.
        """
        return LeaderBoardFactory._generate_obstacle_vehicle(scenario, ["bicycle", "motorcycle"],
                                                             distance=distance, speed=speed)

    @staticmethod
    def generate_pedestrian_emerging(scenario, distance=10, speed=0.8):
        """
        Generate a pedestrian emerging scenario.

        :param scenario: The scenario object to modify.
        :param distance: Distance between the ego vehicle and the pedestrian to be spawned. Default is :data:`10`.
        :param speed: Speed of the pedestrian. Default is :data:`0.8`.
        :return: Modified scenario object.
        """
        from impl.scenario.scenario_definition import Walker
        ego_start = scenario.trajectory["start"]
        walker = Walker.generate_random()
        walker.radius = distance
        walker.angle = 30.0
        walker.yaw = ego_start["yaw"] - 90.0
        walker.speed = speed
        walker.update_region()
        scenario.walkers.append(walker)
        return scenario

    @staticmethod
    def _generate_obstacle_vehicle(scenario, vehicle_types, distance=15, distance_from_junction=0, speed=0.0):
        """
        Generate an obstacle vehicle.

        :param scenario: The scenario object to modify.
        :param vehicle_types: List of vehicle types.
        :param distance: Distance from the ego vehicle to generate the hazard. Default is :data:`15`.
        :param distance_from_junction: Distance from the junction to place the obstacle. Default is :data:`0`.
        :param speed: Speed of the vehicles. Default is :data:`0.0`.
        :return: Modified scenario object.
        """
        waypoints = [CarlaDataProvider.get_map().get_waypoint(dict_to_location(loc))
                     for loc in scenario.trajectory["trajectory"]]
        candidate_wps = [wp for wp in waypoints if wp.previous(1)[0].is_junction]
        if candidate_wps:
            target_wp = candidate_wps[0]
            if distance_from_junction > 0:
                target_wp = target_wp.next(distance_from_junction)[0]
        else:
            target_wp = waypoints[0].next(distance)[0]
        vehicle = LeaderBoardFactory._generate_vehicle(scenario, target_wp.transform,
                                                       vehicle_types=vehicle_types, speed=speed)
        scenario.vehicles.append(vehicle)
        return scenario

    @staticmethod
    def _generate_lane_vehicles(scenario, lane_dir, num_vehicles, distance_between, speed, vehicle_types, autopilot):
        """
        Helper method to generate vehicles in a lane.

        :param scenario: The scenario object to modify.
        :param lane_dir: The direction of the lane.
        :param num_vehicles: Number of vehicles to generate.
        :param distance_between: Distance between vehicles.
        :param speed: Speed of the vehicles.
        :param vehicle_types: List of vehicle types.
        :return: Modified scenario object.
        """
        ego_location = carla.Location(x=scenario.trajectory["start"]["x"], y=scenario.trajectory["start"]["y"],
                                      z=scenario.trajectory["start"]["z"])

        _, junction = get_junction(ego_location)
        entry_wps, _ = get_junction_topology(junction)

        # Filter waypoints for the specified lane direction
        source_entry_wps = filter_junction_wp_direction(scenario.trajectory["start"]["yaw"], entry_wps, lane_dir)

        if not source_entry_wps:
            logger.warning(f"No '{lane_dir}' lane found in the junction")
            return scenario

        source_wp = random.choice(source_entry_wps)

        # Generate vehicles in the specified lane
        for _ in range(num_vehicles):
            vehicle = LeaderBoardFactory._generate_vehicle(scenario, source_wp.transform, vehicle_types, speed,
                                                           autopilot)
            scenario.vehicles.append(vehicle)
            source_wp = source_wp.previous(distance_between)[0]

        return scenario

    @staticmethod
    def _generate_vehicle(scenario, spawn_transform, vehicle_types, speed, autopilot=True):
        """
        Helper method to generate a vehicle.

        :param scenario: The scenario object to modify.
        :param spawn_transform: The transform of the vehicle to be spawned.
        :param vehicle_types: List of vehicle types.
        :param speed: Speed of the vehicle.
        :return: Modified scenario object.
        """
        from impl.scenario.scenario_definition import Vehicle
        ego_start = scenario.trajectory["start"]
        vehicle = Vehicle.generate_random(base_model=vehicle_types)
        vehicle.autopilot = autopilot
        radius, angle = cartesian_to_polar(spawn_transform.location.x - ego_start["x"],
                                           spawn_transform.location.y - ego_start["y"])
        vehicle.radius = radius
        vehicle.angle = angle - ego_start["yaw"]
        vehicle.yaw = spawn_transform.rotation.yaw
        vehicle.speed = speed
        vehicle.update_region()
        return vehicle
