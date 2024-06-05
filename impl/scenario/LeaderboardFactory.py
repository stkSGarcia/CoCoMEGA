import inspect
import logging
import math
import random

import carla
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from impl.scenario.carla_utils import get_junction_topology, filter_junction_wp_direction, get_junction, \
    dict_to_location

# Set up logging
logger = logging.getLogger(__name__)

crossings = {
    "left": ["left", "opposite", "right"],
    "forward": ["left", "right"],
    "right": ["left"]
}


def cartesian_to_polar(x, y):
    """
    Convert Cartesian coordinates to polar coordinates.

    Args:
        x (float): The x-coordinate.
        y (float): The y-coordinate.

    Returns:
        tuple: A tuple containing the radius (r) and the angle (theta) in degrees.
    """
    r = math.sqrt(x ** 2 + y ** 2)
    theta = math.degrees(math.atan2(y, x))
    return r, theta


class LeaderBoardFactory:
    @classmethod
    def generate(cls, scenario, scenario_type, **kwargs):
        """
        Generate a scenario based on the given scenario type.

        Args:
            scenario (object): The scenario object to be modified.
            scenario_type (str): The type of scenario to generate.
            **kwargs: Additional keyword arguments for the scenario generation methods.

        Returns:
            object: The modified scenario object.

        Raises:
            ValueError: If the scenario type is unknown.
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
    def generate_random_lane_vehicles(scenario, num_vehicles=4, distance_between=15, speed=10, vehicle_types=None):
        """
        Generate a random lane vehicle scenario.

        Args:
            scenario (object): The scenario object to be modified.
            num_vehicles (int): Number of vehicles to generate. Default is 4.
            distance_between (int): Distance between vehicles. Default is 15.
            speed (int): Speed of the vehicles. Default is 10.
            vehicle_types (list): List of vehicle types. Default is ['car', 'truck', 'van'].

        Returns:
            object: The modified scenario object.
        """
        if vehicle_types is None:
            vehicle_types = ['car', 'truck', 'van']
        lane_dir = random.choice(['ref', 'left', 'right', 'opposite'])
        scenario = LeaderBoardFactory._generate_lane_vehicles(
            scenario, lane_dir, num_vehicles, distance_between, speed, vehicle_types)
        return scenario

    @staticmethod
    def generate_random_crossing(scenario, num_vehicles=4, distance_between=15, speed=10, vehicle_types=None):
        """
        Generate a random lane crossing scenario.

        Args:
            scenario (object): The scenario object to be modified.
            num_vehicles (int): Number of vehicles to generate. Default is 4.
            distance_between (int): Distance between vehicles. Default is 15.
            speed (int): Speed of the vehicles. Default is 10.
            vehicle_types (list): List of vehicle types. Default is ['car', 'truck', 'van'].

        Returns:
            object: The modified scenario object.
        """
        if vehicle_types is None:
            vehicle_types = ['car', 'truck', 'van']
        lane_dir = random.choice(crossings[scenario.trajectory["direction"]])
        scenario = LeaderBoardFactory._generate_lane_vehicles(
            scenario, lane_dir, num_vehicles, distance_between, speed, vehicle_types)
        return scenario

    @staticmethod
    def generate_crossing_negotiation(scenario, min_vehicles=1, max_vehicles=2, distance_between=15, speed=10,
                                      vehicle_types=None):
        """
        Generate a crossing negotiation scenario.

        Args:
            scenario (object): The scenario object to be modified.
            min_vehicles (int): Minimum number of vehicles on each lane. Default is 1.
            max_vehicles (int): Maximum number of vehicles on each lane. Default is 2.
            distance_between (int): Distance between vehicles on each lane. Default is 15.
            speed (int): Speed of the vehicles. Default is 10.
            vehicle_types (list): List of vehicle types. Default is ['car', 'truck', 'van'].

        Returns:
            object: The modified scenario object.
        """
        if vehicle_types is None:
            vehicle_types = ['car', 'truck', 'van']
        for lane_dir in ['left', 'right', 'opposite']:
            num_vehicles = random.randint(min_vehicles, max_vehicles)
            scenario = LeaderBoardFactory._generate_lane_vehicles(
                scenario, lane_dir, num_vehicles, distance_between, speed, vehicle_types)
        return scenario

    @staticmethod
    def generate_motorcycles_crossing(scenario, num_vehicles=4, distance_between=10, speed=12):
        """
        Generate a random lane motorcycle crossing scenario.

        Args:
            scenario (object): The scenario object to be modified.
            num_vehicles (int): Number of vehicles to generate. Default is 4.
            distance_between (int): Distance between vehicles. Default is 15.
            speed (int): Speed of the vehicles. Default is 10.

        Returns:
            object: The modified scenario object.
        """
        return LeaderBoardFactory.generate_random_crossing(
            scenario, num_vehicles, distance_between, speed, vehicle_types=["motorcycle"])

    @staticmethod
    def generate_bicycles_crossing(scenario, num_vehicles=4, distance_between=8, speed=5):
        """
        Generate a random lane bicycle crossing scenario.

        Args:
            scenario (object): The scenario object to be modified.
            num_vehicles (int): Number of vehicles to generate. Default is 4.
            distance_between (int): Distance between vehicles. Default is 15.
            speed (int): Speed of the vehicles. Default is 10.

        Returns:
            object: The modified scenario object.
        """
        return LeaderBoardFactory.generate_random_crossing(
            scenario, num_vehicles, distance_between, speed, vehicle_types=["bicycle"])

    @staticmethod
    def generate_obstacle_avoidance(scenario, distance_from_junction=0):
        """
        Generate an obstacle avoidance scenario.

        Args:
            scenario (object): The scenario object to be modified.
            distance_from_junction (int): Distance from the junction to place the obstacle. Default is 0.

        Returns:
            object: The modified scenario object.
        """
        return LeaderBoardFactory._generate_obstacle_vehicle(scenario,
                                                             ["car", "truck", "van", "bicycle", "motorcycle"],
                                                             distance_from_junction=distance_from_junction)

    @staticmethod
    def generate_slow_moving_hazard(scenario, distance=15, speed=0.5):
        return LeaderBoardFactory._generate_obstacle_vehicle(scenario, ["bicycle", "motorcycle"],
                                                             distance=distance, speed=speed)

    @staticmethod
    def _generate_obstacle_vehicle(scenario, vehicle_types, distance=15, distance_from_junction=0, speed=0):
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
    def _generate_lane_vehicles(scenario, lane_dir, num_vehicles, distance_between, speed, vehicle_types):
        """
        Helper method to generate vehicles in a lane.

        Args:
            scenario (object): The scenario object to be modified.
            lane_dir (str): The direction of the lane.
            num_vehicles (int): Number of vehicles to generate.
            distance_between (int): Distance between vehicles.
            speed (int): Speed of the vehicles.
            vehicle_types (list): List of vehicle types.

        Returns:
            object: The modified scenario object.

        Raises:
            ValueError: If no waypoints are found for the specified lane direction.
        """
        ego_location = carla.Location(x=scenario.trajectory["start"]["x"], y=scenario.trajectory["start"]["y"], z=0)

        ego_waypoint, junction = get_junction(ego_location)
        entry_wps, _ = get_junction_topology(junction)

        # Filter waypoints for the specified lane direction
        source_entry_wps = filter_junction_wp_direction(ego_waypoint, entry_wps, lane_dir)

        if not source_entry_wps:
            raise ValueError(f"No '{lane_dir}' lane found in the junction")

        source_wp = random.choice(source_entry_wps)

        # Generate vehicles in the specified lane
        for _ in range(num_vehicles):
            vehicle = LeaderBoardFactory._generate_vehicle(scenario, source_wp.transform, vehicle_types, speed)
            scenario.vehicles.append(vehicle)
            source_wp = source_wp.previous(distance_between)[0]

        return scenario

    @staticmethod
    def _generate_vehicle(scenario, spawn_transform, vehicle_types, speed):
        """
        Helper method to generate a vehicle.

        Args:
            scenario (object): The scenario object to be modified.
            spawn_transform (carla.Transform): The transform of the vehicle to be spawned.
            vehicle_types (list): List of vehicle types.
            speed (int): Speed of the vehicle.

        Returns:
            object: The generated vehicle object.
        """
        from impl.scenario.scenario_definition import Vehicle
        ego_start = scenario.trajectory["start"]
        vehicle = Vehicle.generate_random(base_model=vehicle_types)
        vehicle.autopilot = False
        radius, angle = cartesian_to_polar(spawn_transform.location.x - ego_start["x"],
                                           spawn_transform.location.y - ego_start["y"])
        vehicle.radius = radius
        vehicle.angle = angle - ego_start["yaw"]
        vehicle.yaw = spawn_transform.rotation.yaw
        vehicle.speed = speed
        return vehicle
