import logging
import math

import carla
import inspect
import random
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from impl.scenario.carla_utils import get_junction_topology, filter_junction_wp_direction

logger = logging.getLogger(__name__)


def cartesian_to_polar(x, y):
    r = math.sqrt(x ** 2 + y ** 2)
    theta = math.degrees(math.atan2(y, x))
    return r, theta


class LeaderBoardFactory:
    @classmethod
    def generate(cls, scenario, scenario_type, **kwargs):
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
    def generate_opposite_lane_vehicles(scenario, num_vehicles=4, distance_between=15):
        from impl.scenario.scenario_definition import Vehicle
        ego_location = carla.Location(x=scenario.trajectory["start"]["x"], y=scenario.trajectory["start"]["y"], z=0)

        ego_waypoint = CarlaDataProvider.get_map().get_waypoint(ego_location)

        # Find the nearest junction
        while not ego_waypoint.is_junction:
            ego_waypoint = ego_waypoint.next(1.0)[0]

        junction = ego_waypoint.get_junction()
        entry_wps, _ = get_junction_topology(junction)

        # Filter waypoints for the opposite lane direction
        source_entry_wps = filter_junction_wp_direction(ego_waypoint, entry_wps, 'opposite')

        if not source_entry_wps:
            raise ValueError("No opposite lane found in the junction")

        source_wp = random.choice(source_entry_wps)

        # Calculate the positions of the opposite lane vehicles
        scenario.vehicles = []
        for _ in range(num_vehicles):
            vehicle = Vehicle.generate_random(base_model=['car', 'truck', 'van'])
            vehicle.autopilot = False
            source_transform = source_wp.transform
            radius, angle = cartesian_to_polar(source_transform.location.x - ego_location.x,
                                               source_transform.location.y - ego_location.y)
            vehicle.radius = radius
            vehicle.angle = angle - scenario.trajectory["start"]['yaw']
            vehicle.yaw = source_transform.rotation.yaw
            vehicle.speed = 10
            scenario.vehicles.append(vehicle)
            source_wp = source_wp.previous(distance_between)[0]

        return scenario
