import logging
import math

import carla
import inspect
import random
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from impl.scenario.carla_utils import get_junction_topology, filter_junction_wp_direction
from impl.scenario.docker_utils import setup_carla
from impl.config import CONFIG

logger = logging.getLogger(__name__)


def cartesian_to_polar(x, y):
    r = math.sqrt(x ** 2 + y ** 2)
    theta = math.degrees(math.atan2(y, x))
    return r, theta


class LeaderBoardFactory:
    @classmethod
    def initialize_carla(cls, town):
        try:
            # Initialize the Carla client and the world
            if CarlaDataProvider.get_client() is None:
                if CONFIG["simulation"]["docker"]["enabled"]:
                    conf = CONFIG["simulation"]["docker"]["instances"][0]
                    setup_carla(container_name=f"{CONFIG['simulation']['docker']['image']}-{conf['port']}",
                                port=conf['port'])
                cls.client = carla.Client(conf["host"], conf["port"])
                cls.client.set_timeout(CONFIG["simulation"]["client_timeout"])
                CarlaDataProvider.set_client(cls.client)
                CarlaDataProvider.set_random_seed(2000)

            if CarlaDataProvider.get_world() is None or CarlaDataProvider.get_world().get_map().name != town:
                world = cls.client.load_world(town)
                CarlaDataProvider.set_world(world)
        except Exception as e:
            logger.error(f"Error initializing Carla: {e}")
            raise e

    @classmethod
    def generate(cls, scenario, scenario_type, **kwargs):
        cls.initialize_carla(scenario.town)
        scenario_types = [name[9:] for name, _ in
                          inspect.getmembers(LeaderBoardFactory, predicate=inspect.isfunction) if
                          name.startswith("generate")]

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
        ego_location = scenario.trajectory[0]
        ego_location = carla.Location(x=ego_location['x'], y=ego_location['y'], z=0)

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
            vehicle = Vehicle.generate_random()
            source_transform = source_wp.transform
            radius, angle = cartesian_to_polar(source_transform.location.x - ego_location.x,
                                               source_transform.location.y - ego_location.y)
            vehicle.radius = radius
            vehicle.angle = angle - scenario.trajectory[0]['yaw']
            vehicle.yaw = source_transform.rotation.yaw
            vehicle.speed = 10
            scenario.vehicles.append(vehicle)
            source_wp = source_wp.previous(distance_between)[0]

        return scenario
