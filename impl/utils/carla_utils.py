import logging
import math

import carla
from impl.config import CONFIG
from impl.utils.docker_utils import setup_carla
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

logger = logging.getLogger(__name__)


def initialize_carla(host=None, port=None, tm_port=None, gpu_device=None, seed=2000):
    try:
        # Initialize the Carla client and the world
        conf = CONFIG["simulation"]["instances"][0]
        host = host or conf["host"]
        port = port or conf["port"]
        tm_port = tm_port or conf["tm_port"]
        gpu_device = gpu_device or conf["gpu_device"]
        if CONFIG["docker"]["enabled"]:
            setup_carla(container_name=f"{CONFIG['docker']['image']}-{port}",
                        port=port,
                        gpu_device=gpu_device,
                        )
        client = carla.Client(host, port)
        client.set_timeout(CONFIG["simulation"]["client_timeout"])

        CarlaDataProvider.set_client(client)
        CarlaDataProvider.set_traffic_manager_port(tm_port)
        CarlaDataProvider.set_random_seed(seed)

    except Exception as e:
        logger.error(f"Error initializing Carla: {e}")
        raise e


def get_junction_topology(junction):
    """
    Given a junction, returns a two list of waypoints corresponding to the entry
    and exit lanes of the junction
    """

    def get_lane_key(waypoint):
        return str(waypoint.road_id) + '*' + str(waypoint.lane_id)

    def get_junction_entry_wp(entry_wp):
        while entry_wp.is_junction:
            entry_wps = entry_wp.previous(0.2)
            if len(entry_wps) == 0:
                return None
            entry_wp = entry_wps[0]
        return entry_wp

    def get_junction_exit_wp(exit_wp):
        while exit_wp.is_junction:
            exit_wps = exit_wp.next(0.2)
            if len(exit_wps) == 0:
                return None
            exit_wp = exit_wps[0]
        return exit_wp

    used_entry_lanes = []
    used_exit_lanes = []
    entry_wps = []
    exit_wps = []
    for entry_wp, exit_wp in junction.get_waypoints(carla.LaneType.Driving):
        entry_wp = get_junction_entry_wp(entry_wp)
        if not entry_wp:
            continue
        if get_lane_key(entry_wp) not in used_entry_lanes:
            used_entry_lanes.append(get_lane_key(entry_wp))
            entry_wps.append(entry_wp)

        exit_wp = get_junction_exit_wp(exit_wp)
        if not exit_wp:
            continue
        if get_lane_key(exit_wp) not in used_exit_lanes:
            used_exit_lanes.append(get_lane_key(exit_wp))
            exit_wps.append(exit_wp)

    return entry_wps, exit_wps


def filter_junction_wp_direction(reference_wp, wp_list, direction='opposite'):
    """
    Given a list of entry / exit wps of a junction, filters them according to a specific direction,
    returning all waypoint part of lanes that are at 'direction' with respect to the reference.
    This might fail for complex junctions, as only the wp yaws is checked, not their relative positions
    """

    filtered_wps = []
    reference_yaw = reference_wp.transform.rotation.yaw
    for wp in wp_list:
        diff = (wp.transform.rotation.yaw - reference_yaw) % 360
        if diff > 330.0:
            wp_direction = 'ref'
        elif diff > 225.0:
            wp_direction = 'right'
        elif diff > 135.0:
            wp_direction = 'opposite'
        elif diff > 30.0:
            wp_direction = 'left'
        else:
            wp_direction = 'ref'

        if wp_direction == direction:
            filtered_wps.append(wp)

    return filtered_wps


def get_junction(location):
    waypoint = CarlaDataProvider.get_map().get_waypoint(location)

    # Find the nearest junction
    while not waypoint.is_junction:
        waypoint = waypoint.next(1.0)[0]

    junction = waypoint.get_junction()
    return waypoint, junction


def wp_dist(wp1, wp2):
    return math.sqrt(
        math.pow(wp1.transform.location.x - wp2.transform.location.x, 2) \
        + math.pow(wp1.transform.location.y - wp2.transform.location.y, 2)
    )


def get_closest_wp(wp_list, reference_wp):
    return min(wp_list, key=lambda wp: wp_dist(wp, reference_wp))


def transform_to_dict(transform):
    return {
        'x': transform.location.x,
        'y': transform.location.y,
        'z': transform.location.z,
        'yaw': transform.rotation.yaw,
    }


def dict_to_transform(_dict):
    return carla.Transform(
        location=carla.Location(x=_dict["x"], y=_dict["y"], z=_dict["z"]),
        rotation=carla.Rotation(yaw=_dict["yaw"], pitch=0, roll=0)
    )


def dict_to_location(_dict):
    return carla.Location(x=_dict["x"], y=_dict["y"], z=_dict["z"])


def copy_transform(transform):
    return carla.Transform(
        carla.Location(
            x=transform.location.x,
            y=transform.location.y,
            z=transform.location.z,
        ),
        carla.Rotation(
            pitch=transform.rotation.pitch,
            yaw=transform.rotation.yaw,
            roll=transform.rotation.roll,
        )
    )


def load_world(town):
    if CarlaDataProvider.get_client() is None:
        initialize_carla()
    if CarlaDataProvider.get_world() is None or CarlaDataProvider.get_world().get_map().name != town:
        world = CarlaDataProvider.get_client().load_world(town)
        CarlaDataProvider.set_world(world)