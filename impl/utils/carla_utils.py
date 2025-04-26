import logging
import math

import carla
from impl.config import CONFIG
from impl.utils.docker_utils import setup_carla
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

logger = logging.getLogger(__name__)


def initialize_carla(host=None, port=None, tm_port=None, seed=2000):
    """
    Initialize the CARLA client and data provider.

    Sets client, traffic manager, random seed, and starts CARLA server if needed.

    :param host: IP address of the CARLA server.
    :param port: Port of the CARLA server.
    :param tm_port: Port of the traffic manager.
    :param gpu_device: GPU device ID to use.
    :param seed: Random seed for reproducibility.
    """
    try:
        # Initialize the Carla client and the world
        conf = CONFIG["simulation"]["docker"]["instances"][0]
        host = host or conf["host"]
        port = port or conf["port"]
        tm_port = tm_port or conf["tm_port"]
        if CONFIG["simulation"]["docker"]["enabled"]:
            setup_carla(container_name=f"{CONFIG['simulation']['docker']['image']}-{port}",
                        port=port)
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
    Retrieve entry and exit waypoints for a given junction.

    :param junction: CARLA junction object.
    :return: (List of entry waypoints, List of exit waypoints)
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
    Filter junction waypoints by relative direction to a reference yaw.

    :param reference_yaw: Reference yaw in degrees.
    :param wp_list: List of CARLA waypoints.
    :param direction: Target direction (:data:`left`, :data:`right`, :data:`forward`, or :data:`opposite`).
    :return: Filtered list of waypoints matching the direction.
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
    """
    Find the nearest junction from a location.

    :param location: CARLA location.
    :param distance_limit: Maximum distance to search.
    :return: (Partial trajectory to junction, junction object or :data:`None`)
    """
    waypoint = CarlaDataProvider.get_map().get_waypoint(location)

    # Find the nearest junction
    while not waypoint.is_junction:
        waypoint = waypoint.next(1.0)[0]

    junction = waypoint.get_junction()
    return waypoint, junction


def wp_dist(wp1, wp2):
    """
    Compute Euclidean distance between two CARLA locations.

    :param wp1: First CARLA location.
    :param wp2: Second CARLA location.
    :return: Distance in meters.
    """
    return math.sqrt(
        math.pow(wp1.transform.location.x - wp2.transform.location.x, 2) \
        + math.pow(wp1.transform.location.y - wp2.transform.location.y, 2)
    )


def get_closest_wp(wp_list, reference_wp):
    """
    Find the closest waypoint to a reference location.

    :param wp_list: List of waypoints.
    :param reference_loc: CARLA location to compare against.
    :return: Closest waypoint.
    """
    return min(wp_list, key=lambda wp: wp_dist(wp, reference_wp))


def transform_to_dict(transform):
    """
    Convert a CARLA transform to a dictionary.

    :param transform: CARLA transform.
    :return: Dictionary with `x`, `y`, `z`, `yaw`.
    """
    return {
        'x': transform.location.x,
        'y': transform.location.y,
        'z': transform.location.z,
        'yaw': transform.rotation.yaw,
    }


def dict_to_transform(_dict):
    """
    Convert a dictionary into a CARLA transform.

    :param _dict: Dictionary with `x`, `y`, `z`, `yaw` keys.
    :return: CARLA transform.
    """
    return carla.Transform(
        location=carla.Location(x=_dict["x"], y=_dict["y"], z=_dict["z"]),
        rotation=carla.Rotation(yaw=_dict["yaw"], pitch=0, roll=0)
    )


def dict_to_location(_dict):
    """
    Convert a dictionary into a CARLA location.

    :param _dict: Dictionary with `x`, `y`, `z` keys.
    :return: CARLA location.
    """
    return carla.Location(x=_dict["x"], y=_dict["y"], z=_dict["z"])


def copy_transform(transform):
    """
    Deep copy a CARLA transform.

    :param transform: CARLA transform to copy.
    :return: A new CARLA transform with the same data.
    """
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
    """
    Load a CARLA world by town name if not already loaded.

    :param town: Name of the CARLA map/town.
    """
    if CarlaDataProvider.get_client() is None:
        initialize_carla()
    if CarlaDataProvider.get_world() is None or CarlaDataProvider.get_world().get_map().name != town:
        world = CarlaDataProvider.get_client().load_world(town)
        CarlaDataProvider.set_world(world)
