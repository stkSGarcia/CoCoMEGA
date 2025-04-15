import logging
import math
import numpy as np

import carla
from impl import config as cfg
from impl.utils.docker_utils import setup_carla
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from leaderboard.utils.route_manipulation import interpolate_trajectory

logger = logging.getLogger(__name__)


def initialize_carla(host=None, port=None, tm_port=None, gpu_device=None, seed=2000):
    try:
        # Initialize the Carla client and the world
        conf = cfg.CONFIG["simulation"]["instances"][0]
        host = host or conf["host"]
        port = port or conf["port"]
        tm_port = tm_port or conf["tm_port"]
        gpu_device = gpu_device or conf["gpu_device"]
        if cfg.CONFIG["docker"]["enabled"]:
            setup_carla(container_name=f"{cfg.CONFIG['docker']['image']}-{port}",
                        port=port,
                        gpu_device=gpu_device,
                        )
        client = carla.Client(host, port)
        client.set_timeout(cfg.CONFIG["simulation"]["client_timeout"])

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


def filter_junction_wp_direction(reference_yaw, wp_list, direction='opposite'):
    """
    Given a list of entry / exit wps of a junction, filters them according to a specific direction,
    returning all waypoint part of lanes that are at 'direction' with respect to the reference.
    This might fail for complex junctions, as only the wp yaws is checked, not their relative positions
    """

    filtered_wps = []
    available_dirs = []
    for wp in wp_list:
        diff = (wp.transform.rotation.yaw - reference_yaw) % 360
        if diff > 315.0 or diff < 45.0:
            wp_direction = 'ref'
            available_dirs.append('ref')
        elif 210.0 < diff <= 315.0:
            wp_direction = 'left'
            available_dirs.append('left')
        elif 45.0 <= diff < 150.0:
            wp_direction = 'right'
            available_dirs.append('right')
        else:
            wp_direction = 'opposite'
            available_dirs.append('opposite')

        if wp_direction == direction:
            filtered_wps.append(wp)
    return filtered_wps


def traj_interpolation(trajectory):
    gps_route, route = interpolate_trajectory(CarlaDataProvider.get_world(), trajectory)

    trajectory = [location_to_dict(t) for t in trajectory]
    gps_route = [(t, ro.name) for t, ro in gps_route]
    route = [(transform_to_dict(t), ro.name) for t, ro in route]

    return trajectory, gps_route, route


def get_junction(location, distance_limit=None):
    waypoint = CarlaDataProvider.get_map().get_waypoint(location)
    partial_trajectory = [waypoint]

    # Find the nearest junction
    dist = 0
    while not waypoint.is_junction:
        waypoint = waypoint.next(1.0)[0]
        dist += 1
        partial_trajectory.append(waypoint)
        if (distance_limit is not None) and dist > distance_limit:
            return partial_trajectory, None

    junction = waypoint.get_junction()
    return partial_trajectory, junction


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


def location_to_dict(location):
    return {
        'x': location.x,
        'y': location.y,
        'z': location.z,
    }


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
    if CarlaDataProvider.get_world() is None or CarlaDataProvider.get_world().get_map().name.lower() != town.lower():
        world = CarlaDataProvider.get_client().load_world(town)
        CarlaDataProvider.set_world(world)


def compass_to_yaw(compass):
    yaw = (compass * 180 / np.pi - 90)
    if yaw > 180:
        yaw = yaw - 360
    elif yaw < -180:
        yaw = compass + 360
    return yaw


def get_direction(trajectory):
    direction = "forward"
    reached_junction = False
    reference_wp = None
    for location, _ in trajectory:
        waypoint = CarlaDataProvider.get_map().get_waypoint(location)

        # Find the nearest junction
        if waypoint.is_junction:
            if not reached_junction:
                reference_wp = waypoint
            reached_junction = True
            continue

        if reached_junction and not waypoint.is_junction:
            diff = (waypoint.transform.rotation.yaw - reference_wp.transform.rotation.yaw) % 360
            if (diff > 315.0) or (diff < 45.0):
                direction = 'forward'
            elif 210.0 < diff <= 315.0:
                direction = 'left'
            elif 45.0 <= diff < 150.0:
                direction = 'right'
            else:
                direction = 'opposite'

        return direction


def get_available_directions(initial_transform, distance_limit=50):
    initial_location = initial_transform.location
    initial_waypoint = CarlaDataProvider.get_map().get_waypoint(initial_location)
    reference_yaw = initial_transform.rotation.yaw
    _, junction = get_junction(initial_location, distance_limit=distance_limit)
    if junction is None: return [("forward", None)]

    _, exit_wps = get_junction_topology(junction)

    wp_dict = {"forward": [], "left": [], "right": []}
    for wp in exit_wps:
        diff = (wp.transform.rotation.yaw - reference_yaw) % 360
        if (diff > 315.0) or (diff < 45.0):
            wp_dict["forward"].append(wp)
        elif 210.0 < diff <= 315.0:
            wp_dict["left"].append(wp)
        elif 45.0 <= diff < 150.0:
            wp_dict["right"].append(wp)

    directions = [(direction, get_closest_wp(wp_list, reference_wp=initial_waypoint)) for direction, wp_list in
                  wp_dict.items()]
    return directions
