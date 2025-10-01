import logging
import math
import os

import numpy as np

import carla
from impl import config as cfg
from impl.ads.utils.docker_utils import setup_carla
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from leaderboard.utils.route_manipulation import interpolate_trajectory


logger = logging.getLogger(__name__)


def initialize_carla(host=None, port=None, tm_port=None, gpu_device=None, seed=2000, tag="default"):
    """
    Initialize the CARLA client and data provider.

    Sets client, traffic manager, random seed, and starts CARLA server if needed.

    :param host: IP address of the CARLA server.
    :param port: Port of the CARLA server.
    :param tm_port: Port of the traffic manager.
    :param gpu_device: GPU device ID to use.
    :param seed: Random seed for reproducibility.
    :param tag: A specific tag used in container name.
    """
    try:
        # Initialize the Carla client and the world
        if "standalone" in cfg.CONFIG["simulation"] and cfg.CONFIG["simulation"]["standalone"]:
            default_conf = cfg.CONFIG["simulation"]["standalone"]
        else:
            default_conf = cfg.CONFIG["simulation"]["instances"][0]
        host = host or default_conf["host"]
        port = port or default_conf["port"]
        tm_port = tm_port or default_conf["tm_port"]
        gpu_device = gpu_device or default_conf["gpu_device"]
        if cfg.CONFIG["docker"]["enabled"]:
            setup_carla(container_name=f"{cfg.CONFIG['docker']['image']}-{tag}-{port}",
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


def carla_partial_cleanup():
    """
    Partially cleanup by destroying all spawned actors and clearing actor-related maps,
    but keeping the client alive for reloading the world.
    """
    DestroyActor = carla.command.DestroyActor
    batch = []

    # Destroy all actors that are alive while keeping world settings intact
    for actor_id in CarlaDataProvider._carla_actor_pool.copy():
        actor = CarlaDataProvider._carla_actor_pool[actor_id]
        # Stop and detach sensor streams before destruction to avoid dangling streams on the server
        try:
            if hasattr(actor, 'type_id') and isinstance(actor.type_id, str) and actor.type_id.startswith("sensor."):
                try:
                    # Stop streaming callbacks
                    if hasattr(actor, 'stop'):
                        actor.stop()
                except Exception:
                    pass
                try:
                    # Detach any listener callback
                    if hasattr(actor, 'listen'):
                        actor.listen(lambda _: None)
                except Exception:
                    pass
        except Exception:
            # Defensive: best-effort cleanup
            pass

        if actor.is_alive:
            batch.append(DestroyActor(actor))

    # Apply the batch destruction safely
    if CarlaDataProvider._client:
        try:
            CarlaDataProvider._client.apply_batch_sync(batch)
        except RuntimeError as e:
            if "time-out" in str(e):
                pass
            else:
                raise e

    # Let the server process the destructions (single tick)
    try:
        world = CarlaDataProvider.get_world()
        if world is not None:
            world.tick()
    except Exception:
        pass

    # Now only clear actor-related internal maps
    CarlaDataProvider._actor_velocity_map.clear()
    CarlaDataProvider._actor_location_map.clear()
    CarlaDataProvider._actor_transform_map.clear()
    CarlaDataProvider._traffic_light_map.clear()
    CarlaDataProvider._carla_actor_pool.clear()

    CarlaDataProvider._spawn_points = None
    CarlaDataProvider._spawn_index = 0
    CarlaDataProvider._ego_vehicle_route = None


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


def filter_junction_wp_direction(reference_yaw, wp_list, direction='opposite'):
    """
    Filter junction waypoints by relative direction to a reference yaw.

    :param reference_yaw: Reference yaw in degrees.
    :param wp_list: List of CARLA waypoints.
    :param direction: Target direction (:data:`left`, :data:`right`, :data:`forward`, or :data:`opposite`).
    :return: Filtered list of waypoints matching the direction.
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


def trajectory_interpolation(trajectory):
    """
    Interpolate a trajectory to get GPS route and navigation route.

    :param trajectory: List of CARLA locations.
    :return: (Interpolated trajectory dicts, GPS route, Full route)
    """
    gps_route, route = interpolate_trajectory(CarlaDataProvider.get_world(), trajectory)

    trajectory = [location_to_dict(t) for t in trajectory]
    gps_route = [(t, ro.name) for t, ro in gps_route]
    route = [(transform_to_dict(t), ro.name) for t, ro in route]

    return trajectory, gps_route, route


def get_junction(location, distance_limit=None):
    """
    Find the nearest junction from a location.

    :param location: CARLA location.
    :param distance_limit: Maximum distance to search.
    :return: (Partial trajectory to junction, junction object or :data:`None`)
    """
    waypoint = CarlaDataProvider.get_map().get_waypoint(location)
    partial_trajectory = [waypoint.transform]

    # Find the nearest junction
    dist = 0
    while not waypoint.is_junction:
        waypoint = waypoint.next(1.0)[0]
        dist += 1
        partial_trajectory.append(waypoint.transform)
        if (distance_limit is not None) and dist > distance_limit:
            return partial_trajectory, None

    junction = waypoint.get_junction()
    return partial_trajectory, junction


def loc_dist(loc1, loc2):
    """
    Compute Euclidean distance between two CARLA locations.

    :param loc1: First CARLA location.
    :param loc2: Second CARLA location.
    :return: Distance in meters.
    """
    return math.sqrt(
        math.pow(loc1.x - loc2.x, 2) \
        + math.pow(loc1.y - loc2.y, 2)
    )


def get_closest_wp(wp_list, reference_loc):
    """
    Find the closest waypoint to a reference location.

    :param wp_list: List of waypoints.
    :param reference_loc: CARLA location to compare against.
    :return: Closest waypoint.
    """
    return min(wp_list, key=lambda wp: loc_dist(wp.transform.location, reference_loc))


def transform_to_dict(transform):
    """
    Convert a CARLA transform to a dictionary.

    :param transform: CARLA transform.
    :return: Dictionary with ``x``, ``y``, ``z``, ``yaw``.
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

    :param _dict: Dictionary with ``x``, ``y``, ``z``, ``yaw`` keys.
    :return: CARLA transform.
    """
    return carla.Transform(
        location=carla.Location(x=_dict["x"], y=_dict["y"], z=_dict["z"]),
        rotation=carla.Rotation(yaw=_dict["yaw"], pitch=0, roll=0)
    )


def location_to_dict(location):
    """
    Convert a CARLA location to a dictionary.

    :param location: CARLA location.
    :return: Dictionary with ``x``, ``y``, ``z``.
    """
    return {
        'x': location.x,
        'y': location.y,
        'z': location.z,
    }


def dict_to_location(_dict):
    """
    Convert a dictionary into a CARLA location.

    :param _dict: Dictionary with ``x``, ``y``, ``z`` keys.
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


def load_world(town, cleanup=True):
    """
    Load a CARLA world by town name if not already loaded.

    :param town: Name of the CARLA map/town.
    """

    if CarlaDataProvider.get_client() is None:
        tag = os.environ.get("tag", "default")
        initialize_carla(tag=tag)
    if CarlaDataProvider.get_world() is None or CarlaDataProvider.get_world().get_map().name.lower() != town.lower():
        if cleanup:
            carla_partial_cleanup()
        world = CarlaDataProvider.get_client().load_world(town)
        CarlaDataProvider.set_world(world)


def compass_to_yaw(compass):
    """
    Convert compass radians to CARLA yaw degrees.

    :param compass: Compass value in radians.
    :return: Yaw in degrees (-180, 180].
    """
    yaw = (compass * 180 / np.pi - 90)
    if yaw > 180:
        yaw = yaw - 360
    elif yaw < -180:
        yaw = compass + 360
    return yaw


def get_direction(trajectory):
    """
    Infer route direction (:data:`forward`, :data:`left`, or :data:`right`) from a trajectory.
    Note: 'opposite' direction is not supported and will be mapped to 'forward'.

    :param trajectory: List of (location, road option) tuples.
    :return: String representing the general direction.
    """
    direction = "forward"
    reached_junction = False
    reference_wp = None
    warned_opposite = False
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
                # Map 'opposite' direction to 'forward' since opposite is not supported
                direction = 'forward'
                if not warned_opposite:
                    logger.warning(f"Opposite direction detected (diff={diff:.1f}), mapping to 'forward'")
                    warned_opposite = True

    return direction


def get_available_directions(initial_transform, distance_limit=None):
    """
    Get available exit directions from a junction based on an initial transform.

    :param initial_transform: CARLA transform.
    :param distance_limit: Distance limit to search for junctions.
    :return: List of (direction, closest waypoint) tuples.
    """
    trajectory, junction = get_junction(initial_transform.location, distance_limit=distance_limit)
    if junction is None: return [("forward", None)]
    wp_dict = group_junction_directions(junction, reference_yaw=trajectory[-1].rotation.yaw)
    directions = [(direction, get_closest_wp(wp_list, reference_loc=trajectory[-1].location)) for direction, wp_list in
                  wp_dict.items() if len(wp_list) > 0]
    return directions


def group_junction_directions(junction, reference_yaw):
    """
    Group junction exit waypoints into directions relative to a reference yaw.

    :param junction: CARLA junction object.
    :param reference_yaw: Yaw to compare against.
    :return: Dictionary with keys ``forward``, ``left``, ``right`` and lists of waypoints.
    """
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

    return wp_dict
