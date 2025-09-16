import os
import yaml
import logging
import numpy as np

from impl import config as cfg
from impl.ads.scenario.scenario_definition import ScenarioDefinition, Vehicle, Walker, Static
from impl.ads.utils.carla_utils import trajectory_interpolation, location_to_dict, compass_to_yaw
from impl.ads.utils.math_utils import cartesian_to_polar, vector_norm, polar_to_cartesian

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

logger = logging.getLogger(__name__)

routes = {}
routes[
    "training_routes/routes_town01_short.xml"
] = "scenarios/town01_all_scenarios.json"
routes["training_routes/routes_town01_tiny.xml"] = "scenarios/town01_all_scenarios.json"
routes[
    "training_routes/routes_town02_short.xml"
] = "scenarios/town02_all_scenarios.json"
routes["training_routes/routes_town02_tiny.xml"] = "scenarios/town02_all_scenarios.json"
routes[
    "training_routes/routes_town03_short.xml"
] = "scenarios/town03_all_scenarios.json"
routes["training_routes/routes_town03_tiny.xml"] = "scenarios/town03_all_scenarios.json"
routes[
    "training_routes/routes_town04_short.xml"
] = "scenarios/town04_all_scenarios.json"
routes["training_routes/routes_town04_tiny.xml"] = "scenarios/town04_all_scenarios.json"
routes[
    "training_routes/routes_town05_short.xml"
] = "scenarios/town05_all_scenarios.json"
routes["training_routes/routes_town05_tiny.xml"] = "scenarios/town05_all_scenarios.json"
routes["training_routes/routes_town05_long.xml"] = "scenarios/town05_all_scenarios.json"
routes[
    "training_routes/routes_town06_short.xml"
] = "scenarios/town06_all_scenarios.json"
routes["training_routes/routes_town06_tiny.xml"] = "scenarios/town06_all_scenarios.json"
routes[
    "training_routes/routes_town07_short.xml"
] = "scenarios/town07_all_scenarios.json"
routes["training_routes/routes_town07_tiny.xml"] = "scenarios/town07_all_scenarios.json"
routes[
    "training_routes/routes_town10_short.xml"
] = "scenarios/town10_all_scenarios.json"
routes["training_routes/routes_town10_tiny.xml"] = "scenarios/town10_all_scenarios.json"
routes[
    "additional_routes/routes_town01_long.xml"
] = "scenarios/town01_all_scenarios.json"
routes[
    "additional_routes/routes_town02_long.xml"
] = "scenarios/town02_all_scenarios.json"
routes[
    "additional_routes/routes_town03_long.xml"
] = "scenarios/town03_all_scenarios.json"
routes[
    "additional_routes/routes_town04_long.xml"
] = "scenarios/town04_all_scenarios.json"
routes[
    "additional_routes/routes_town06_long.xml"
] = "scenarios/town06_all_scenarios.json"

towned_routes = {}
for route, scenario in routes.items():
    town = route.split("/")[1].split("_")[1]
    town = route.split("/")[1].split("_")[1]

    if town not in towned_routes:
        towned_routes[town] = {}
    towned_routes[town][route] = scenario

reweight_array = np.array([1.0, 3.5, 3.5, 2.0, 3.5, 2.0, 8.0])


def get_enviroment_confs():
    """
    Generate a list of environment configurations based on towns, weathers, and route types.

    :return: List of environment configurations.
    """
    weathers = cfg.CONFIG["runtime"]["weathers"]
    towns = cfg.CONFIG["runtime"]["towns"]
    route_types = cfg.CONFIG["runtime"]["routes"]
    confs = []
    for town in towns:
        for weather in weathers:
            for route, scenario in towned_routes[town].items():
                route_type = route.split("_")[-1][:-4]
                if route_type not in route_types:
                    continue
                conf = {
                    "town": town,
                    "weather": weather,
                    "route": route,
                    "route_name": route.split("/")[1].split(".")[0],
                    "scenario": scenario,
                }
                confs.append(conf)
    return confs


def make_yamls():
    """
    Create YAML files for data collection configuration for each weather condition.

    Saves YAML files under the configured YAML root directory.
    """
    conf = cfg.CONFIG["data_collection"].copy()
    os.makedirs(cfg.CONFIG["data_collection"]["yaml_root"], exist_ok=True)
    for weather in cfg.CONFIG["runtime"]["weathers"]:
        conf["weather"] = weather
        file_path = os.path.join(cfg.CONFIG["data_collection"]["yaml_root"], f"weather-{weather}.yaml")
        try:
            with open(file_path, "w") as file:
                yaml.dump(conf, file, default_flow_style=False, sort_keys=True)
        except Exception as e:
            logger.error(f"An error occurred while saving the YAML file {file_path}")
            raise e


def vectorize_realtime_data(rt_data):
    """
    Convert runtime simulation data into a :class:`ScenarioDefinition` object.

    :param rt_data: Dictionary containing runtime simulation data.
    :return: Generated :class:`ScenarioDefinition` instance.
    """
    scenario_def = ScenarioDefinition._generate_empty_scenario()
    trajectory = rt_data["sim_data"]["trajectory"]
    scenario_def.set_trajectory({
        "town": rt_data["sim_data"]["town"],
        **trajectory,
    })
    scenario_def.weather = int(rt_data["sim_data"]["weather"])
    scenario_def.set_brightness(rt_data["sim_data"]["brightness"])
    scenario_def.town = rt_data["sim_data"]["town"]
    scenario_def.is_junction_est = rt_data["traffic"]["is_junction"]
    scenario_def.stop_sign_est = rt_data["traffic"]["stop_sign"]
    scenario_def.red_light_est = rt_data["traffic"]["red_light"]

    for _id, other_actor in rt_data["sim_data"]["other_actors"].items():
        relative_position = (
            other_actor["loc"][0] - trajectory["start"]["x"], other_actor["loc"][1] - trajectory["start"]["y"])
        radius, angle = cartesian_to_polar(*relative_position)
        relative_angle = angle - rt_data["sim_data"]["trajectory"]["start"]["yaw"]
        yaw = other_actor["ori"]["yaw"]
        if other_actor["tpe"] == 0:
            speed = vector_norm(other_actor["vel"])
            model = Vehicle.get_actor_index(other_actor["blueprint"])
            scenario_def.vehicles.append(
                Vehicle(radius=radius, angle=relative_angle, yaw=yaw, model=model, speed=speed))
        if other_actor["tpe"] == 1:
            speed = vector_norm(other_actor["vel"])
            model = Walker.get_actor_index(other_actor["blueprint"])
            scenario_def.walkers.append(Walker(radius=radius, angle=relative_angle, yaw=yaw, model=model, speed=speed))
        if other_actor["tpe"] == 2:
            scenario_def.statics.append(Static(radius=radius, angle=relative_angle, yaw=yaw, model=0))
    return scenario_def


def rulefit_vectorize(rt_data):
    """
    Convert runtime simulation data into a feature vector based on the generated rules from Rulefit model.

    :param rt_data: Dictionary containing runtime simulation data.
    :return: Feature vector representing the scenario.
    """
    return np.ones(100)


def create_dataset_index(dataset_root, weathers=None, towns=None):
    """
    Create a ``dataset_index.txt`` file listing paths and frame counts for training or evaluation data.

    :param dataset_root: Path to the dataset root directory.
    :param weathers: Optional list of weather IDs to filter data.
    :param towns: Optional list of town IDs to filter data.
    """
    index_file_path = os.path.join(dataset_root, "dataset_index.txt")

    with open(index_file_path, "w") as index_file:
        for root, dirs, files in os.walk(dataset_root):
            # Filter for directories containing relevant data frames
            if "rgb_front" in dirs:
                data_weather = int(os.path.basename(os.path.dirname(root)).split("-")[-1])
                data_town = int(root.split("/")[-1].split("_")[1][4:])
                if (weathers and data_weather not in weathers) or (towns and data_town not in towns): continue
                # Count the number of frames in the directory
                rgb_dir_path = os.path.join(root, "rgb_front")
                frame_count = len(
                    [file for file in os.listdir(rgb_dir_path) if os.path.isfile(os.path.join(rgb_dir_path, file))])

                # Write the relative path and frame count to the index file
                relative_path = os.path.relpath(root, dataset_root)
                index_file.write(f"{relative_path} {frame_count}\n")


def find_peak_box(data):
    """
    Find peak points in a detection tensor and categorize detected objects by size.

    :param data: Detection tensor (shape: 20x20 grid with detection channels).
    :return: List of detected objects with position and size category.
    """
    det_data = np.zeros((22, 22, 7))
    det_data[1:21, 1:21] = data
    det_data[19:21, 1:21, 0] -= 0.1
    res = []
    for i in range(1, 21):
        for j in range(1, 21):
            if det_data[i, j, 0] > 0.9 or (
                    det_data[i, j, 0] > 0.4
                    and det_data[i, j, 0] > det_data[i, j - 1, 0]
                    and det_data[i, j, 0] > det_data[i, j + 1, 0]
                    and det_data[i, j, 0] > det_data[i + 1, j + 1, 0]
                    and det_data[i, j, 0] > det_data[i - 1, j + 1, 0]
                    and det_data[i, j, 0] > det_data[i + 1, j - 1, 0]
                    and det_data[i, j, 0] > det_data[i + 1, j + 1, 0]
                    and det_data[i, j, 0] > det_data[i - 1, j, 0]
                    and det_data[i, j, 0] > det_data[i + 1, j, 0]
            ):
                res.append((i - 1, j - 1))

    # box_info = {"big": [], "medium": [], "small": [], "tiny": []}
    box_info = []

    for instance in res:
        i, j = instance
        box = np.array(det_data[i + 1, j + 1, 4:6])  # Extract width and height

        if box[0] > 3.0:  # Trucks, large vehicles
            box_info.append({"poi": (i, j), "box": box, "size": "big"})
        elif box[0] > 1.8:  # Regular cars
            box_info.append({"poi": (i, j), "box": box, "size": "medium"})
        elif box[0] > 0.7:  # Bikes or motorcycles
            box_info.append({"poi": (i, j), "box": box, "size": "small"})
        else:  # Pedestrians
            box_info.append({"poi": (i, j), "box": box, "size": "tiny"})

    return box_info


def estimate_other_actor_data(det_data, compass):
    """
    Estimate positions, speeds, and orientations of other actors based on detection data.

    :param det_data: Detection tensor containing bounding box and movement information.
    :param compass: Compass angle of the ego vehicle (in radians).
    :return: List of dictionaries representing estimated actor information.
    """
    actor_data = []
    det_data = det_data * reweight_array
    box_info = find_peak_box(det_data)
    ego_yaw = compass_to_yaw(compass)

    for data in box_info:
        i, j = data["poi"]
        size = data["size"]

        speed = max(4, det_data[i, j, 6]) if size == "tiny" else det_data[i, j, 6]

        center_x, center_y = 17.5 - i, j - 9.5
        radius, relative_angle = cartesian_to_polar(center_x, center_y)

        angle = ego_yaw + relative_angle
        relative_x, relative_y = polar_to_cartesian(radius, angle)

        yaw = det_data[i, j, 3] * 180 - 90

        actor_data.append({
            "loc": {"radius": radius, "angle": ego_yaw + relative_angle, "rel_x": relative_x, "rel_y": relative_y},
            "ori": {"yaw": yaw},
            "box": data["box"],
            "vel": speed,
            "size": size,
        })

    return actor_data
