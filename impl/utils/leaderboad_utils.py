import os
import yaml
import logging

from impl.config import CONFIG
from impl.scenario.scenario_definition import ScenarioDefinition, Vehicle, Walker, Static
from impl.utils.carla_utils import traj_interpolation, location_to_dict, compass_to_yaw
from impl.utils.math_utils import cartesian_to_polar, vector_norm

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
    town = route.split('/')[1].split('_')[1]

    if town not in towned_routes:
        towned_routes[town] = {}
    towned_routes[town][route] = scenario


def get_enviroment_confs():
    weathers = CONFIG["runtime"]["weathers"]
    towns = CONFIG["runtime"]["towns"]
    route_types = CONFIG["runtime"]["routes"]
    confs = []
    for town in towns:
        for weather in weathers:
            for route, scenario in towned_routes[town].items():
                route_type = route.split('_')[-1][:-4]
                if route_type not in route_types:
                    continue
                conf = {
                    "town": town,
                    "weather": weather,
                    "route": route,
                    "route_name": route.split('/')[1].split('.')[0],
                    "scenario": scenario,
                }
                confs.append(conf)
    return confs


def make_yamls():
    conf = CONFIG["data_collection"].copy()
    os.makedirs(CONFIG["data_collection"]["yaml_root"], exist_ok=True)
    for weather in CONFIG["runtime"]["weathers"]:
        conf["weather"] = weather
        file_path = os.path.join(CONFIG["data_collection"]["yaml_root"], f"weather-{weather}.yaml")
        try:
            with open(file_path, 'w') as file:
                yaml.dump(conf, file, default_flow_style=False, sort_keys=True)
        except Exception as e:
            logger.error(f"An error occurred while saving the YAML file {file_path}")
            raise e


def vectorize_runtime_data(rt_data):
    scenario_def = ScenarioDefinition._generate_empty_scenario()
    trajectory = rt_data["trajectory"]
    scenario_def.set_trajectory({
        "town": rt_data["town"],
        **trajectory,
    })
    scenario_def.weather = int(rt_data["weather"])
    scenario_def.set_brightness(rt_data["brightness"])
    scenario_def.town = rt_data["town"]

    for _id, other_actor in rt_data['other_actors'].items():
        relative_position = (
            other_actor['loc'][0] - trajectory["start"]["x"], other_actor['loc'][1] - trajectory["start"]["y"])
        radius, angle = cartesian_to_polar(*relative_position)
        yaw = other_actor['ori']['yaw']
        if other_actor['tpe'] == 0:
            speed = vector_norm(other_actor['vel'])
            scenario_def.vehicles.append(Vehicle(radius=radius, angle=angle, yaw=yaw, model=0, speed=speed))
        if other_actor['tpe'] == 1:
            speed = vector_norm(other_actor['vel'])
            scenario_def.walkers.append(Walker(radius=radius, angle=angle, yaw=yaw, model=0, speed=speed))
        if other_actor['tpe'] == 2:
            scenario_def.statics.append(Static(radius=radius, angle=angle, yaw=yaw, model=0))
    return scenario_def


def create_dataset_index(dataset_root, weathers=None, towns=None):
    """
        Creates a dataset_index.txt file that lists the training/evaluation data.

        Args:
            dataset_root (str): Path to the root directory of the dataset.
        """
    index_file_path = os.path.join(dataset_root, 'dataset_index.txt')

    with open(index_file_path, 'w') as index_file:
        for root, dirs, files in os.walk(dataset_root):
            # Filter for directories containing relevant data frames
            if 'rgb_front' in dirs:
                data_weather = int(os.path.basename(os.path.dirname(root)).split('-')[-1])
                data_town = int(root.split('/')[-1].split('_')[1][4:])
                if (weathers and data_weather not in weathers) or (towns and data_town not in towns): continue
                # Count the number of frames in the directory
                rgb_dir_path = os.path.join(root, 'rgb_front')
                frame_count = len(
                    [file for file in os.listdir(rgb_dir_path) if os.path.isfile(os.path.join(rgb_dir_path, file))])

                # Write the relative path and frame count to the index file
                relative_path = os.path.relpath(root, dataset_root)
                index_file.write(f"{relative_path} {frame_count}\n")
