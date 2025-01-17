import os.path
import yaml
import logging

from impl.config import CONFIG
from impl.scenario.scenario_definition import ScenarioDefinition, Vehicle, Walker, Static
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
    confs = []
    for town in towns:
        for weather in weathers:
            for route, scenario in towned_routes[town].items():
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
    # TODO assign trajectory
    scenario_def.weather = rt_data['weather'] # TODO add weather  to rt_data
    scenario_def.town = CarlaDataProvider.get_map().name
    ego_vechile_coordinates = rt_data['gps'] # gps should be rotated 90 degrees clockwise

    for _id, other_actor in rt_data['other_actors'].items():
        relative_position = (
        other_actor['loc'][0] - ego_vechile_coordinates[0], other_actor['loc'][1] - ego_vechile_coordinates[1])
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
