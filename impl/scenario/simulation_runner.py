import itertools
import logging
import os
import traceback
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Queue, Manager

import pandas as pd

from impl.config import CONFIG
from impl.scenario.docker_utils import restart_carla
from impl.scenario.exceptions import InvalidScenarioDefinitionError
from impl.scenario.scenario_definition import ScenarioDefinition

arguments = [
    ("SCENARIOS", "scenarios",
     os.path.join(CONFIG["simulation"]["repo"], "leaderboard/data/scenarios/town05_all_scenarios.json")),
    ("ROUTES", "routes",
     os.path.join(CONFIG["simulation"]["repo"], "leaderboard/data/training_routes/routes_town05_long.xml")),
    ("REPETITIONS", "repetitions", 1),
    ("CHALLENGE_TRACK_CODENAME", "track", "SENSORS"),
    ("CHECKPOINT_ENDPOINT", "checkpoint", os.path.join(CONFIG["workspace"]["sim_result"], "checkpoint.json")),
    ("TEAM_AGENT", "agent", "impl/scenario/interfuser_agent.py"),
    ("TEAM_CONFIG", "agent_config", "impl/scenario/interfuser_config.py"),
    ("DEBUG_CHALLENGE", "debug", 0),
    ("RESUME", "resume", False),
    ("SAVE_PATH", None, CONFIG["workspace"]["sim_save"]),
    (None, "trafficManagerSeed", "1"),
    (None, "carlaProviderSeed", "2000"),
    (None, "record", ""),
    (None, "timeout", "600.0"),
]

for env, _, v in arguments:
    if env is not None: os.environ[env] = str(v)

from impl.scenario.interfuser_scenario_evaluator import ScenarioEvaluator

logger = logging.getLogger(__name__)
config = type("", (object,), {arg: value for _, arg, value in arguments})()

evaluated_scenarios = Manager().list()

carla_host = None
carla_port = None
tm_port = None
cuda_device = None


def _init_carla(instance_configs):
    assert instance_configs.qsize() > 0
    global carla_host, carla_port, tm_port, cuda_device
    carla_host, carla_port, tm_port, cuda_device = instance_configs.get()


def run_scenario(scenario: ScenarioDefinition, rerun=False):
    """Run a scenario defined in ScenarioDefinition.

    @return: The simulation result and whether the scenario was actually executed.
    """
    if not rerun:
        for evaluated_scenario, evaluated_result in evaluated_scenarios:
            if scenario == evaluated_scenario:
                logger.debug(f"Scenario evaluated: {evaluated_scenario}.")
                return evaluated_result, False

    global carla_host, carla_port, tm_port, cuda_device
    assert carla_host is not None and carla_port is not None and tm_port is not None and cuda_device is not None
    setattr(config, "host", carla_host)
    setattr(config, "port", carla_port)
    setattr(config, "trafficManagerPort", tm_port)
    setattr(config, "cuda_device", cuda_device)
    logger.debug(f"Starting simulation, scenario id: {scenario.id_}, carla instance: {config.host}:{config.port}, "
                 f"traffic manager port: {config.trafficManagerPort} on cuda device {config.cuda_device}.")
    logger.debug(scenario)

    is_successful = False
    for _ in range(1 + CONFIG["simulation"]["retry_times"]):
        try:
            evaluator = ScenarioEvaluator(scenario, config)
            evaluator.run(config)
            del evaluator
            is_successful = True
            break
        except InvalidScenarioDefinitionError as e:
            logger.error(f"Scenario failed: {scenario}, message: {e}.")
            # if CONFIG['debug']:
            #     traceback.print_exc()
            del evaluator
            is_successful = False
            break
        except Exception as e:
            logger.error(f"Scenario failed: {scenario}, message: {e}.")
            if CONFIG['debug']:
                traceback.print_exc()
            del evaluator
            if CONFIG['simulation']['docker']['enabled']:
                container_name = f"{CONFIG['simulation']['docker']['image']}-{carla_port}"
                restart_carla(container_name, carla_port)
            is_successful = False
    if not is_successful:
        return None, False

    result = pd.read_csv(os.path.join(CONFIG["workspace"]["sim_result"], f"{scenario.id_}.csv"))
    result.set_index(result.columns[0], inplace=True)
    evaluated_scenarios.append((scenario, result))
    return result, True


def run_scenarios(scenarios, rerun=False):
    """Run scenarios.

    @return: A list of simulation results and the number of simulations.
    """
    if CONFIG["simulation"]["parallel"]:
        queue = Queue()
        [queue.put((instance["host"], instance["port"], instance["tm_port"], instance["gpu_device"]))
         for instance in CONFIG["simulation"]["docker"]["instances"]]
        with ProcessPoolExecutor(max_workers=len(CONFIG["simulation"]["docker"]["instances"]),
                                 initializer=_init_carla, initargs=(queue,)) as executor:
            results = executor.map(run_scenario, scenarios, itertools.repeat(rerun, len(scenarios)))
    else:
        global carla_host, carla_port, tm_port, cuda_device
        instance = CONFIG["simulation"]["docker"]["instances"][0]
        carla_host = instance["host"]
        carla_port = instance["port"]
        tm_port = instance["tm_port"]
        cuda_device = instance["gpu_device"]
        results = map(run_scenario, scenarios, itertools.repeat(rerun, len(scenarios)))

    results, is_executed = zip(*results)
    return results, is_executed.count(True)
