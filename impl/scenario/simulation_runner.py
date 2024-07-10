import itertools
import logging
import os
import pickle
import traceback
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from multiprocessing import Manager

import carla
import pandas as pd
from deap import tools
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from impl.config import CONFIG
from impl.scenario.exceptions import InvalidScenarioDefinitionError
from impl.scenario.scenario_definition import ScenarioDefinition
from impl.utils.carla_utils import initialize_carla

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
    if env is not None and v is not None:
        os.environ[env] = str(v)

from impl.scenario.interfuser_scenario_evaluator import ScenarioEvaluator

logger = logging.getLogger(__name__)
config = type("", (object,), {arg: value for _, arg, value in arguments})()
evaluated_scenarios = Manager().list()
carla_host = carla_port = tm_port = cuda_device = None


def _init_carla(instance_configs):
    global carla_host, carla_port, tm_port, cuda_device
    carla_host, carla_port, tm_port, cuda_device = instance_configs.get(timeout=10)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_device)
    CarlaDataProvider.cleanup()
    initialize_carla(carla_host, carla_port, tm_port)


def run_scenario(scenario: ScenarioDefinition, rerun=False, process_configs=None):
    """Run a scenario defined in ScenarioDefinition.

    @return: The simulation result and whether the scenario was actually executed.
    """
    global carla_host, carla_port, tm_port, cuda_device
    assert carla_host is not None and carla_port is not None and tm_port is not None and cuda_device is not None

    if not rerun:
        for evaluated_scenario, evaluated_result in evaluated_scenarios:
            if scenario == evaluated_scenario:
                logger.debug(f"Scenario evaluated: {evaluated_scenario}.")
                return evaluated_result, False

    setattr(config, "host", carla_host)
    setattr(config, "port", carla_port)
    setattr(config, "trafficManagerPort", tm_port)
    setattr(config, "cuda_device", cuda_device)
    logger.debug(f"Starting simulation, scenario id: {scenario.id_}, carla instance: {config.host}:{config.port}, "
                 f"traffic manager port: {config.trafficManagerPort} on cuda device {config.cuda_device}.")
    logger.debug(scenario)

    is_successful = False
    for _ in range(1 + CONFIG["simulation"]["retry_times"]):
        evaluator = None
        try:
            evaluator = ScenarioEvaluator(scenario, config)
            evaluator.run(config)
            is_successful = True
            break
        except InvalidScenarioDefinitionError as e:
            logger.error(f"Scenario failed: {scenario}, message: {e}.")
            is_successful = False
            break
        except Exception as e:
            logger.error(f"Scenario failed: {scenario}, message: {e}.")
            if CONFIG['debug']:
                traceback.print_exc()
            is_successful = False
        finally:
            del evaluator
    if not is_successful:
        return None, False

    result_path = os.path.join(CONFIG["workspace"]["sim_result"], f"{scenario.id_}.csv")
    if not os.path.exists(result_path):
        logger.warning(f"Scenario results cannot be found: {scenario.id_}.")
        return None, False

    result = pd.read_csv(result_path)
    result.set_index(result.columns[0], inplace=True)
    evaluated_scenarios.append((scenario, result))
    return result, True


def run_scenarios(scenarios, rerun=False):
    """Run scenarios.

    @return: A list of simulation results and the number of simulations.
    """
    if CONFIG["simulation"]["parallel"]:
        process_configs = Manager().Queue()
        for instance in CONFIG["simulation"]["docker"]["instances"]:
            process_configs.put((instance["host"], instance["port"], instance["tm_port"], instance["gpu_device"]))
        # FIXME: Traffic manager may cause memory leak.
        # https://github.com/carla-simulator/carla/issues/3584
        # https://github.com/carla-simulator/carla/issues/3540
        # https://github.com/carla-simulator/leaderboard/issues/81
        # https://github.com/carla-simulator/carla/issues/2781
        with ProcessPoolExecutor(max_workers=len(CONFIG["simulation"]["docker"]["instances"]),
                                 initializer=_init_carla, initargs=(process_configs,)) as executor:
            results = executor.map(run_scenario, scenarios, itertools.repeat(rerun, len(scenarios)))
    else:
        global carla_host, carla_port, tm_port, cuda_device
        instance = CONFIG["simulation"]["docker"]["instances"][0]
        carla_host, carla_port, tm_port, cuda_device = (instance["host"], instance["port"],
                                                        instance["tm_port"], instance["gpu_device"])
        results = map(run_scenario, scenarios, itertools.repeat(rerun, len(scenarios)))

    results, is_executed = zip(*results)
    return results, is_executed.count(True)


def run_solutions(file: str, top: int = 1):
    """Run scenarios from a solution file.

    @param file: The solution file.
    @param top: Number of top scenarios to run.
    """
    with open(file, "rb") as f:
        solutions = pickle.load(f)
    solutions = tools.selBest([ind for ind in solutions if ind.is_violated], top)
    for i, (source, perturbations) in enumerate(solutions):
        source.id_ = f"top{i + 1}_source"
        follow_up = deepcopy(source)
        follow_up.id_ = f"top{i + 1}_follow-up"
        perturbations.perturb(follow_up)
        run_scenarios([source, follow_up], rerun=True)
