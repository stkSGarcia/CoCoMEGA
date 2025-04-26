import itertools
import logging
import os
import pickle
import shutil
import subprocess
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from multiprocessing import Manager, Process

import pandas as pd
from deap import tools
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from impl import config as cfg
from impl.scenario.exceptions import InvalidScenarioDefinitionError
from impl.scenario.scenario_definition import ScenarioDefinition
from impl.utils.carla_utils import initialize_carla
from impl.utils.process_utils import run_silently

arguments = [
    ("SCENARIOS", "scenarios",
     os.path.join(cfg.CONFIG["interfuser"]["repo"], "leaderboard/data/scenarios/town05_all_scenarios.json")),
    ("ROUTES", "routes",
     os.path.join(cfg.CONFIG["interfuser"]["repo"], "leaderboard/data/training_routes/routes_town05_long.xml")),
    ("REPETITIONS", "repetitions", 1),
    ("CHALLENGE_TRACK_CODENAME", "track", "SENSORS"),
    # ("CHECKPOINT_ENDPOINT", "checkpoint", os.path.join(CONFIG["workspace"]["sim_result"], "checkpoint.json")),
    ("TEAM_AGENT", "agent", "impl/scenario/interfuser_agent.py"),
    ("TEAM_CONFIG", "agent_config", "impl/scenario/interfuser_config_v1.py"),
    ("DEBUG_CHALLENGE", "debug", 0),
    ("RESUME", "resume", False),
    ("SAVE_PATH", None, None),
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
evaluated_scenarios = Manager().dict()
carla_host = carla_port = tm_port = gpu_device = None


def _init_carla(instance_configs):
    """
    Initialize CARLA connection based on provided instance configuration.

    This sets host, port, traffic manager port, and GPU device.
    Cleans previous CARLA actors and prepares environment.

    :param instance_configs: Queue providing instance settings.
    """
    global carla_host, carla_port, tm_port, gpu_device
    carla_host, carla_port, tm_port, gpu_device = instance_configs.get(timeout=10)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_device)
    CarlaDataProvider.cleanup()
    initialize_carla(carla_host, carla_port, tm_port, gpu_device)


def run_free_environments(confs):
    """
    Run multiple environment instances in parallel or sequentially.

    :param confs: List of configuration dictionaries for each environment run.
    :return: A list of results from each environment.
    """
    if cfg.CONFIG["runtime"]["parallel"]:
        process_configs = Manager().Queue()
        for instance in cfg.CONFIG["runtime"]["instances"]:
            process_configs.put((instance["host"], instance["port"], instance["tm_port"], instance["gpu_device"]))
        with ProcessPoolExecutor(max_workers=len(cfg.CONFIG["runtime"]["instances"]),
                                 initializer=_init_carla, initargs=(process_configs,)) as executor:
            results = executor.map(run_environment, confs)
    else:
        global carla_host, carla_port, tm_port, gpu_device
        instance = cfg.CONFIG["runtime"]["instances"][0]
        carla_host, carla_port, tm_port, gpu_device = (instance["host"], instance["port"],
                                                       instance["tm_port"], instance["gpu_device"])
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_device)
        CarlaDataProvider.cleanup()
        initialize_carla(carla_host, carla_port, tm_port, gpu_device)
        results = map(run_environment, confs)
        results = zip(*results)
    return results


def run_environment(conf):
    """
    Run a single simulation environment with the given configuration.

    Sets environment variables and executes a subprocess to run the simulation.

    :param conf: Configuration dictionary for one environment instance.
    :return: Process return code (:data:`0` for success).
    """
    global carla_host, carla_port, tm_port

    cp_path = os.path.join(conf["cp_root"], f"weather-{conf['weather']}", f"{conf['route_name']}.json")
    output_path = os.path.join(conf["output_root"], f"weather-{conf['weather']}")
    os.makedirs(os.path.dirname(cp_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    scenarios = os.path.join(cfg.CONFIG["interfuser"]["repo"], "leaderboard", "data", conf["scenario"])
    routes = os.path.join(cfg.CONFIG["interfuser"]["repo"], "leaderboard", "data", conf["route"])

    child_env = os.environ.copy()

    # Set environment variables as in the bash script
    child_env.update({
        "DATA_ROOT": str(cfg.CONFIG["workspace"]["runtime_data"]),
        "CARLA_ROOT": os.path.join(cfg.CONFIG["interfuser"]["repo"], "carla"),
        "CARLA_SERVER": os.path.join(cfg.CONFIG["interfuser"]["repo"], "carla", "CarlaUE4.sh"),
        "CARLA_WEATHER": str(conf["weather"]),
        "LEADERBOARD_ROOT": os.path.join(cfg.CONFIG["interfuser"]["repo"], "leaderboard"),
        "CHECKPOINT_ENDPOINT": cp_path,
        "SAVE_PATH": output_path,
        "TRAFFIC_SEED": "2000",
        "CARLA_SEED": "2000",
        "SCENARIOS": scenarios,
        "ROUTES": routes,
        "TM_PORT": str(tm_port),
        "PORT": str(carla_port),
        "HOST": carla_host,
        "CHALLENGE_TRACK_CODENAME": "SENSORS",
        "DEBUG_CHALLENGE": "0",
        "REPETITIONS": "1",
        "TEAM_AGENT": conf["agent_path"],
        "RESUME": "True",
        "COLLECTION_DELAY_LOWER": conf.get("collection_delay_lower", None),
        "COLLECTION_DELAY_UPPER": conf.get("collection_delay_upper", None),
        "COLLECTION_DURATION": conf.get("collection_duration", None),
        "COLLECTION_INTERVAL": conf.get("collection_interval", None),
        "COLLECTION_ROUTE_LIMIT": conf.get("collection_route_limit", None),
        # Add RECORD_PATH if needed
        # "RECORD_PATH": "path/to/record",
    })

    child_env["PYTHONPATH"] = os.pathsep.join(sys.path)

    command = (
        # f"{sys.executable} {os.path.join(CONFIG['interfuser']['repo'], 'leaderboard/leaderboard/leaderboard_evaluator.py')}"
        f"{sys.executable} {os.path.join('impl', 'scenario', 'leaderboard_evaluator.py')}"
        f" --scenarios {scenarios}"
        f" --routes {routes}"
        f" --repetitions 1"
        f" --track SENSORS"
        f" --checkpoint {cp_path}"
        f" --agent {conf['agent_path']}"
        f" --agent-config {conf['agent_config']}"
        f" --debug 0"
        f" --resume True"
        f" --port {carla_port}"
        f" --host {carla_host}"
        f" --trafficManagerPort {tm_port}"
        f" --carlaProviderSeed 2000"
        f" --trafficManagerSeed 2000"
    )

    process = subprocess.run(command, env=child_env, check=True, shell=True, text=True, stdout=None, stderr=None)

    return process.returncode


def run_scenario(scenario: ScenarioDefinition, agent_name, rerun=False, additional_config=None):
    """
    Run a single :class:`ScenarioDefinition` through CARLA simulation.

    :param scenario: :class:`ScenarioDefinition` object to simulate.
    :param agent_name: Name of the agent configuration to use.
    :param rerun: Whether to force rerunning even if results exist.
    :param additional_config: Additional parameters to pass.
    :return: (Result `Dataframe`, whether simulation was newly executed).
    """
    global carla_host, carla_port, tm_port, gpu_device
    assert carla_host is not None and carla_port is not None and tm_port is not None and gpu_device is not None

    agent_config = [conf["agent_config"] for conf in cfg.CONFIG["interfuser"]["versions"] if conf["name"] == agent_name]
    if len(agent_config) == 0:
        raise ValueError(f"Agent not defined: \"{agent_name}\".")
    agent_config = agent_config[0]
    setattr(config, "agent_config", agent_config)

    if not rerun and agent_name in evaluated_scenarios:
        for evaluated_scenario, evaluated_result in evaluated_scenarios[agent_name]:
            if scenario == evaluated_scenario:
                logger.debug(f"Scenario evaluated for agent {agent_name}: {evaluated_scenario}.")
                return evaluated_result, False

    setattr(config, "host", carla_host)
    setattr(config, "port", carla_port)
    setattr(config, "trafficManagerPort", tm_port)
    setattr(config, "gpu_device", gpu_device)
    setattr(config, "additional_config", additional_config)
    logger.debug(f"Starting simulation, scenario id: {scenario.id_}, carla instance: {config.host}:{config.port}, "
                 f"traffic manager port: {config.trafficManagerPort} on cuda device {config.gpu_device}.")
    logger.debug(scenario)

    is_successful = False
    for _ in range(1 + cfg.CONFIG["simulation"]["retry_times"]):
        evaluator = None
        try:
            evaluator = ScenarioEvaluator(scenario, config)
            evaluator.run(config)
            is_successful = True
            break
        except InvalidScenarioDefinitionError as e:
            #logger.error(f"Scenario failed: {scenario}, message: {e}.")
            logger.error(f"Scenario failed, message: {e}.")
            is_successful = False
            break
        except Exception as e:
            # logger.error(f"Scenario failed: {scenario}, message: {e}.")
            logger.error(f"Scenario failed, message: {e}.")
            if cfg.CONFIG['debug']:
                traceback.print_exc()
            is_successful = False
        finally:
            del evaluator
    if not is_successful:
        return None, False

    result_path = cfg.CONFIG["workspace"]["sim_result"] / f"{scenario.id_}.csv"
    if not result_path.exists():
        logger.warning(f"Scenario results cannot be found: {scenario.id_}.")
        return None, False

    try:
        result = pd.read_csv(result_path)
        result.set_index(result.columns[0], inplace=True)
    except Exception as e:
        logger.error(f"Scenario results cannot be read: {scenario.id_}, message: {e}.")
        return None, False
    if agent_name not in evaluated_scenarios:
        evaluated_scenarios[agent_name] = []
    evaluated_scenarios[agent_name].append((scenario, result))
    return result, True


def run_scenarios(scenarios, agent_name="v1", rerun=False, additional_confs=None):
    """
    Run a batch of scenarios either sequentially or in parallel.

    :param scenarios: List of :class:`ScenarioDefinition` objects to simulate.
    :param agent_name: Name of agent configuration to use.
    :param rerun: Whether to rerun already evaluated scenarios.
    :param additional_confs: Optional additional configurations per scenario.
    :return: (List of results, number of successful runs).
    """
    if additional_confs is None:
        additional_confs = list(itertools.repeat(None, len(scenarios)))
    assert len(additional_confs) == len(scenarios)
    if cfg.CONFIG["simulation"]["parallel"]:
        process_configs = Manager().Queue()
        for instance in cfg.CONFIG["simulation"]["instances"]:
            process_configs.put((instance["host"], instance["port"], instance["tm_port"], instance["gpu_device"]))
        # FIXME: Traffic manager may cause memory leak.
        # https://github.com/carla-simulator/carla/issues/3584
        # https://github.com/carla-simulator/carla/issues/3540
        # https://github.com/carla-simulator/leaderboard/issues/81
        # https://github.com/carla-simulator/carla/issues/2781
        with ProcessPoolExecutor(max_workers=len(cfg.CONFIG["simulation"]["instances"]),
                                 initializer=_init_carla, initargs=(process_configs,)) as executor:
            results = executor.map(run_scenario, scenarios, itertools.repeat(agent_name, len(scenarios)),
                                   itertools.repeat(rerun, len(scenarios)), additional_confs)
    else:
        global carla_host, carla_port, tm_port, gpu_device
        instance = cfg.CONFIG["simulation"]["instances"][0]
        carla_host, carla_port, tm_port, gpu_device = (instance["host"], instance["port"],
                                                       instance["tm_port"], instance["gpu_device"])
        results = map(run_scenario, scenarios, itertools.repeat(agent_name, len(scenarios)),
                      itertools.repeat(rerun, len(scenarios)), additional_confs)

    results, is_executed = zip(*results)
    return results, is_executed.count(True)


def run_solutions(file: str, top: int = -1, verbose=True, agent_name="v1", record_video=False):
    """
    Load and run scenarios from a saved solution file.

    Each solution typically contains a source and a follow-up scenario.

    :param file: Path to the pickle file containing solutions.
    :param top: Number of top solutions to run (:data:`-1` runs all).
    :param verbose: If :data:`True`, run with output logs; otherwise silent.
    :param agent_name: Agent name to use.
    :param record_video: Whether to record video outputs for the runs.
    """
    solution_name = file.split("/")[-1].split(".")[0]
    with open(file, "rb") as f:
        solutions = pickle.load(f)
    if top == -1:
        top = len(solutions)
    solutions = tools.selBest([ind for ind in solutions if ind.is_violated], top)
    additional_confs = None
    for i, (source, perturbations) in enumerate(solutions):
        if record_video:
            solution_path = cfg.CONFIG["workspace"]["recordings"] / f"{solution_name}-{i + 1}"
            if already_recorded(solution_path):
                logger.info(f"Solution {solution_name}-{i + 1} is already recorded, skipping...")
                continue
            elif solution_path.exists():
                shutil.rmtree(solution_path)
            additional_confs = [
                {"recording_save_path": solution_path / "source"},
                {"recording_save_path": solution_path / "follow-up"},
            ]
        source.id_ = f"top{i + 1}_source"
        follow_up = deepcopy(source)
        follow_up.id_ = f"top{i + 1}_follow-up"
        perturbations.perturb(follow_up)
        logger.info(f"Running solution {solution_name}-{i + 1}...")
        if verbose:
            run_process = Process(target=run_scenarios, args=([source, follow_up], agent_name, True, additional_confs))
        else:
            run_process = Process(target=run_silently,
                                  args=(run_scenarios, [source, follow_up], agent_name, True, additional_confs))
        run_process.start()
        run_process.join()


def already_recorded(solution_path):
    """
    Check if a recording already exists for a given solution path.

    :param solution_path: Path where the source and follow-up recordings are expected.
    :return: :data:`True` if already recorded, :data:`False` otherwise.
    """
    try:
        if len(os.listdir(solution_path / "source")) > 0 and len(os.listdir(solution_path / "follow-up")) > 0:
            return True
    except FileNotFoundError:
        pass
    return False
