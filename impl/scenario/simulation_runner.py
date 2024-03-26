import logging
import os
import subprocess
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from impl.config import CONFIG

root = "impl/scenario"
for p in [
    os.path.join(root, "carla/PythonAPI"),
    os.path.join(root, "carla/PythonAPI/carla"),
    os.path.join(root, "carla/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg"),
    os.path.join(root, "leaderboard"),
    os.path.join(root, "leaderboard", "team_code"),
    os.path.join(root, "scenario_runner"),
]:
    sys.path.append(p)

arguments = [
    ("SCENARIOS", "scenarios", os.path.join(root, "leaderboard/data/scenarios/town05_all_scenarios.json")),
    ("ROUTES", "routes", os.path.join(root, "leaderboard/data/training_routes/routes_town05_long.xml")),
    ("REPETITIONS", "repetitions", 1),
    ("CHALLENGE_TRACK_CODENAME", "track", "SENSORS"),
    ("CHECKPOINT_ENDPOINT", "checkpoint",
     os.path.join(CONFIG["workspace"], CONFIG["simulation"]["result"], "sample_result.json")),
    ("TEAM_AGENT", "agent", os.path.join(root, "leaderboard/team_code/interfuser_agent.py")),
    ("TEAM_CONFIG", "agent_config", os.path.join(root, "leaderboard/team_code/interfuser_config.py")),
    ("DEBUG_CHALLENGE", "debug", 0),
    ("RESUME", "resume", True),
    ("HOST", "host", CONFIG["simulation"]["host"]),
    ("CARLA_ROOT", None, os.path.join(root, "carla")),
    ("CARLA_SERVER", None, os.path.join(root, "carla/CarlaUE4.sh")),
    ("LEADERBOARD_ROOT", None, os.path.join(root, "leaderboard")),
    ("SAVE_PATH", None, os.path.join(CONFIG["workspace"], CONFIG["simulation"]["save"])),
    ("TM_PORT", None, 2500),
    (None, "trafficManagerPort", "2500"),
    (None, "trafficManagerSeed", "1"),
    (None, "carlaProviderSeed", "2000"),
    (None, "record", ""),
    (None, "timeout", "6000.0"),
]

for env, _, v in arguments:
    if env is not None: os.environ[env] = str(v)

from impl.scenario.scenario_definition import ScenarioDefinition
from impl.scenario.interfuser_scenario_evaluator import ScenarioEvaluator

logger = logging.getLogger(__name__)
config = type("", (object,), {arg: value for _, arg, value in arguments})()
evaluated = {}


def run_scenario(scenario: ScenarioDefinition, port):
    if scenario.id_ in evaluated: return evaluated[scenario.id_]
    setattr(config, "port", port)
    logger.debug(f"Starting simulation, scenario id: {scenario.id_}, carla port: {config.port}.")
    logger.debug(scenario)
    # carla_ps = subprocess.Popen([CONFIG["simulation"]["carla"], f"-carla-port={config.port}"])  # TODO: run containers
    try:
        evaluator = ScenarioEvaluator(scenario, config)
        evaluator.run(config)
    except Exception:
        traceback.print_exc()
    finally:
        # carla_ps.kill()
        del evaluator

    results = pd.read_csv(os.path.join(CONFIG["workspace"], CONFIG["simulation"]["result"], f"{scenario.id_}.csv"))
    results.set_index(results.columns[0], inplace=True)
    evaluated[scenario.id_] = results
    return results


def run_scenarios(scenarios):
    ports = [2000 + i for i in range(len(scenarios))]
    # with ProcessPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
    #     results = executor.map(run_scenario, scenarios, ports)
    results = map(run_scenario, scenarios, [2000 for _ in range(len(scenarios))])
    return list(results)
