import csv
import os
import sys
import traceback

from impl.config import CONFIG
# from impl.scenario.interfuser_scenario_evaluator import ScenarioEvaluator
from impl.scenario.scenario_definition import ScenarioDefinition

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
    ("CHECKPOINT_ENDPOINT", "checkpoint", os.path.join(CONFIG["result"], "sample_result.json")),
    ("TEAM_AGENT", "agent", os.path.join(root, "leaderboard/team_code/interfuser_agent.py")),
    ("TEAM_CONFIG", "agent-config", os.path.join(root, "leaderboard/team_code/interfuser_config.py")),
    ("DEBUG_CHALLENGE", "debug", 0),
    ("RESUME", "resume", True),
    ("HOST", "host", "172.30.32.1"),
    ("PORT", "port", 2000),
    ("CARLA_ROOT", None, os.path.join(root, "carla")),
    ("CARLA_SERVER", None, os.path.join(root, "carla/CarlaUE4.sh")),
    ("LEADERBOARD_ROOT", None, os.path.join(root, "leaderboard")),
    ("SAVE_PATH", None, os.path.join(CONFIG["workspace"], "data/eval")),
    ("TM_PORT", None, 2500),
]

for env, _, value in arguments:
    if env is not None: os.environ[env] = str(value)


class SimulationRunner:
    def __init__(self):
        self.arguments = {arg: str(v) for _, arg, v in arguments}

    def run(self, scenario: ScenarioDefinition):
        # try:
        #     evaluator = ScenarioEvaluator(scenario, self.arguments)
        #     evaluator.run(self.arguments)
        # except Exception as e:
        #     traceback.print_exc()
        # finally:
        #     del evaluator

        # results = SimulationRunner._read_record(os.path.join(CONFIG["result"], f"{scenario.id_}.csv"))
        results = SimulationRunner._read_record(os.path.join(CONFIG["result"], "4c5d1c3708994a4fafbe53bc47ff82f9.csv"))
        setattr(scenario, "results", results)
        return scenario

    @staticmethod
    def _read_record(path):
        results = {}
        with open(path) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                results[row[0]] = (row[1], row[2], row[3])
        return results


simulation_runner = SimulationRunner()
