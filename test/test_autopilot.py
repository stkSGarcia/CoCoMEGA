import test
from impl.scenario.carla_utils import initialize_carla
from impl.scenario.scenario_definition import ScenarioDefinition, Vehicle
from impl.scenario.simulation_runner import run_scenarios

config = test.CONFIG
initialize_carla()
scenario = ScenarioDefinition._generate_empty_scenario()
scenario.vehicles = [
    Vehicle(radius=40.0, angle=-5.2, yaw=90.0, model=1, speed=-100, autopilot=True),
    Vehicle(radius=3.5, angle=-60.0, yaw=-90.0, model=2, speed=70, autopilot=True)
]
run_scenarios([scenario], rerun=True)
