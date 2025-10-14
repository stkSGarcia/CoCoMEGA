import importlib
import sys
import traceback
from pathlib import Path
import impl.config as cfg

import scenic
import scenic.core.errors as errors
from agents.navigation.local_planner import RoadOption
from leaderboard.autoagents.agent_wrapper import AgentWrapper, AgentError
from leaderboard.envs.sensor_interface import SensorConfigurationInvalid, SensorReceivedNoData
from scenic.core.simulators import SimulationCreationError
from scenic.simulators.carla.simulator import CarlaSimulator, CarlaSimulation
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime

from impl.ads.scenario.scenario_definition import ScenarioDefinition
from impl.ads.utils.carla_utils import dict_to_transform


def generate_scenario(scenic_string):
    return errors.callBeginningScenicTrace(lambda: scenic.scenarioFromString(scenic_string, model="model"))


def generate_scene(scenario, max_iteration=2000):
    return errors.callBeginningScenicTrace(lambda: scenario.generate(maxIterations=max_iteration))


def get_simulator(scenario):
    return errors.callBeginningScenicTrace(scenario.getSimulator)


def run_simulation(simulator, scene, time=None):
    try:
        simulation = errors.callBeginningScenicTrace(lambda: simulator.simulate(
            scene,
            maxSteps=time,
            maxIterations=1,
        ))
    except SimulationCreationError as e:
        print(f"Failed to create simulation: {e}")
        return None
    return simulation


class ScenicSimulator(CarlaSimulator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)

    def createSimulation(self, scene, verbosity=0):
        self.scenario_number += 1
        return ScenicSimulation(
            scene,
            self.client,
            self.tm,
            self.timestep,
            render=self.render,
            record=self.record,
            scenario_number=self.scenario_number,
            verbosity=verbosity
        )


class ScenicSimulation(CarlaSimulation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        agent_path = Path(cfg.CONFIG["interfuser"]["agent_path"])
        sys.path.insert(0, str(agent_path.parent))
        self.module_agent = importlib.import_module(agent_path.stem)
        for obj in self.objects:
            if obj.rolename == "hero":
                self.ego = obj
                break

    def load_agent(self):
        # --- begin: hard reset of sensor interface state per run ---
        try:
            # If leaderboard SensorInterface keeps globals/singletons, reset them
            from leaderboard.envs.sensor_interface import SensorInterface
            if hasattr(SensorInterface, "_SensorInterface__sensors"):
                SensorInterface._SensorInterface__sensors.clear()
            if hasattr(SensorInterface, "_SensorInterface__queues"):
                SensorInterface._SensorInterface__queues.clear()
        except Exception:
            pass
        # --- end: hard reset ---

        try:
            agent_class_name = getattr(self.module_agent, "get_entry_point")()
            agent_config_path = cfg.CONFIG["interfuser"]["versions"][0]["agent_config"]
            agent_instance = getattr(self.module_agent, agent_class_name)(agent_config_path)

            AgentWrapper.validate_sensor_configuration(agent_instance.sensors(), agent_instance.track, "SENSORS")

            _, gps_route, route, _ = ScenarioDefinition._build_trajectory({
                "start": {
                    "x": self.ego.position.x,
                    "y": self.ego.position.y,
                    "z": 0,
                    "yaw": self.ego.heading,
                    "speed": self.ego.speed,
                },
                "direction": "forward",
            })

            gps_route = [(t, getattr(RoadOption, name)) for t, name in gps_route]
            route = [(dict_to_transform(t), getattr(RoadOption, name)) for t, name in route]
            CarlaDataProvider.set_ego_vehicle_route([(t.location, ro) for t, ro in route])
            agent_instance.set_global_plan(gps_route, route)

            CarlaDataProvider._carla_actor_pool[self.ego.carlaActor.id] = self.ego.carlaActor
            CarlaDataProvider.register_actor(self.ego.carlaActor)

            GameTime.restart()
            self.agent = AgentWrapper(agent_instance)
            self.agent.setup_sensors(self.ego.carlaActor, False)

        except SensorConfigurationInvalid as e:
            # The sensors are invalid -> set the ejecution to rejected and stop
            print("\n\033[91mThe sensor's configuration used is invalid:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()
            crash_message = "Agent's sensors were invalid"
            entry_status = "Rejected"
            return

        except Exception as e:
            # The agent setup has failed -> start the next route
            print("\n\033[91mCould not set up the required agent:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()
            crash_message = "Agent couldn't be set up"
            return

    def run(self, maxSteps):
        self.load_agent()
        super().run(maxSteps)

    def get_ego_control(self):
        timestamp = None
        if self.world:
            snapshot = self.world.get_snapshot()
            if snapshot:
                timestamp = snapshot.timestamp
        if timestamp:
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()

            try:
                ego_action = self.agent()
            # Special exception inside the agent that isn't caused by the agent
            except SensorReceivedNoData as e:
                raise RuntimeError(e)
            except Exception as e:
                raise AgentError(e)
            return ego_action
        return None

    def executeActions(self, allActions):
        super().executeActions(allActions)

        # Apply control updates which were accumulated while executing the actions
        for obj in self.agents:
            if obj.rolename == "hero":
                ctrl = self.get_ego_control()
            else:
                ctrl = obj._control
            if ctrl is not None:
                obj.carlaActor.apply_control(ctrl)
                obj._control = None


if __name__ == "__main__":
    scenic_string_ = """
param map = localPath('conf/maps/Town01.xodr')
param carla_map = 'Town01'
param address = '172.30.32.1'
param timeout = 1000
model scenic.simulators.carla.model

## CONSTANTS
EGO_MODEL = "vehicle.lincoln.mkz2017"
EGO_SPEED = 10

## DEFINING BEHAVIORS
# EGO BEHAVIOR: Follow lane, and brake after passing a threshold distance to the leading car
behavior EgoBehavior(speed=10):
    do FollowLaneBehavior(speed)

## DEFINING SPATIAL RELATIONS
# Please refer to scenic/domains/driving/roads.py how to access detailed road infrastructure
# 'network' is the 'class Network' object in roads.py

# make sure to put '*' to uniformly randomly select from all elements of the list, 'lanes'
lane = Uniform(*network.lanes)

start = OrientedPoint on lane.centerline
ego = Car at start,
    with blueprint EGO_MODEL,
    with behavior EgoBehavior(EGO_SPEED),
    with rolename "hero"

debris1 = Debris following roadDirection for Range(10, 20)
debris2 = Debris following roadDirection from debris1 for Range(5, 10)
debris3 = Debris following roadDirection from debris2 for Range(5, 10)

require (distance to intersection) > 50
terminate when (distance from debris3 to ego) > 10 and (distance to start) > 50
    """

    scenario_ = generate_scenario(scenic_string_)
    simulator_ = get_simulator(scenario_)
    scene_, _ = generate_scene(scenario_)
    simulation_ = run_simulation(simulator_, scene_, 2000)
    print(simulation_.result)
