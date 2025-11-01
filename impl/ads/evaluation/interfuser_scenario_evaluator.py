#!/usr/bin/env python
# Copyright (c) 2018-2019 Intel Corporation.
# authors: German Ros (german.ros@intel.com), Felipe Codevilla (felipe.alcm@gmail.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
CARLA Challenge Evaluator Routes.

Provisional code to evaluate Autonomous Agents for the CARLA Autonomous Driving challenge.
"""
from __future__ import print_function

import traceback
import argparse
from argparse import RawTextHelpFormatter
import importlib
import os
import gc
import sys
import signal
import logging
import carla

from impl import config as cfg
from impl.ads.evaluation.exceptions import StoppingScenarioFailedError, SimulationError, InvalidScenarioDefinitionError, \
    LoadingScenarioFailedError, AgentSetupFailedError
from impl.ads.evaluation.scenario_manager import ScenarioManager

from impl.ads.scenario.scenario_definition import ScenarioDefinition
from srunner.scenariomanager.carla_data_provider import *
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog

from impl.ads.evaluation.route_scenario import RouteScenario
from leaderboard.envs.sensor_interface import SensorInterface, SensorConfigurationInvalid
from leaderboard.autoagents.agent_wrapper import AgentWrapper, AgentError

from impl.ads.utils.carla_utils import initialize_carla

logger = logging.getLogger(__name__)
sensors_to_icons = {
    'sensor.camera.rgb': 'carla_camera',
    'sensor.camera.semantic_segmentation': 'carla_camera',
    'sensor.camera.depth': 'carla_camera',
    'sensor.lidar.ray_cast': 'carla_lidar',
    'sensor.lidar.ray_cast_semantic': 'carla_lidar',
    'sensor.other.radar': 'carla_radar',
    'sensor.other.gnss': 'carla_gnss',
    'sensor.other.imu': 'carla_imu',
    'sensor.opendrive_map': 'carla_opendrive_map',
    'sensor.speedometer': 'carla_speedometer'
}


class ScenarioEvaluator(object):
    """
    The ScenarioEvaluator is responsible for managing and executing scenario-based evaluations in CARLA.

    This includes:
    - Initializing the CARLA environment and loading the correct map.
    - Loading and validating the autonomous agent.
    - Configuring the simulation environment and traffic manager.
    - Executing scenario simulations with watchdogs for timeout.
    - Handling errors and cleanup.
    """

    ego_vehicles = []

    # Tunable parameters
    client_timeout = cfg.CONFIG["simulation"]["client_timeout"]  # in seconds
    wait_for_world = cfg.CONFIG["simulation"]["wait_for_world"]  # in seconds
    frame_rate = cfg.CONFIG["simulation"]["frame_rate"]  # in Hz

    def __init__(self, scenario_definition, args):
        """
        Initialize the ScenarioEvaluator.

        :param scenario_definition: An instance of ScenarioDefinition containing map and route info.
        :param args: Parsed command-line arguments for simulation configuration.
        """
        self.scenario_definition = scenario_definition

        self.sensors = None
        self.sensor_icons = []

        # First of all, we need to create the client that will send the requests
        # to the simulator.
        if CarlaDataProvider.get_client() is None:
            tag = os.environ.get("tag", "default")
            initialize_carla(tag=tag)

        self.client = CarlaDataProvider.get_client()

        if cfg.CONFIG["simulation"]["autopilot"]:
            self.traffic_manager = self.client.get_trafficmanager(int(args.trafficManagerPort))

        # dist = pkg_resources.get_distribution("carla")
        # if dist.version != 'leaderboard':
        #     if LooseVersion(dist.version) < LooseVersion('0.9.10'):
        #         raise ImportError("CARLA version 0.9.10.1 or newer required. CARLA version found: {}".format(dist))

        # Load agent
        module_name = os.path.basename(args.agent).split('.')[0]
        if os.path.dirname(args.agent) not in sys.path:
            sys.path.insert(0, os.path.dirname(args.agent))
        self.module_agent = importlib.import_module(module_name)

        # Create the ScenarioManager
        self.manager = ScenarioManager(args.timeout, args.debug > 1)

        # Time control for summary purposes
        self._start_time = GameTime.get_time()
        self._end_time = None

        # Create the agent timer
        self._agent_watchdog = Watchdog(int(float(args.timeout)))
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """
        Signal handler to stop simulation cleanly on user interruption or watchdog timeout.

        :param signum: Signal number.
        :param frame: Current stack frame.
        """
        if self._agent_watchdog and not self._agent_watchdog.get_status():
            raise RuntimeError("Timeout: Agent took too long to setup")
        elif self.manager:
            self.manager.signal_handler(signum, frame)

    def __del__(self):
        """
        Cleanup and delete actors, ScenarioManager and CARLA world.
        """
        # self._cleanup()
        if hasattr(self, 'manager') and self.manager:
            del self.manager
        if hasattr(self, 'world') and self.world:
            del self.world

    def _cleanup(self):
        """
        Cleanup the CARLA world and destroy all actors. Also resets internal states and watchdogs.
        """
        if self.manager and self.manager.get_running_status() \
                and hasattr(self, 'world') and self.world:
            self.world.set_weather(CarlaDataProvider.find_weather_presets()[0][0])
            if cfg.CONFIG["simulation"]["autopilot"]:
                for v in CarlaDataProvider.get_world().get_actors().filter('vehicle.*'):
                    try:
                        v.set_autopilot(False, self.traffic_manager.get_port())
                    except Exception:
                        pass

                self.traffic_manager.set_hybrid_physics_mode(False)
                self.traffic_manager.set_synchronous_mode(False)

        if self.manager:
            self.manager.cleanup()

        GameTime.restart()
        DestroyActor = carla.command.DestroyActor
        batch = []

        for actor_id in CarlaDataProvider._carla_actor_pool.copy():
            actor = CarlaDataProvider._carla_actor_pool[actor_id]
            if "sensor." in actor.type_id:
                try:
                    actor.stop()
                    actor.listen(lambda _: None)  # detach callback
                except Exception:
                    pass
            if actor.is_alive:
                batch.append(DestroyActor(actor))

        if CarlaDataProvider._client:
            try:
                CarlaDataProvider._client.apply_batch_sync(batch)
            except RuntimeError as e:
                if "time-out" in str(e):
                    pass
                else:
                    raise e

        self.world.tick()

        CarlaDataProvider._actor_velocity_map.clear()
        CarlaDataProvider._actor_location_map.clear()
        CarlaDataProvider._actor_transform_map.clear()
        CarlaDataProvider._traffic_light_map.clear()
        CarlaDataProvider._ego_vehicle_route = None
        CarlaDataProvider._carla_actor_pool = dict()
        CarlaDataProvider._spawn_index = 0

        for i, _ in enumerate(self.ego_vehicles):
            if self.ego_vehicles[i]:
                self.ego_vehicles[i].destroy()
                self.ego_vehicles[i] = None
        self.ego_vehicles = []

        if self._agent_watchdog._timer:
            self._agent_watchdog.stop()

        if hasattr(self, 'agent_instance') and self.agent_instance:
            self.agent_instance.destroy()
            self.agent_instance = None

        gc.collect()

    def _prepare_ego_vehicles(self, ego_vehicles, wait_for_ego_vehicles=False):
        """
        Spawn ego vehicles or attach to existing ones if wait_for_ego_vehicles is True.

        :param ego_vehicles: List of ego vehicle blueprints and transforms.
        :param wait_for_ego_vehicles: Whether to wait for pre-spawned ego vehicles.
        """

        if not wait_for_ego_vehicles:
            for vehicle in ego_vehicles:
                self.ego_vehicles.append(
                    CarlaDataProvider.request_new_actor(
                        vehicle.model,
                        vehicle.transform, vehicle.rolename,
                        color=vehicle.color,
                        vehicle_category=vehicle.category,
                    )
                )

        else:
            ego_vehicle_missing = True
            while ego_vehicle_missing:
                self.ego_vehicles = []
                ego_vehicle_missing = False
                for ego_vehicle in ego_vehicles:
                    ego_vehicle_found = False
                    carla_vehicles = CarlaDataProvider.get_world().get_actors().filter('vehicle.*')
                    for carla_vehicle in carla_vehicles:
                        if carla_vehicle.attributes['role_name'] == ego_vehicle.rolename:
                            ego_vehicle_found = True
                            self.ego_vehicles.append(carla_vehicle)
                            break
                    if not ego_vehicle_found:
                        ego_vehicle_missing = True
                        break

            for i, _ in enumerate(self.ego_vehicles):
                self.ego_vehicles[i].set_transform(ego_vehicles[i].transform)

        # sync state
        CarlaDataProvider.get_world().tick()

    def _load_and_wait_for_world(self, args):
        """
        Load the required CARLA map and configure the simulation environment.

        :param args: Parsed command-line arguments with simulation settings.
        """

        self.world = self.client.load_world(self.scenario_definition.town)
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 1.0 / self.frame_rate
        settings.synchronous_mode = True
        self.world.apply_settings(settings)
        CarlaDataProvider.set_world(self.world)

        self.world.reset_all_traffic_lights()

        if cfg.CONFIG["simulation"]["autopilot"]:
            self.traffic_manager = self.client.get_trafficmanager(int(args.trafficManagerPort))
            self.traffic_manager.set_synchronous_mode(True)
            self.traffic_manager.set_hybrid_physics_mode(False)
            self.traffic_manager.set_random_device_seed(int(args.trafficManagerSeed))

        for _ in range(5):
            self.world.tick()

        if CarlaDataProvider.get_map().name.lower() != self.scenario_definition.town.lower():
            raise Exception("The CARLA server uses the wrong map!"
                            "This scenario requires to use map {}".format(self.scenario_definition.town))

    # @profile
    def _load_and_run_scenario(self, args, repetition_index, save_snapshot=False):
        """
        Load and run a scenario from start to finish, including agent setup and scenario execution.

        :param args: Simulation arguments including agent path and configs.
        :param repetition_index: Index to distinguish between repeated trials.
        :param save_snapshot: Whether to save scenario snapshot (optional).
        :raises: SimulationError, LoadingScenarioFailedError, AgentSetupFailedError
        """
        logger.info(
            f"\n\033[1m========= Preparing {self.scenario_definition.id_} (repetition {repetition_index}) =========")
        logger.info("> Setting up the agent\033[0m")

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

        # Set up the user's agent, and the timer to avoid freezing the simulation
        try:
            self._agent_watchdog.start()
            agent_class_name = getattr(self.module_agent, 'get_entry_point')()
            self.agent_instance = getattr(self.module_agent, agent_class_name)(args.agent_config,
                                                                               args.additional_config)
            self.sensors = self.agent_instance.sensors()
            track = self.agent_instance.track
            AgentWrapper.validate_sensor_configuration(self.sensors, track, args.track)
            self.sensor_icons = [sensors_to_icons[sensor['type']] for sensor in self.sensors]

            self._agent_watchdog.stop()

        except SensorConfigurationInvalid as e:
            logger.error(f"\n\033[91mThe sensor's configuration used is invalid: {e}")
            self._cleanup()
            raise e

        except Exception as e:
            # The agent setup has failed -> start the next route
            logger.error(f"\n\033[91mCould not set up the required agent: {e}")
            traceback.print_exc()
            self._cleanup()
            raise AgentSetupFailedError(f"\n\033[91mCould not set up the required agent: {e}")

        logger.info("\033[1m> Loading the world\033[0m")

        # Load the world and the scenario
        try:
            self._load_and_wait_for_world(args)

            scenario = RouteScenario(world=self.world, scenario_definition=self.scenario_definition,
                                     agent_instance=self.agent_instance,
                                     debug_mode=args.debug)

            # Load scenario and run it
            if args.record:
                self.client.start_recorder(
                    "{}/{}_rep{}.log".format(args.record, self.scenario_definition.id_, repetition_index))
            self.manager.load_scenario(scenario, self.agent_instance, repetition_index)
        except InvalidScenarioDefinitionError as e:
            self._cleanup()
            raise e
        except Exception as e:
            # The scenario is wrong -> set the ejecution to crashed and stop
            logger.error(f"\n\033[91mThe scenario could not be loaded: {e}")
            if cfg.CONFIG['debug']:
                traceback.print_exc()
            self._cleanup()
            raise LoadingScenarioFailedError(f"\n\033[91mThe scenario could not be loaded: {e}")

        logger.info("\033[1m> Running the scenario\033[0m")

        # Run the scenario
        try:
            self.manager.run_scenario()
        except AgentError as e:
            # The agent has failed -> stop the route
            logger.error(f"\n\033[91mStopping the route, the agent has crashed: {e}")
            if cfg.CONFIG["debug"]:
                traceback.print_exc()
            self._cleanup()
            raise e

        except Exception as e:
            logger.error(f"\n\033[91mError during the simulation: {e}")
            if cfg.CONFIG['debug']:
                traceback.print_exc()
            self._cleanup()
            raise SimulationError(f"\n\033[91mError during the simulation: {e}")

        # Stop the scenario
        try:
            logger.info("\033[1m> Stopping the route\033[0m")
            self.manager.stop_scenario()
            # self._register_statistics(config, args.checkpoint, entry_status, crash_message)

            if args.record:
                self.client.stop_recorder()

            self._cleanup()

        except Exception as e:
            logger.error(f"\n\033[91mFailed to stop the scenario: {e}")
            traceback.print_exc()
            raise StoppingScenarioFailedError(f"\n\033[91mFailed to stop the scenario: {e}")
        finally:
            if args.record:
                self.client.stop_recorder()

    def run(self, args):
        """
        Run all scenarios defined by arguments.

        :param args: Command-line arguments specifying settings.
        """

        for i in range(args.repetitions):
            # run
            self._load_and_run_scenario(args, repetition_index=i,
                                        save_snapshot=cfg.CONFIG["simulation"]["save_snapshot"])


def main():
    description = "CARLA AD Leaderboard Evaluation: evaluate your Agent in CARLA scenarios\n"

    # general parameters
    parser = argparse.ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--port', default='2000', help='TCP port to listen to (default: 2000)')
    parser.add_argument('--trafficManagerPort', default='2500',
                        help='Port to use for the TrafficManager (default: 8000)')
    parser.add_argument('--trafficManagerSeed', default='1',
                        help='Seed used by the TrafficManager (default: 0)')
    parser.add_argument('--carlaProviderSeed', default='2000',
                        help='Seed used by the CarlaProvider (default: 2000)')
    parser.add_argument('--debug', type=int, help='Run with debug output', default=0)
    parser.add_argument('--record', type=str, default='',
                        help='Use CARLA recording feature to create a recording of the scenario')
    parser.add_argument('--timeout', default="600.0",
                        help='Set the CARLA client timeout value in seconds')

    # simulation setup
    parser.add_argument('--routes',
                        help='Name of the route to be executed. Point to the route_xml_file to be executed.',
                        default='leaderboard/data/training_routes/routes_town05_long.xml')
    parser.add_argument('--scenarios',
                        help='Name of the scenario annotation file to be mixed with the route.',
                        default='leaderboard/data/scenarios/town05_all_scenarios.json')
    parser.add_argument('--repetitions',
                        type=int,
                        default=1,
                        help='Number of repetitions per route.')

    # agent-related options
    parser.add_argument("-a", "--agent", type=str, help="Path to Agent's py file to evaluate",
                        default="leaderboard/team_code/interfuser_agent.py")
    parser.add_argument("--agent-config", type=str, help="Path to Agent's configuration file",
                        default="leaderboard/team_code/interfuser_config.py")

    parser.add_argument("--track", type=str, default='SENSORS', help="Participation track: SENSORS, MAP")
    parser.add_argument('--resume', type=bool, default=False, help='Resume execution from last checkpoint?')
    parser.add_argument("--checkpoint", type=str,
                        default='results/sample_result.json',
                        help="Path to checkpoint used for saving statistics and resuming")

    arguments = parser.parse_args()

    scenario_definition = ScenarioDefinition.generate_random()

    try:
        leaderboard_evaluator = ScenarioEvaluator(scenario_definition, arguments)
        leaderboard_evaluator.run(arguments)

    except Exception as e:
        traceback.print_exc()
    finally:
        del leaderboard_evaluator


if __name__ == '__main__':
    main()
