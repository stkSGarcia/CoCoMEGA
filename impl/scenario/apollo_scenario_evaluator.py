import json
import os
import time

from docker.errors import NotFound
import docker
import carla
from carla.libcarla import Vector3D
from impl.scenario.scenario import ScenarioDefinition
import subprocess

USER = os.environ["USER"]
BRIDGE_CONTAINER = 'carla_cyber_0.9.14'
APOLLO_CONTAINER = f'apollo_dev_{USER}'
BRIDGE_HOME = f'/home/{USER}/carla_apollo_bridge'
APOLLO_HOME = f'/home/{USER}/apollo'
CARLA_HOME = f'/home/{USER}/carla0.9.15/CARLA_0.9.15'


class ScenarioRunner:
    def __init__(self, scenario):
        self.scenario = scenario
        # self.carla_client = carla.Client('localhost', 2000)
        # self.world = self.carla_client.get_world()
        self.docker_client = docker.from_env()
        self.apollo_container = self._get_container(APOLLO_CONTAINER)
        self.bridge_container = self._get_container(BRIDGE_CONTAINER)

    def run_scenario(self):
        """
        simulate the scenario in an update loop:
        - apply controls to the actors (speed, direction, acceleration, state change)
        - apply freeze time logic
        - retrieve, process and save measures
        """
        apollo_modules = ['routing', 'control', 'localization', 'perception', 'prediction', 'planning']
        self.setup_env()
        file_name = self.generate_object_json()
        self.spawn_objects(file_name)
        self.activate_apollo_modules(apollo_modules)
        # self.send_routing_request()
        # while True:
        #     # TODO keep track of time, apply controls to Actors after freeze time, and measure throttle, break, steering angle of ego_vehicle
        #     self.world.tick()

    def setup_env(self):
        """
        Setup the evnironment for running scenarios. This includes running the apollo, Carla and the bridge
        """
        if self.apollo_container is None or self.apollo_container.status == 'exited':
            self._build_apollo()

        if self.bridge_container is None:
            self._build_bridge()

        self._run_apollo()
        self._run_carla()  # self._run_carla_docker()
        time.sleep(20)
        self._run_bridge()
        time.sleep(10)

    def setup_carla(self):
        """
        Setup Carla world configurations
        """
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        # self.carla_client.reload_world(False)

    def generate_object_json(self):
        """
        Generates a json file containing type, location and rotation of all objects to spawn.
        """
        file_name = 'objects.json'
        save_path = os.path.join(BRIDGE_HOME, 'src', 'carla_spawn_objects', 'config', f'{file_name}')
        with open('apollo_modules.json') as handle:
            modules = json.loads(handle.read())
        objects = modules['objects']
        ego_vehicle_index, ego_vehicle_object = [(i, v) for i, v in enumerate(objects) if v['id'] == 'ego_vehicle'][0]
        objects[ego_vehicle_index]['spawn_point'] = self.scenario.ego_vehicle['spawn_point']
        for vehicle in self.scenario.other_vehicles:
            objects.append(vehicle)
        for walker in self.scenario.walkers:
            objects.append(walker)
        for traffic_sign in self.scenario.traffic_signs:
            objects.append(traffic_sign)

        with open(save_path, 'w') as handle:
            json.dump(modules, handle, indent=4)

        return file_name

    def spawn_objects(self, object_file_name):
        """
        Spawns objects including the ego vehicle via carla-apollo bridge
        """
        print("spawning objects...")
        try:
            bridge = self.docker_client.containers.get(BRIDGE_CONTAINER)
            response = bridge.exec_run(
                f'bash -c "source /apollo/cyber/setup.bash && cd /apollo/cyber/carla_bridge && python carla_spawn_objects/carla_spawn_objects.py"',
                detach=False,
                environment=[
                    'CARLA_PYTHON_ROOT="/apollo/cyber/carla_bridge/carla_python"',
                    'PYTHONPATH="$PYTHONPATH:/apollo/py_proto:/apollo/cyber:/apollo/cyber/carla_bridge/carla_python/carla:/apollo/cyber/carla_bridge:/apollo/cyber/carla_bridge/cyber_compatibility"'
                ],
            )
            # output = response.output.decode("utf-8")
            print(response)
        except NotFound:
            raise Exception(f'Error: Container "{BRIDGE_CONTAINER}" is not running.')

    def activate_apollo_modules(self, modules):
        """
        activate necessary modules for apollo using CyberRT API
        """
        apollo = self.docker_client.containers.get(APOLLO_CONTAINER)
        for module in modules:
            response = apollo.exec_run(
                f'bash -c "source /apollo/scripts/{module}.sh"'
            )
            output = response.output.decode("utf-8")
            print(output)

    def send_routing_request(self):
        """
        Sends a routing request that contains waypoints to apollo.
        """

    def apply_control(self, actor, direction, speed):
        """
        applies a specific control to an actor
        """
        pass

    def _build_apollo(self):
        """
        Build the Apollo docker container
        """
        process = subprocess.run([f"{APOLLO_HOME}/docker/scripts/dev_start.sh"],
                                 shell=True, stdout=subprocess.PIPE)
        if process.returncode != 0:
            raise Exception(f"There was an error building apollo: {process.stdout}")

        apollo = self.docker_client.containers.get(APOLLO_CONTAINER)
        response = apollo.exec_run(
            f'bash -c "source /apollo/apollo.sh build_gpu"',
            user=USER,
            detach=False,
        )
        if response.exit_code != 0:
            raise Exception(f"There was an error while building apollo: {response.output}")
        print(response)

    def _build_bridge(self):
        """
        Build the bridge docker container
        """
        try:
            bridge = self.docker_client.containers.get(BRIDGE_CONTAINER)
        except NotFound:
            process = subprocess.run([f"cd {BRIDGE_HOME}/docker && ./build_docker.sh && ./run_docker.sh"],
                                     shell=True, stdout=subprocess.PIPE)

            bridge = self.docker_client.containers.get(BRIDGE_CONTAINER)

            response = bridge.exec_run(
                f'bash -c "source /apollo/cyber/setup.bash && source /apollo/apollo.sh build_cyber opt"',
                detach=False,
                environment=[
                    'CARLA_PYTHON_ROOT="/apollo/cyber/carla_bridge/carla_python"',
                    'PYTHONPATH="$PYTHONPATH:/apollo/py_proto:/apollo/cyber:/apollo/cyber/carla_bridge/carla_python/carla:/apollo/cyber/carla_bridge:/apollo/cyber/carla_bridge/cyber_compatibility"'
                ],
            )
            if response.exit_code != 0:
                raise Exception(f"There was an error while building bridge: {response.output}")

    def _run_carla(self):
        """
        Run Carla without docker
        """
        process = subprocess.Popen(
            [f"{CARLA_HOME}/CarlaUE4.sh", "-quality-level=Low", "-resX=300", "-resY=200", '-RenderOffScreen'],
            shell=True, stdout=subprocess.PIPE)
        return process

    def _run_carla_docker(self):
        """
        Run Carla docker container
        """
        process = subprocess.run([f"{BRIDGE_HOME}/docker/scripts/docker_run_carla.sh"],
                                 shell=True, stdout=subprocess.PIPE)
        return process

    def _run_apollo(self):
        apollo = self.docker_client.containers.get(APOLLO_CONTAINER)
        response = apollo.exec_run(
            f'bash -c "source /apollo/scripts/bootstrap.sh"',
            user=USER,
            detach=False,
        )
        print(response)

    def _run_bridge(self):
        bridge = self.docker_client.containers.get(BRIDGE_CONTAINER)

        response = bridge.exec_run(
            f'bash -c "source /apollo/cyber/setup.bash && cd /apollo/cyber/carla_bridge && python -u carla_cyber_bridge/bridge.py"',
            detach=True,
            environment=[
                'CARLA_PYTHON_ROOT="/apollo/cyber/carla_bridge/carla_python"',
                'PYTHONPATH="$PYTHONPATH:/apollo/py_proto:/apollo/cyber:/apollo/cyber/carla_bridge/carla_python/carla:/apollo/cyber/carla_bridge:/apollo/cyber/carla_bridge/cyber_compatibility"'
            ],
        )
        print(response)

    def _get_container(self, container_name):
        try:
            container = self.docker_client.containers.get(container_name)
            if container.status != 'running':
                return None
            return container
        except NotFound:
            return None


if __name__ == '__main__':
    v = Vector3D(0, 0, 0)
    scenario = ScenarioDefinition(v)
    sr = ScenarioRunner(scenario)
    sr.run_scenario()
