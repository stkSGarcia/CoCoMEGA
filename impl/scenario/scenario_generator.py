import json
import os
from docker.errors import NotFound
import docker
import carla
from carla.libcarla import Vector3D
from impl.scenario.scenario import Scenario

BRIDGE_CONTAINER = 'carla_cyber_0.9.14'
BRIDGE_HOME = '/home/hossein/carla_apollo_bridge'


class ScenarioRunner:
    def __init__(self, scenario):
        self.scenario = scenario
        self.carla_client = carla.Client('localhost', 2000)
        self.world = self.carla_client.get_world()

    def run_scenario(self):
        """
        simulate the scenario in an update loop:
        - apply controls to the actors (speed, direction, acceleration, state change)
        - apply freeze time logic
        - retrieve, process and save measures
        """
        self.setup_carla()
        file_name = self.generate_object_json()
        self.spawn_objects(file_name)
        self.activate_apollo_modules()
        while True:
            # TODO keep track of time, apply controls to Actors after freeze time, and measure throttle, break, steering angle of ego_vehicle
            self.world.tick()

    def setup_carla(self):
        """
        Setup Carla world configurations
        """
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        self.carla_client.reload_world(False)

    def generate_object_json(self):
        """
        Generates a json file containing type, location and rotation of all objects to spawn.
        """
        file_name = 'objects2.json'
        save_path = os.path.join(BRIDGE_HOME, 'src', 'carla_spawn_objects', 'config', f'{file_name}.json')
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
        docker_client = docker.from_env()
        try:
            bridge = docker_client.containers.get(BRIDGE_CONTAINER)
            response = bridge.exec_run(
                f'bash -c "source /apollo/cyber/setup.bash && python /apollo/cyber/carla_bridge/carla_spawn_objects/new_carla_spawn_objects.py {object_file_name}"',
                detach=False,
                environment=[
                    'CARLA_PYTHON_ROOT="/apollo/cyber/carla_bridge/carla_python"',
                    'PYTHONPATH="$PYTHONPATH:/apollo/py_proto:/apollo/cyber:/apollo/cyber/carla_bridge/carla_python/carla"'
                ],
            )
            output = response.output.decode("utf-8")
            print(output)
        except NotFound:
            raise Exception(f'Error: Container "{BRIDGE_CONTAINER}" is not running.')

    def activate_apollo_modules(self):
        """
        activate necessary modules for apollo using CyberRT API
        """
        pass

    def apply_control(self, actor, direction, speed):
        """
        applies a specific control to an actor
        """
        pass


if __name__ == '__main__':
    v = Vector3D(0, 0, 0)
    scenario = Scenario(v)
    sr = ScenarioRunner(scenario)
    sr.run_scenario()
