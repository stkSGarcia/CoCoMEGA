import random
from typing import List


class ScenarioDefinition:
    def __init__(self, vector):
        """Constructor.

        @param vector:
        1- Pedestrian parameters:
        Location_x
        Location_y
        Rotation_yaw
        Rotation_pitch
        Speed_x
        Speed_y
        child_or_adult
        freeze_time

        2- Car parameters:
        Location_x,
        Location_y,
        Rotation_yaw,
        Rotation_pitch,
        Speed_x,
        Speed_y,
        type (bike, Regular car, Truck),
        freeze_time,
        acceleration_x,
        acceleration_y,

        3- Sign parameters:
        Location_x,
        Location_y,
        Rotation_yaw,
        Rotation_pitch,

        4- Weather
        5- Darkness (Numerical or Categorical)
        """
        self.vector = vector
        self.ego_vehicle = self._get_ego_vehicle()
        self.other_vehicles = self._get_other_vehicles()
        self.walkers = self._get_walkers()
        self.traffic_signs = self._get_traffic_signs()

    def encode(self):
        pass

    def decode(self):
        pass

    def update_pedestrian(self, loc_x, loc_y, yaw, pitch, speed_x, speed_y, typ, freeze_time):
        self._update_vector(range(0, 8), loc_x, loc_y, yaw, pitch, speed_x, speed_y, typ, freeze_time)

    def update_vehicle(self, loc_x, loc_y, yaw, pitch, speed_x, speed_y, typ, freeze_time):
        self._update_vector(range(8, 18), loc_x, loc_y, yaw, pitch, speed_x, speed_y, typ, freeze_time)

    def update_object(self, loc_x, loc_y, yaw, pitch):
        self._update_vector(range(18, 22), loc_x, loc_y, yaw, pitch)

    def update_weather(self, value):
        self.vector[22] = value

    def update_darkness(self, value):
        self.vector[23] = value

    def _update_vector(self, index_range, *args):
        for index, candidate in zip(index_range, args):
            if candidate is not None:
                self.vector[index] = candidate

    def _get_ego_vehicle(self):
        return {
            "spawn_point": {
                "x": 200.0,
                "y": -2.0,
                "z": 0.5,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0
            }
        }

    def get_trajectory(self):
        return [
            {
                'x': -188.04,
                'y': 111.89,
                'z': 0.0,
                'yaw': -90,
            },
            {
                'x': -208.309,
                'y': 87.82,
                'z': 0.0,
                'yaw': 180,
            },
            {
                'x': -241.01,
                'y': 87.77,
                'z': 0.0,
                'yaw': 180,
            },

        ]

    def _get_other_vehicles(self):
        return [
            {
                "model": "vehicle.tesla.model3",
                "id": "vehicle_nearby",
                "speed": 2,
                "color": "green",
                "spawn_point": {
                    "x": 250.0,
                    "y": -2.0,
                    "z": 0.5,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 0.0
                }
            },
            {
                "model": "vehicle.tesla.model3",
                "id": "vehicle_nearby2",
                "speed": 2,
                "color": "green",
                "spawn_point": {
                    "x": 270.0,
                    "y": -2.0,
                    "z": 0.5,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 0.0
                }
            },
            {
                "model": "vehicle.tesla.model3",
                "id": "vehicle_nearby3",
                "speed": 2,
                "color": "(20, 240, 20)",
                "spawn_point": {
                    "x": -188.0,
                    "y": 125.0,
                    "z": 0.0,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": -90
                }
            },
            {
                "model": "vehicle.tesla.model3",
                "id": "vehicle_nearby3",
                "speed": 20,
                "autopilot": True,
                "color": "(20, 20, 240)",
                "spawn_point": {
                    "x": -188.0,
                    "y": 100.0,
                    "z": 0.0,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": -90
                }
            },
        ]

    def _get_walkers(self):
        return [
            {
                "model": "walker.pedestrian.0001",  # Adult
                "id": "pedestrian1",
                "speed": 2,
                "spawn_point": {
                    "x": -184.0,
                    "y": 100.0,
                    "z": 0.0,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 90.0
                }
            },
            {
                "model": "walker.pedestrian.0011",  # Child
                "id": "pedestrian2",
                "speed": 2,
                "spawn_point": {
                    "x": -184.0,
                    "y": 95.0,
                    "z": 0.0,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 90.0,
                }
            },
        ]

    def get_other_actors(self):
        return self._get_other_vehicles() + self._get_walkers()

    def get_trigger_position(self):
        return {
            'x': -188.04,
            'y': 111.89,
            'z': 0.0,
            'yaw': -90,
        }

    def _get_traffic_signs(self):
        return {}

    @staticmethod
    def generate_random_scenario(boundary: List[List]):
        vector = []
        for lower, upper in boundary:
            if isinstance(lower, float):
                vector.append(random.uniform(lower, upper))
            elif isinstance(lower, int):
                vector.append(random.randint(lower, upper))
            else:
                raise ValueError(f"Invalid boundary type: {type(lower)}.")
        return ScenarioDefinition(vector)
