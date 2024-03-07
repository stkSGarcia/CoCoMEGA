import hashlib
import math
import random


class ScenarioDefinition:
    BOUNDARY = [
        (0.0, 1000.0),
        (0.0, 1000.0),
        (0.0, 360.0),
        (0.0, 360.0),
        (0.0, 100.0),
        (0.0, 100.0),
        (0, 1),
        (0.0, 100.0),

        (0.0, 1000.0),
        (0.0, 1000.0),
        (0.0, 360.0),
        (0.0, 360.0),
        (0.0, 100.0),
        (0.0, 100.0),
        (0, 2),
        (0.0, 100.0),
        (0.0, 100.0),
        (0.0, 100.0),

        (0.0, 1000.0),
        (0.0, 1000.0),
        (0.0, 360.0),
        (0.0, 360.0),

        (0, 10),
        (0, 5),
    ]
    RANGES = [upper - lower if isinstance(lower, float) else None for lower, upper in BOUNDARY]

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
        self.definition_id = self._generate_def_id()
        self.ego_vehicle = self._get_ego_vehicle()
        self.other_vehicles = self._get_other_vehicles()
        self.walkers = self._get_walkers()
        self.traffic_signs = self._get_traffic_signs()

    def encode(self):
        pass

    def decode(self):
        pass

    def update(self, values):
        for i, value in enumerate(values):
            if value is not None:
                self.vector[i] += value
                self.vector[i] = max(ScenarioDefinition.BOUNDARY[i][0], self.vector[i])
                self.vector[i] = min(ScenarioDefinition.BOUNDARY[i][1], self.vector[i])

    def heterogeneous_distance(self, other):
        assert len(self.vector) == len(other.vector)
        dist = 0
        for this_attr, other_attr, attr_range in zip(self.vector, other.vector, ScenarioDefinition.RANGES):
            if attr_range is not None:
                if this_attr is None: this_attr = 0
                if other_attr is None: other_attr = 0
                dist += pow(abs(this_attr - other_attr) / attr_range, 2)
            else:
                dist += 0 if this_attr == other_attr else 1
        return math.sqrt(dist)

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
                "id": "vehicle_nearby1",
                "speed": 2,
                "color": "(20, 240, 20)",
                "spawn_point": {
                    "x": -195.0,
                    "y": 120.0,
                    "z": 0.0,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": -90
                }
            },
            {
                "model": "vehicle.tesla.model3",
                "id": "vehicle_nearby2",
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
                "speed": 1,
                "spawn_point": {
                    "x": -182.0,
                    "y": 100.0,
                    "z": 0.0,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 180.0
                }
            },
            {
                "model": "walker.pedestrian.0011",  # Child
                "id": "pedestrian2",
                "speed": 1.5,
                "spawn_point": {
                    "x": -182.0,
                    "y": 102.0,
                    "z": 0.0,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 180.0,
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
    def generate_random_scenario():
        vector = []
        for lower, upper in ScenarioDefinition.BOUNDARY:
            if isinstance(lower, float):
                vector.append(random.uniform(lower, upper))
            elif isinstance(lower, int):
                vector.append(random.randint(lower, upper))
            else:
                raise ValueError(f"Invalid boundary type: {type(lower)}.")
        return ScenarioDefinition(vector)

    def _generate_def_id(self):
        return hashlib.sha256(str(self.vector).encode()).hexdigest()[:32]
