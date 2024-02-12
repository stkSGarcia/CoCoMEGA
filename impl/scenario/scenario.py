import random


class Scenario:
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

    def _get_other_vehicles(self):
        return [
            {
                "type": "vehicle.tesla.model3",
                "id": "vehicle_nearby",
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
                "type": "vehicle.tesla.model3",
                "id": "vehicle_nearby2",
                "spawn_point": {
                    "x": 270.0,
                    "y": -2.0,
                    "z": 0.5,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 0.0
                }
            },
        ]

    def _get_walkers(self):
        return [
            {
                "type": "walker.pedestrian.0001",  # Adult
                "id": "pedestrian1",
                "spawn_point": {
                    "x": 215.0,
                    "y": -5.0,
                    "z": 1.0,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 90.0
                }
            },
            {
                "type": "walker.pedestrian.0011",  # Child
                "id": "pedestrian2",
                "spawn_point": {
                    "x": 220.0,
                    "y": -5.0,
                    "z": 1.0,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 90.0,
                }
            },
        ]

    def _get_traffic_signs(self):
        return {}

    @staticmethod
    def generate_random_scenario():
        return Scenario([
            random.uniform(0.0, 1000.0),
            random.uniform(0.0, 1000.0),
            random.uniform(0.0, 360.0),
            random.uniform(0.0, 360.0),
            random.uniform(0.0, 100.0),
            random.uniform(0.0, 100.0),
            random.randint(0, 1),
            random.uniform(0.0, 100.0),

            random.uniform(0.0, 1000.0),
            random.uniform(0.0, 1000.0),
            random.uniform(0.0, 360.0),
            random.uniform(0.0, 360.0),
            random.uniform(0.0, 100.0),
            random.uniform(0.0, 100.0),
            random.randint(0, 2),
            random.uniform(0.0, 100.0),
            random.uniform(0.0, 100.0),
            random.uniform(0.0, 100.0),

            random.uniform(0.0, 1000.0),
            random.uniform(0.0, 1000.0),
            random.uniform(0.0, 360.0),
            random.uniform(0.0, 360.0),

            random.randint(0, 10),
            random.randint(0, 5),
        ])
