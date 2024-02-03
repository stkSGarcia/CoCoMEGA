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
