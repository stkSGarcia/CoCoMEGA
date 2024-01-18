import random
from typing import Dict, List


class Actor:
    def __init__(self, uid, blueprint, x, y, z, pitch, yaw, roll, velocity):
        self.uid = uid
        self.blueprint = blueprint
        self.x = x
        self.y = y
        self.z = z
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll
        self.velocity = velocity

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if value is not None:
                setattr(self, key, value)


class Scenario:
    def __init__(self, time, weather, road_type, road_length, actors: Dict[str, Actor]):
        self.time = time
        self.weather = weather
        self.road_type = road_type
        self.road_length = road_length
        self.actors = actors

    @staticmethod
    def generate_random_source_scenarios(size: int) -> List["Scenario"]:
        scenarios = []
        for _ in range(size):
            scenarios.append(Scenario(
                random.randint(0, 3),
                random.randint(0, 6),
                random.randint(0, 3),
                random.randint(0, 3),
                {"ego": Actor(
                    "ego",
                    None,
                    random.uniform(0.0, 1000.0),
                    random.uniform(0.0, 1000.0),
                    random.uniform(0.0, 1.0),
                    random.uniform(0.0, 1.0),
                    random.uniform(0.0, 360.0),
                    random.uniform(0.0, 1.0),
                    random.uniform(0.0, 100.0),
                )}
            ))
        return scenarios
