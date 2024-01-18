import random

from impl.algorithms.mosa import MOSA


def evaluator(individual):
    return [
        random.uniform(0.0, 150.0),  # velocity
        random.uniform(0.0, 360.0),  # steering angle
    ]


objectives = [10, 20, 5]
bounds = [
    [1, 6],  # weather
    "bool",
    [143.3, 198.4],  # location x
    [12.3, 18.3],  # location y
    [0.0, 360.0],  # rotation yaw
]
mosa = MOSA(10,
            evaluator,
            objectives,
            bounds,
            0.9,
            0.3,
            3600,
            250,
            435802)

mosa.solve()
