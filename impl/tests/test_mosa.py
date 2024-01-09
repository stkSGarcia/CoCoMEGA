import random

from impl.algorithms.mosa import MOSA


def evaluator(individual):
    return [
        random.uniform(1, 10.01),
        random.uniform(10, 20.01),
        random.uniform(0, 5.01),
    ]


objectives = [10, 20, 5]
bounds = [
    [1, 6],
    [2.3, 8.5],
    "bool",
    [0.1, 1.1],
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
