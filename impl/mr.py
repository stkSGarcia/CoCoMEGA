from enum import Enum, auto
from typing import List


class Perturbation:
    def __init__(self, index, upper_bound, lower_bound):
        self.index = index
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound


class RelationType(Enum):
    Invariance = auto()
    Increasing = auto()
    Decreasing = auto()


class Relation:
    def __init__(self, index, type_: RelationType, threshold = 0.2):
        self.index = index
        self.type_ = type_
        self.threshold = threshold

    def to_fitness(self):
        pass


class MR:
    def __init__(self, perturbations: List[Perturbation], relations: List[Relation]):
        self.perturbations = perturbations
        self.relations = relations

    def to_operator(self):
        pass

    def to_fitness(self):
        pass
