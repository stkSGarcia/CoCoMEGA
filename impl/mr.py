import random
from abc import abstractmethod, ABC
from collections import defaultdict
from typing import List, Callable

from impl.scenario import Scenario, Actor


class Perturbation(ABC):
    def __init__(self, uid):
        self.uid = uid

    @abstractmethod
    def perturb(self, scenario: Scenario):
        raise NotImplementedError


class ActorPerturbation(Perturbation):
    def __init__(self, uid, blueprint, x, y, z, pitch, yaw, roll, velocity):
        super().__init__(uid)
        self.blueprint = blueprint
        self.x = x
        self.y = y
        self.z = z
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll
        self.velocity = velocity

    def perturb(self, scenario: Scenario):
        if self.uid not in scenario.actors:
            scenario.actors[self.uid] = Actor(self.uid, self.blueprint, self.x, self.y, self.z, self.pitch, self.yaw,
                                              self.roll, self.velocity)
        else:
            scenario.actors[self.uid].update(blueprint=self.blueprint, x=self.x, y=self.y, z=self.z, pitch=self.pitch,
                                             yaw=self.yaw, roll=self.roll, velocity=self.velocity)


class EnvPerturbation(Perturbation):
    def __init__(self, uid, value):
        super().__init__(uid)
        self.value = value

    def perturb(self, scenario: Scenario):
        setattr(scenario, self.uid, self.value)


class PerturbationFactory(ABC):
    @abstractmethod
    def spawn(self) -> Perturbation:
        raise NotImplementedError


class ActorPerturbationFactory(PerturbationFactory):
    def __init__(self, blueprint, x_range, y_range, z_range, pitch_range, yaw_range, roll_range, velocity_range=None):
        self.blueprint = blueprint
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.pitch_range = pitch_range
        self.yaw_range = yaw_range
        self.roll_range = roll_range
        self.velocity_range = velocity_range
        self.last_uid = 0

    def spawn(self) -> ActorPerturbation:
        uid = f"{self.last_uid}{self.blueprint}"
        self.last_uid += 1
        return ActorPerturbation(
            uid,
            self.blueprint,
            random.uniform(self.x_range[0], self.x_range[1]),
            random.uniform(self.y_range[0], self.y_range[1]),
            random.uniform(self.z_range[0], self.z_range[1]),
            random.uniform(self.pitch_range[0], self.pitch_range[1]),
            random.uniform(self.yaw_range[0], self.yaw_range[1]),
            random.uniform(self.roll_range[0], self.roll_range[1]),
            random.uniform(self.velocity_range[0], self.velocity_range[1]) if self.velocity_range is not None else None,
        )


class EnvPerturbationFactory(PerturbationFactory):
    def __init__(self, field, dtype, lower_bound=None, upper_bound=None):
        self.field = field
        self.dtype = dtype
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def spawn(self) -> EnvPerturbation:
        if self.dtype == float:
            return EnvPerturbation(self.field, random.uniform(self.lower_bound, self.upper_bound))
        elif self.dtype == int:
            return EnvPerturbation(self.field, random.randint(self.lower_bound, self.upper_bound))
        elif self.dtype == bool:
            return EnvPerturbation(self.field, bool(random.getrandbits(1)))
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}.")


class Relation(ABC):
    def __init__(self, field):
        self.field = field

    @abstractmethod
    def to_objective(self, source_result) -> Callable:
        raise NotImplementedError


class Invariance(Relation):
    def __init__(self, field, threshold=0.01):
        super().__init__(field)
        self.threshold = threshold

    def to_objective(self, source_result) -> (Callable, float):
        def fitness(result):
            diff = abs(result - source_result)
            return diff > self.threshold, diff

        return fitness, 1.0


class Decreasing(Relation):
    def __init__(self, field, threshold=0.1):
        super().__init__(field)
        self.threshold = threshold

    def to_objective(self, source_result) -> (Callable, float):
        def fitness(result):
            diff = source_result - result
            return diff < source_result * self.threshold, diff

        return fitness, -1.0


class Increasing(Relation):
    def __init__(self, field, threshold=0.1):
        super().__init__(field)
        self.threshold = threshold

    def to_objective(self, source_result) -> (Callable, float):
        def fitness(result):
            diff = result - source_result
            return diff < source_result * self.threshold, diff

        return fitness, -1.0


class MR:
    def __init__(self, perturbation_factories: List[PerturbationFactory], relations: List[Relation]):
        self.perturbation_factories = perturbation_factories
        self.relations = relations

    def generate_perturbation(self) -> Perturbation:
        return random.choice(self.perturbation_factories).spawn()


class MRSet:
    def __init__(self, mrs: List[MR]):
        self.mrs = mrs

        # merge relations
        self.relations = []
        relations = [relation for mr in self.mrs for relation in mr.relations]
        relation_dict = defaultdict(list)
        for relation in relations:
            relation_dict[relation.field].append(relation)
        for relation_list in relation_dict.values():
            pass  # TODO

    def to_objectives(self, source_results: List) -> (List, List):
        """Transform the set of metamorphic relations into fitness functions.

        @param source_results: simulation results of the source scenarios
        @return: (fitness functions, weights)
        """
        assert len(source_results) == len(self.relations)
        fitness_functions = []
        weights = []
        for relation, source_result in zip(self.relations, source_results):
            fitness, weight = relation.to_objectives(source_result)
            fitness_functions.append(fitness)
            weights.append(weight)
        return fitness_functions, weights

    def mutate(self, mutpb, *individuals):
        """Mutates individuals according to the probability `mutpb`.

        @param mutpb: mutation probability
        """
        for individual in individuals:
            # add perturbations
            times = 1
            while random.uniform(0, 1) < mutpb ** times:
                perturbation = random.choice(self.mrs).generate_perturbation()
                individual.append(perturbation)
                times += 1

            # TODO: Squash perturbations
