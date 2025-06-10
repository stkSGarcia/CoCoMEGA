from typing import Any

from impl.core.mr.base_mr import AbstractPerturbation, AbstractPerturbationFactory, Perturbations, AbstractRelation
from impl.core.scenario.base_scenario import AbstractScenarioDefinition


class Perturbation(AbstractPerturbation):
    def dist(self, other: 'AbstractPerturbation', **kwargs) -> float:
        # TODO Implement distance function
        pass

    def mate(self, other: 'AbstractPerturbation', cxpb, **kwargs) -> None:
        # TODO Implement crossover operator
        pass

    def mutate(self, mutpb, *args, **kwargs) -> None:
        # TODO Implement mutation operator
        pass

    def __eq__(self, other: Any) -> bool:
        # TODO Implement equality criteria
        pass

    def apply_perturbation(self, scenario: AbstractScenarioDefinition):
        # TODO Implement perturbation apply function
        pass


class PerturbationFactory(AbstractPerturbationFactory):
    def spawn(self) -> AbstractPerturbation:
        # TODO Implement spawn function to randomly generate a perturbation
        pass

    def has(self, perturbations: Perturbations):
        # TODO Implement has function
        pass


class Relation(AbstractRelation):
    def is_violated(self, original_output: Any, perturbed_output: Any, **kwargs) -> (bool, float):
        # TODO Implement is_violated function
        pass
