from deap import creator

import impl.config as cfg
from impl.core.mr.base_mr import Perturbations
from impl.sorting.evaluation.sorting_evaluation import SortEvaluator
from impl.sorting.mr.sorting_predefined import mr_set
from impl.sorting.scenario.sorting_scenario import ScenarioDefinition

#: Defined :const:`mr_set`.
mr_set = mr_set


def _pop_scenario():
    """Initialize the scenario population"""
    pop_scenario = [creator.Scenario(instance=mr_set.generate_scenario()) for _ in
                    range(cfg.CONFIG["scenario"]["pop_size"])]
    return pop_scenario


def _pop_perturbation():
    """Initialize the perturbation population."""
    pop_size = cfg.CONFIG["perturbation"]["pop_size"]
    pop_perturbation = [creator.Perturbation(mr_set.generate_perturbation()) for _ in range(pop_size)]

    return pop_perturbation


#: Defined :const:`DOMAIN_REGISTRY`.
DOMAIN_REGISTRY = {
    "Scenario": ScenarioDefinition,
    "ScenarioInit": _pop_scenario,
    "Perturbation": Perturbations,
    "PerturbationInit": _pop_perturbation,
    "Evaluation": SortEvaluator,
    "MRSet": mr_set,
}
