from deap import creator

import impl.config as cfg
from impl.core.mr.base_mr import Perturbations
from impl.domain_template.mr.domain_predefined import mr_set
from impl.domain_template.scenario.domain_scenario import ScenarioDefinition
from impl.sorting.evaluation.sorting_evaluation import SortEvaluator

#: Define :const:`mr_set`.
mr_set = mr_set


def _pop_scenario():
    """Initialize the scenario population"""
    pop_scenario = [creator.Scenario(instance=mr_set.generate_scenario()) for _ in
                    range(cfg.CONFIG["scenario"]["pop_size"])]
    return pop_scenario


#: Define :const:`DOMAIN_REGISTRY`.
DOMAIN_REGISTRY = {
    "Scenario": ScenarioDefinition,
    "ScenarioInit": _pop_scenario,
    "Perturbation": Perturbations,
    "Evaluation": SortEvaluator,
    "MRSet": mr_set,
    "Config": "config.yaml",
}
