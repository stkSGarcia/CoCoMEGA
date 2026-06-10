from deap import creator

import impl.config as cfg
from impl.core.mr.base_mr import Perturbations
from impl.domain_template.evaluation.domain_evaluation import DomainEvaluator
from impl.domain_template.mr.domain_predefined import mr_set
from impl.domain_template.scenario.domain_scenario import ScenarioDefinition

#: Define :const:`mr_set`.
mr_set = mr_set


def _pop_scenario():
    """Initialize the scenario population"""
    pop_size = cfg.CONFIG["scenario"]["pop_size"]
    pop_scenario = [creator.Scenario(instance=mr_set.generate_scenario()) for _ in range(pop_size)]
    return pop_scenario


def _pop_perturbation():
    """Initialize the perturbation population."""
    pop_size = cfg.CONFIG["perturbation"]["pop_size"]
    pop_perturbation = [creator.Perturbation(mr_set.generate_perturbation()) for _ in range(pop_size)]

    return pop_perturbation


#: Define :const:`DOMAIN_REGISTRY`.
DOMAIN_REGISTRY = {
    "Scenario": ScenarioDefinition,
    "ScenarioInit": _pop_scenario,
    "Perturbation": Perturbations,
    "PerturbationInit": _pop_perturbation,
    "Evaluation": DomainEvaluator,
    "MRSet": mr_set,
    "Config": "config.yaml",
}
