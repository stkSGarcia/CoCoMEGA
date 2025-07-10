import random
import pickle
import sys
from deap import creator
from impl.ads.evaluation.simulation_runner import ADSEvaluator
from impl.ads.mr.predefined import mr_set1
from impl.ads.scenario.scenario_definition import ScenarioDefinition
from impl.core.mr.base_mr import Perturbations

from impl import config as cfg

mr_set = mr_set1
mr_set.labels = set(factory.get_label()
                    for mr in mr_set.mrs
                    for factory in mr.perturbation_factories
                    if factory.category in ScenarioDefinition.DYNAMIC)

# Load runtime scenarios.
if (cfg.CONFIG["search"]["runtime_data_as_seeds"] or
        cfg.CONFIG["search"]["constraint"]["enable"] or
        cfg.CONFIG["search"]["multi_objective"]["enable"]):
    runtime_scenarios = []

    import impl.ads.scenario.scenario_definition as scen_def_module
    sys.modules["impl.scenario.scenario_definition"] = scen_def_module

    for data_path in cfg.CONFIG["workspace"]["runtime_scenario"].rglob("*.*"):
        runtime_scenarios += pickle.loads(data_path.read_bytes())
    if len(runtime_scenarios) == 0:
        raise ValueError("No runtime scenarios found.")


def _pop_scenario():
    """Initialize the scenario population"""

    if cfg.CONFIG["search"]["runtime_data_as_seeds"]:
        pop_scenario = [creator.Scenario(scenario) for scenario in
                        random.choices(runtime_scenarios, k=cfg.CONFIG["scenario"]["pop_size"])]
    else:
        n = cfg.CONFIG["scenario"]["pop_size"] * cfg.CONFIG["scenario"]["init_selection_factor"]
        pop_scenario = [creator.Scenario(instance=mr_set.generate_scenario()) for _ in range(n)]
        if cfg.CONFIG["scenario"]["init_selection_factor"] > 1:
            pop_scenario = sorted(pop_scenario, key=lambda x: x.score(),
                                  reverse=True)[:cfg.CONFIG["scenario"]["pop_size"]]
    return pop_scenario


DOMAIN_REGISTRY = {
    "Scenario": ScenarioDefinition,
    "ScenarioInit": _pop_scenario,
    "Perturbation": Perturbations,
    "Evaluation": ADSEvaluator,
    "MRSet": mr_set1,
    "Config": "config.yaml",
}
