import pickle
import random
import os
import sys
from impl import config as cfg

# Add InterFuser to the path
for path in [
    "carla/PythonAPI",
    "carla/PythonAPI/carla",
    "carla/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg",
    "leaderboard",
    "leaderboard/team_code",
    "scenario_runner",
]:
    sys.path.append(os.path.join(cfg.CONFIG["interfuser"]["repo"], path))

from deap import creator

from impl.ads.evaluation.simulation_runner import ADSEvaluator
from impl.ads.mr.predefined import mr_set1, mr_set3
from impl.ads.scenario.scenario_definition import ScenarioDefinition
from impl.core.algorithm.base import BaseAlgorithm
from impl.core.mr.base_mr import Perturbations

# FIXME: Workaround for scenario definition restructure.
import impl.ads.mr.mr as base_mr_module
import impl.ads.scenario.scenario_definition as scen_def_module

sys.modules["impl.scenario.scenario_definition"] = scen_def_module
sys.modules["impl.mr.mr"] = base_mr_module

#: Defined :const:`mr_set`.
mr_set = mr_set1
# mr_set = mr_set3
mr_set.labels = set(factory.get_label()
                    for mr in mr_set.mrs
                    for factory in mr.perturbation_factories
                    if factory.category in ScenarioDefinition.DYNAMIC)

# Load runtime scenarios.
if (cfg.CONFIG["search"]["seeds"]["source"] == "runtime" or
        cfg.CONFIG["search"]["constraint"]["enable"] or
        cfg.CONFIG["search"]["multi_objective"]["enable"]):
    runtime_scenarios = []
    if cfg.CONFIG["search"]["seeds"]["version"] == 1.0:
        import impl.ads.scenario.scenario_definition as scen_def_module

        sys.modules["impl.scenario.scenario_definition"] = scen_def_module

        for data_path in cfg.CONFIG["workspace"]["runtime_scenario"].rglob("*.*"):
            runtime_scenarios += pickle.loads(data_path.read_bytes())
        if len(runtime_scenarios) == 0:
            raise ValueError("No runtime scenarios found.")
    elif cfg.CONFIG["search"]["seeds"]["version"] == 2.0:
        rt_root = cfg.CONFIG["workspace"]["realtime_data"] / cfg.CONFIG["search"]["seeds"]["name"]
        if not rt_root.is_dir():
            raise ValueError(f"Runtime scenario directory {rt_root} not found.")
        for data_path in rt_root.rglob("*.pkl"):
            runtime_scenarios.append(pickle.loads(data_path.read_bytes())["scenario"])

# Load seed solutions.
if cfg.CONFIG["search"]["seeds"]["source"] == "previous":
    seed_solutions = []
    for data_path in cfg.CONFIG["workspace"]["previous_solution"].rglob("*.*"):
        seed_solutions += pickle.loads(data_path.read_bytes())
    if len(seed_solutions) == 0:
        raise ValueError("No previous solutions found.")
    seed_solutions = BaseAlgorithm.dpp_sample(seed_solutions, k=min(
        len(seed_solutions),
        max(
            cfg.CONFIG["scenario"]["pop_size"],
            cfg.CONFIG["perturbation"]["pop_size"]
        ) * cfg.CONFIG["scenario"]["init_selection_factor"]
    ))
    seed_scenarios, seed_perturbations = zip(*seed_solutions)


def _pop_scenario():
    """Initialize the scenario population."""
    pop_size = cfg.CONFIG["scenario"]["pop_size"]
    pop_scenario = []

    if cfg.CONFIG["search"]["seeds"]["source"] is not None:
        base_scenarios = runtime_scenarios if cfg.CONFIG["search"]["seeds"]["source"] == "runtime" else seed_scenarios
        unique = BaseAlgorithm.remove_duplicates(base_scenarios)
        pop_scenario += [creator.Scenario(scenario) for scenario in (
            random.sample(unique, pop_size) if len(unique) > pop_size else unique
        )]

    if len(pop_scenario) < pop_size:
        remaining_size = pop_size - len(pop_scenario)
        random_scenarios = [creator.Scenario(instance=mr_set.generate_scenario())
                            for _ in range(remaining_size * cfg.CONFIG["scenario"]["init_selection_factor"])]
        if cfg.CONFIG["scenario"]["init_selection_factor"] > 1:
            random_scenarios = sorted(random_scenarios, key=lambda x: x.score(), reverse=True)[:remaining_size]
        pop_scenario += random_scenarios

    return pop_scenario


def _pop_perturbation():
    """Initialize the perturbation population."""
    pop_size = cfg.CONFIG["perturbation"]["pop_size"]
    pop_perturbation = []

    if cfg.CONFIG["search"]["seeds"]["source"] == "previous":
        unique = BaseAlgorithm.remove_duplicates(seed_perturbations)
        pop_perturbation += [creator.Perturbation(perturbation) for perturbation in (
            random.sample(unique, pop_size) if len(unique) > pop_size else unique
        )]

    if len(pop_perturbation) < pop_size:
        pop_perturbation += [creator.Perturbation(mr_set.generate_perturbation())
                             for _ in range(pop_size - len(pop_perturbation))]

    return pop_perturbation


#: Defined :const:`DOMAIN_REGISTRY`.
DOMAIN_REGISTRY = {
    "Scenario": ScenarioDefinition,
    "ScenarioInit": _pop_scenario,
    "Perturbation": Perturbations,
    "PerturbationInit": _pop_perturbation,
    "Evaluation": ADSEvaluator,
    "MRSet": mr_set,
}
