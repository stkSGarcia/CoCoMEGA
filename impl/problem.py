from copy import deepcopy
from math import factorial

import numpy as np
from deap import creator, base, tools

from impl import config as cfg
from impl.core.algorithm.base import Budget
from impl.core.algorithm.rnsga2 import selRNSGA2WithMemory
from impl.core.algorithm.rnsga3 import selRNSGA3WithMemory
from impl.core.domain_factory import DomainFactory

# Define domain
# domain = DomainFactory("ads")
domain = DomainFactory("sorting")

# Define the metamorphic relation set.
mr_set = domain.get_mr_set()

# Define the budget.
budget = Budget(max_sim=cfg.CONFIG["search"]["budget"]["max_sim"],
                max_time=cfg.CONFIG["search"]["budget"]["max_time"],
                max_gen=cfg.CONFIG["search"]["budget"]["max_gen"])

weights = (1.0, -1.0,) if cfg.CONFIG["search"]["multi_objective"]["enable"] else (1.0,)

# Define individuals.
creator.create("Fitness", base.Fitness, weights=weights)
creator.create("Solution", tuple, fitness=creator.Fitness, is_violated=False)
creator.create("Scenario", domain.get_scenario(), fitness=creator.Fitness)
creator.create("Perturbation", domain.get_perturbation(), fitness=creator.Fitness)

toolbox = base.Toolbox()
toolbox.register("scenario", tools.initIterate, creator.Scenario, mr_set.generate_scenario)
toolbox.register("perturbation", tools.initIterate, creator.Perturbation, mr_set.generate_perturbation)
toolbox.register("pop_scenario", domain.get_scenario_init())
toolbox.register("pop_perturbation", tools.initRepeat, list, toolbox.perturbation,
                 n=cfg.CONFIG["perturbation"]["pop_size"])
toolbox.register("evaluator", domain.get_evaluation())

# Create a complete solution from two individuals.
toolbox.register("collaborate", lambda scenario, perturbation: creator.Solution((scenario, perturbation)))

# Define the multi-objective configurations.
if cfg.CONFIG["search"]["multi_objective"]["enable"]:
    ref_points = np.array(cfg.CONFIG["search"]["multi_objective"]["ref_points"])  # Reference points.
    if cfg.CONFIG["search"]["multi_objective"]["algorithm"] == "rnsga3":
        P = 5
        n_obj = 2
        H = int(factorial(n_obj + P - 1) / (factorial(P) * factorial(n_obj - 1)))
        pop_size = ref_points.shape[0] * H + n_obj
        toolbox.register("select_scenario", selRNSGA3WithMemory(ref_points=ref_points, p=P, mu=0.1),
                         k=cfg.CONFIG["scenario"]["pop_size"])
        toolbox.register("select_perturbation", selRNSGA3WithMemory(ref_points=ref_points, p=P, mu=0.1),
                         k=cfg.CONFIG["perturbation"]["pop_size"])
    else:
        toolbox.register("select_scenario", selRNSGA2WithMemory(ref_points=ref_points),
                         k=cfg.CONFIG["scenario"]["pop_size"])
        toolbox.register("select_perturbation", selRNSGA2WithMemory(ref_points=ref_points),
                         k=cfg.CONFIG["perturbation"]["pop_size"])
