from typing import List

from deap import base, creator, tools

POP_SIZE = 10
CXPB = 0.8
MUTPB = 0.6

# Fitness functions:
# 1. maximize the number of MRs violated
# 2. minimize the length of the representation of the individual

creator.create("Fitness", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.Fitness, covered_objectives=list)

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, )  # TODO: add generator
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxSimulatedBinary, )
toolbox.register("mutate", tools.mutPolynomialBounded)
toolbox.register("select", tools.selNSGA2)


def _evaluate_population(individual, mrs: List):
    pass


toolbox.register("evaluate", _evaluate_population)


def _evaluate_population(self, population: List):
    """Evaluate each individual in the population."""
    for individual in population:
        simulation_results = list(map(self.evaluator, self._generate_follow_up_scenarios(individual)))
        individual.fitness.values = self.fitness_function(simulation_results)


def _generate_follow_up_scenarios(self, individual) -> List[Scenario]:
    scenarios = [deepcopy(scenario) for scenario in self.source_scenarios]
    for scenario in self.source_scenarios:
        for perturbation in individual:
            perturbation.perturb(scenario)
    return scenarios
