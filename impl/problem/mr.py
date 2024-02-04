from deap import creator, base, tools

POP_SIZE = 10
CXPB = 0.8
MUTPB = 0.6

# Fitness functions:
# 1. maximize the extent of violation.
# 2. minimize the length of the perturbation sequence.
creator.create("Fitness", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.Fitness, covered_objectives=list)

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, )  # TODO: add generator
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxSimulatedBinary, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, eta=20.0, indpb=MUTPB)
toolbox.register("select", tools.selNSGA2)
