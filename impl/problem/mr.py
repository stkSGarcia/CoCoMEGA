from deap import creator, base, tools

POP_SIZE = 10
CXPB = 0.8
MUTPB = 0.6

# Fitness functions:
# 1. maximize the extent of violation
# 2. minimize the length of the perturbation sequence.
creator.create("Fitness", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.Fitness, covered_objectives=list)

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, )  # TODO: add generator
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxSimulatedBinary, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, eta=20.0, indpb=MUTPB)
toolbox.register("select", tools.selNSGA2)


def mutate(self, mutpb, *individuals):
    """Mutates individuals according to the probability `mutpb`.

    @param mutpb: mutation probability
    """
    for individual in individuals:
        # add perturbations
        times = 1
        while random.random() < mutpb ** times:
            perturbation = random.choice(self.mrs).generate_perturbation()
            individual.append(perturbation)
            times += 1

        # TODO: Squash perturbations
