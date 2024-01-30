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
