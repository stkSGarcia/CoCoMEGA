import math
from functools import partial

from deap import creator, base, tools

from impl.algorithm.base import Budget
from impl.config import CONFIG
from impl.mr.mr import Perturbations
from impl.mr.predefined import *
from impl.scenario.scenario_definition import ScenarioDefinition
from impl.scenario.simulation_runner import run_scenarios

# Define the metamorphic relation set.
mr_set = mr_set1

# Define the budget.
budget = Budget(max_sim=CONFIG["budget"]["max_sim"],
                max_time=CONFIG["budget"]["max_time"],
                max_gen=CONFIG["budget"]["max_gen"],
                convergence_threshold=CONFIG["budget"]["convergence_threshold"])

# Define individuals.
creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Solution", tuple, fitness=creator.Fitness, is_violated=False)
creator.create("Scenario", ScenarioDefinition, fitness=creator.Fitness)
creator.create("Perturbation", Perturbations, fitness=creator.Fitness)

toolbox = base.Toolbox()
toolbox.register("scenario", tools.initIterate, creator.Scenario, creator.Scenario.generate_random)
toolbox.register("perturbation", tools.initIterate, creator.Perturbation, lambda: [mr_set.spawn()])


def _pop_scenario():
    pop_scenario = tools.initRepeat(list, toolbox.scenario,
                                    n=CONFIG["scenario"]["pop_size"] * CONFIG["scenario"]["init_selection_factor"])
    if CONFIG["scenario"]["init_selection_factor"] > 1:
        pop_scenario = sorted(pop_scenario, key=lambda x: x.trajectory_collision_score(),
                              reverse=True)[:CONFIG["scenario"]["pop_size"]]
    return pop_scenario


toolbox.register("pop_scenario", _pop_scenario)
toolbox.register("pop_perturbation", tools.initRepeat, list, toolbox.perturbation, n=CONFIG["perturbation"]["pop_size"])

# Create a complete solution from two individuals.
toolbox.register("collaborate", lambda scenario, perturbation: creator.Solution((scenario, perturbation)))


# Define genetic operators.
def _determine_individual_type(individual):
    if str(type(individual)) == str(creator.Scenario):
        # if isinstance(individual, creator.Scenario):
        return 0
    elif str(type(individual)) == str(creator.Perturbation):
        # elif isinstance(individual, creator.Perturbation):
        return 1
    else:
        raise ValueError(f"Unrecognized individual type: {type(individual)}.")


toolbox.register("operators",
                 lambda individual: (  # Operators for scenarios.
                     partial(tools.selTournament, tournsize=CONFIG["scenario"]["tournament"]),  # selection
                     lambda ind1, ind2: ind1.mate(ind2),  # crossover
                     ScenarioDefinition.mutate,  # mutation
                     lambda ind: ind.assign_new_id()  # correction
                 ) if _determine_individual_type(individual) == 0 else (  # Operators for perturbations.
                     partial(tools.selTournament, tournsize=CONFIG["perturbation"]["tournament"]),  # selection
                     lambda ind1, ind2: ind1.mate(ind2),  # crossover
                     mr_set.mutate,  # mutation
                     lambda ind: ind,  # correction
                 ))


def _fitness(source, follow_up, mr_set=mr_set):
    """Calculate the fitness value and check if it violates the relation.

    @param source: Simulation results of the source scenario.
    @param follow_up: Simulation results of the follow-up scenario.
    @return: A tuple containing a bool value indicating whether it violates the relation and the fitness value.
    """
    field = mr_set.field()
    if field == "velocity":
        func = lambda row: math.sqrt(row.velocity_x ** 2 + row.velocity_y ** 2)
        source[field] = source.apply(func, axis=1, result_type="reduce")
        follow_up[field] = follow_up.apply(func, axis=1, result_type="reduce")
    is_violated, extent = mr_set.is_violated(source, follow_up)
    return is_violated, (extent,)


def _evaluate_solutions(solutions):
    """Evaluate the complete solutions.

    @return: A list of complete solutions evaluated and the number of simulations.
    """
    scenarios = []
    for solution in solutions:
        scenarios.append(solution[0])
        follow_up = toolbox.clone(solution[0])
        follow_up.assign_new_id()
        solution[1].perturb(follow_up)
        scenarios.append(follow_up)
    assert len(scenarios) == len(solutions) * 2
    results, sim_num = run_scenarios(scenarios)
    for solution, source, follow_up in zip(solutions, results[::2], results[1::2]):
        if source is not None and follow_up is not None:
            solution.is_violated, solution.fitness.values = _fitness(source, follow_up)
        else:
            solution.is_violated = False
            del solution.fitness.values
    return solutions, sim_num


def _evaluate_individual(individual, complete_solutions):
    """Evaluate the individual fitness of a scenario or a sequence of perturbations.

    @param individual: The individual to be evaluated.
    @param complete_solutions: The list of complete solutions with fitness evaluated.
    @return: The individual with fitness evaluated.
    """
    index = _determine_individual_type(individual)
    involved = []
    for solution in complete_solutions:
        if solution[index] == individual:
            involved.append(solution.fitness.values[0])
    if len(involved) > 0:
        individual.fitness.values = max(involved),
    else:
        del individual.fitness.values
    return individual


toolbox.register("evaluate_solutions", _evaluate_solutions)
toolbox.register("evaluate_individual", _evaluate_individual)
