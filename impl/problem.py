from copy import deepcopy
from math import factorial, sqrt
from types import SimpleNamespace

import numpy as np
from deap import creator, base, tools

from impl.algorithm.base import Budget
from impl.algorithm.rnsga2 import selRNSGA2WithMemory
from impl.algorithm.rnsga3 import selRNSGA3WithMemory
from impl.config import CONFIG
from impl.mr.mr import Perturbations
from impl.mr.predefined import *
from impl.scenario.scenario_definition import ScenarioDefinition
from impl.scenario.simulation_runner import run_scenarios

# Define the metamorphic relation set.
mr_set = mr_set1

# Define differential testing configurations.
diff_testing = True
reference_version = "v1"
test_version = "v2"

# Define the multi-objective configurations.
moo = True
rnsga3 = False
ref_points = np.array([[0.1, 0.5], [0.05, 0.6]])
n_obj = 2
if rnsga3:
    P = 5
    H = int(factorial(n_obj + P - 1) / (factorial(P) * factorial(n_obj - 1)))
    pop_size = ref_points.shape[0] * H + n_obj

# Define the budget.
budget = Budget(max_sim=CONFIG["budget"]["max_sim"],
                max_time=CONFIG["budget"]["max_time"],
                max_gen=CONFIG["budget"]["max_gen"])

# Define individuals.
creator.create("Fitness", base.Fitness, weights=(1.0, -1.0,) if moo else (1.0,))
creator.create("Solution", tuple, fitness=creator.Fitness, is_violated=False)
creator.create("Scenario", ScenarioDefinition, fitness=creator.Fitness)
creator.create("Perturbation", Perturbations, fitness=creator.Fitness)

toolbox = base.Toolbox()
toolbox.register("scenario", tools.initIterate, creator.Scenario, creator.Scenario.generate_random_or_leaderboard)
toolbox.register("perturbation", tools.initIterate, creator.Perturbation, mr_set.initialize)


def _pop_scenario():
    pop_scenario = tools.initRepeat(list, toolbox.scenario,
                                    n=CONFIG["scenario"]["pop_size"] * CONFIG["scenario"]["init_selection_factor"])
    if CONFIG["scenario"]["init_selection_factor"] > 1:
        pop_scenario = sorted(pop_scenario, key=lambda x: x.trajectory_score(),
                              reverse=True)[:CONFIG["scenario"]["pop_size"]]
    return pop_scenario


toolbox.register("pop_scenario", _pop_scenario)
toolbox.register("pop_perturbation", tools.initRepeat, list, toolbox.perturbation, n=CONFIG["perturbation"]["pop_size"])

# Create a complete solution from two individuals.
toolbox.register("collaborate", lambda scenario, perturbation: creator.Solution((scenario, perturbation)))
if moo:
    if rnsga3:
        toolbox.register("select", selRNSGA3WithMemory(ref_points=ref_points, p=P, mu=0.1))
    else:
        toolbox.register("select", selRNSGA2WithMemory(ref_points=ref_points))


# Define genetic operators.
def _fitness(source, follow_up, mr_set=mr_set):
    """Calculate the fitness value and check if it violates the relation.

    @param source: Simulation results of the source scenario.
    @param follow_up: Simulation results of the follow-up scenario.
    @return: A tuple containing a bool value indicating whether it violates the relation and the fitness value.
    """
    if mr_set.field == "velocity":
        func = lambda row: sqrt(row.velocity_x ** 2 + row.velocity_y ** 2)
        source[mr_set.field] = source.apply(func, axis=1, result_type="reduce")
        follow_up[mr_set.field] = follow_up.apply(func, axis=1, result_type="reduce")
    is_violated, extent = mr_set.is_violated(source, follow_up)
    return is_violated, (extent,) if extent is not None else None


def _evaluate_solutions(solutions):
    """Evaluate the complete solutions.

    @return: A list of complete solutions evaluated and the number of simulations.
    """
    scenarios = []
    reeval = []
    agent_name = CONFIG["agent"]["versions"][0]["name"]
    for solution in solutions:
        scenarios.append(solution[0])
        follow_up = toolbox.clone(solution[0])
        follow_up.assign_new_id()
        solution[1].perturb(follow_up)
        scenarios.append(follow_up)
    assert len(scenarios) == len(solutions) * 2
    results, sim_num = run_scenarios(scenarios, agent_name=agent_name)
    for solution, source, follow_up in zip(solutions, results[::2], results[1::2]):
        if source is not None and follow_up is not None:
            solution.source = source
            solution.follow_up = follow_up
            solution.is_violated, extent = _fitness(source, follow_up)
            if extent:
                solution.fitness.values = extent
                if extent[0] >= CONFIG["violation"]["reevaluation"]["threshold"]:
                    solution.reeval = True
                    reeval.append(solution)
                else:
                    solution.reeval = False
            else:
                del solution.fitness.values
        else:
            solution.is_violated = False
            del solution.fitness.values

    reeval_sim_num = _reevaluate(reeval)

    return solutions, sim_num + reeval_sim_num


def _evaluate_solutions_dt(solutions):
    """Evaluate the complete solutions (Differential Testing approach).

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

    rv_solutions, rv_sim_num = _perform_evaluation(solutions, scenarios, agent_name=reference_version)
    tv_solutions, tv_sim_num = _perform_evaluation(rv_solutions, scenarios, agent_name=test_version)
    for solution in solutions:
        rv_fitness = getattr(solution, reference_version).fitness
        tv_fitness = getattr(solution, test_version).fitness
        if rv_fitness and tv_fitness:
            solution.fitness.values = np.abs(rv_fitness.values[0] - tv_fitness.values[0])
            solution.fitness_type = reference_version if rv_fitness.values[0] > tv_fitness.values[0] else test_version
            solution.is_violated = True  # TODO define violation criteria
        else:
            del solution.fitness.values
            solution.is_violated = False
            solution.fitness_type = None

    return tv_solutions, rv_sim_num + tv_sim_num


def _perform_evaluation(solutions, scenarios, agent_name):
    reeval = []
    results, sim_num = run_scenarios(scenarios, agent_name=agent_name)
    for solution, source, follow_up in zip(solutions, results[::2], results[1::2]):
        eval_data = SimpleNamespace()
        if source is not None and follow_up is not None:
            eval_data.source = source
            eval_data.follow_up = follow_up
            eval_data.is_violated, extent = _fitness(source, follow_up)
            if extent:
                eval_data.fitness = extent
                if extent[0] >= CONFIG["violation"]["reevaluation"]["threshold"]:
                    eval_data.reeval = True
                    reeval.append(solution)
                else:
                    eval_data.reeval = False
            else:
                eval_data.fitness = None
        else:
            eval_data.is_violated = False
            eval_data.fitness = None
        setattr(solution, agent_name, eval_data)

    # reeval_sim_num = _reevaluate(reeval)

    return solutions, sim_num  # + reeval_sim_num


def _reevaluate(solutions):
    if len(solutions) == 0:
        return 0
    repeat = CONFIG["violation"]["reevaluation"]["repeat"]
    aggregation = CONFIG["violation"]["reevaluation"]["aggregation"]
    scenarios = []
    for solution in solutions:
        solution.eval_history = [{
            'source': solution.source.copy(),
            'follow_up': solution.follow_up.copy(),
            'is_violated': solution.is_violated,
            'fitness': deepcopy(solution.fitness),
        }]
        solution.aggregation = aggregation
        for repetition in range(1, repeat):
            source = toolbox.clone(solution[0])
            source.assign_new_id()
            scenarios.append(source)
            follow_up = toolbox.clone(solution[0])
            follow_up.assign_new_id()
            solution[1].perturb(follow_up)
            scenarios.append(follow_up)
    assert len(scenarios) == len(solutions) * 2 * (repeat - 1)
    results, sim_num = run_scenarios(scenarios, rerun=True)

    for i, source, follow_up in zip(range(len(solutions) * (repeat - 1)), results[::2], results[1::2]):
        solution = solutions[int(i / (repeat - 1))]
        fitness = deepcopy(solution.fitness)
        if source is not None and follow_up is not None:
            is_violated, extent = _fitness(source, follow_up)
            if extent:
                fitness.values = extent
            else:
                del fitness.values
        else:
            is_violated = False
            del fitness.values

        solution.eval_history.append({
            'source': source,
            'follow_up': follow_up,
            'is_violated': is_violated,
            'fitness': fitness,
        })

    for solution in solutions:
        fitnesses = [(ev['fitness'].values[0] if ev['fitness'].valid else np.nan) for ev in solution.eval_history]
        num_nan_fitnesses = len([f for f in fitnesses if np.isnan(f)])
        if num_nan_fitnesses > float(repeat) / 2:
            del solution.fitness.values
            solution.source = None
            solution.follow_up = None
            solution.is_violated = False
        else:
            aggregate_value = getattr(np, f'nan{aggregation}')(fitnesses)
            aggregation_arg = np.nanargmin(np.abs([f - aggregate_value for f in fitnesses]))
            selected_candidate = solution.eval_history[aggregation_arg]

            solution.fitness.values = (aggregate_value,)
            solution.source = selected_candidate['source']
            solution.follow_up = selected_candidate['follow_up']
            solution.is_violated = selected_candidate['is_violated']

    return sim_num


def _evaluate_individual(individual, complete_solutions):
    """Evaluate the individual fitness of a scenario or a sequence of perturbations.

    @param individual: The individual to be evaluated.
    @param complete_solutions: The list of complete solutions with fitness evaluated.
    @return: The individual with fitness evaluated.
    """
    index = 0 if str(type(individual)) == str(creator.Scenario) else 1
    involved = []
    for solution in complete_solutions:
        if solution[index] == individual:
            involved.append(solution.fitness.values[0])  # FIXME: multi-objective.
    if len(involved) > 0:
        individual.fitness.values = max(involved),
    else:
        del individual.fitness.values
    return individual


toolbox.register("evaluate_solutions", _evaluate_solutions_dt if diff_testing else _evaluate_solutions)
toolbox.register("evaluate_individual", _evaluate_individual)
