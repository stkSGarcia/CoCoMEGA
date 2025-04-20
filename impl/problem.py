import pickle
import random
from copy import deepcopy
from math import factorial, sqrt
from types import SimpleNamespace

import numpy as np
from deap import creator, base, tools

from impl import config as cfg
from impl.algorithm.base import Budget
from impl.algorithm.rnsga2 import selRNSGA2WithMemory
from impl.algorithm.rnsga3 import selRNSGA3WithMemory
from impl.mr.mr import Perturbations
from impl.mr.predefined import *
from impl.scenario.scenario_definition import ScenarioDefinition
from impl.scenario.simulation_runner import run_scenarios

# Define the metamorphic relation set.
mr_set = mr_set1

# Define differential testing configurations.
reference_version = "v1"
test_version = "v2"

# Define the budget.
budget = Budget(max_sim=cfg.CONFIG["search"]["budget"]["max_sim"],
                max_time=cfg.CONFIG["search"]["budget"]["max_time"],
                max_gen=cfg.CONFIG["search"]["budget"]["max_gen"])

# Define individuals.
weights = (1.0, -1.0,) if cfg.CONFIG["search"]["multi_objective"]["enable"] else (1.0,)
creator.create("Fitness", base.Fitness, weights=weights)
creator.create("Solution", tuple, fitness=creator.Fitness, is_violated=False)
creator.create("Scenario", ScenarioDefinition, fitness=creator.Fitness)
creator.create("Perturbation", Perturbations, fitness=creator.Fitness)

# Load runtime scenarios.
if (cfg.CONFIG["search"]["runtime_data_as_seeds"] or
        cfg.CONFIG["search"]["constraint"]["enable"] or
        cfg.CONFIG["search"]["multi_objective"]["enable"]):
    runtime_scenarios = []
    for data_path in cfg.CONFIG["workspace"]["runtime_scenario"].rglob("*.*"):
        runtime_scenarios += pickle.loads(data_path.read_bytes())
    runtime_scenarios = [creator.Scenario(scenario) for scenario in runtime_scenarios]
    if len(runtime_scenarios) == 0:
        raise ValueError("No runtime scenarios found.")

toolbox = base.Toolbox()
toolbox.register("scenario", tools.initIterate, creator.Scenario, creator.Scenario.generate_random_or_leaderboard)
# toolbox.register("scenario", tools.initIterate, creator.Scenario, creator.Scenario.generate_random_with_marked_actors)
toolbox.register("perturbation", tools.initIterate, creator.Perturbation, mr_set.initialize)


def _pop_scenario():
    pop_scenario = tools.initRepeat(
        list, toolbox.scenario,
        n=cfg.CONFIG["scenario"]["pop_size"] * cfg.CONFIG["scenario"]["init_selection_factor"],
    )
    if cfg.CONFIG["scenario"]["init_selection_factor"] > 1:
        pop_scenario = sorted(pop_scenario, key=lambda x: x.trajectory_score(),
                              reverse=True)[:cfg.CONFIG["scenario"]["pop_size"]]
    return pop_scenario


if cfg.CONFIG["search"]["runtime_data_as_seeds"]:
    toolbox.register("pop_scenario", lambda: random.choices(runtime_scenarios, k=cfg.CONFIG["scenario"]["pop_size"]))
else:
    toolbox.register("pop_scenario", _pop_scenario)
toolbox.register("pop_perturbation", tools.initRepeat, list, toolbox.perturbation,
                 n=cfg.CONFIG["perturbation"]["pop_size"])

# Create a complete solution from two individuals.
toolbox.register("collaborate", lambda scenario, perturbation: creator.Solution((scenario, perturbation)))

# Define the multi-objective configurations.
if cfg.CONFIG["search"]["multi_objective"]["enable"]:
    ref_points = np.array(cfg.CONFIG["search"]["multi_objective"]["ref_points"])
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


# Define genetic operators.
def _fitness(source, follow_up, mr_set=mr_set):
    """Calculate the fitness value and check if it violates the relation.

    :param source: Simulation results of the source scenario.
    :param follow_up: Simulation results of the follow-up scenario.
    :return: A tuple containing a bool value indicating whether it violates the relation and the fitness value.
    """
    if mr_set.field == "velocity":
        func = lambda row: sqrt(row.velocity_x ** 2 + row.velocity_y ** 2)
        source[mr_set.field] = source.apply(func, axis=1, result_type="reduce")
        follow_up[mr_set.field] = follow_up.apply(func, axis=1, result_type="reduce")
    is_violated, extent = mr_set.is_violated(source, follow_up)
    return is_violated, (extent,) if extent is not None else None


def _calculate_similarity(scenario, scenarios):
    """Calculate the heterogeneous distance between the given scenario and the list of scenarios.

    :param scenario: The given scenario.
    :param scenarios: A list of scenarios.
    :return: The heterogeneous distance.
    """
    return min([scenario.dist(scen) for scen in scenarios])


def _evaluate_solutions(solutions):
    """Evaluate the complete solutions.

    :return: A list of complete solutions evaluated and the number of simulations.
    """
    scenarios = []
    reeval = []
    agent_name = cfg.CONFIG["interfuser"]["versions"][0]["name"]
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
                if extent[0] >= cfg.CONFIG["violation"]["reevaluation"]["threshold"]:
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

    :return: A list of complete solutions evaluated and the number of simulations.
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
    if cfg.CONFIG["search"]["constraint"]["enable"] or cfg.CONFIG["search"]["multi_objective"]["enable"]:
        similarities = [_calculate_similarity(scenario, runtime_scenarios) for scenario in scenarios]
    for i, solution in enumerate(solutions):
        rv_fitness = getattr(solution, reference_version).fitness
        tv_fitness = getattr(solution, test_version).fitness
        if rv_fitness and tv_fitness:
            solution.fitness.values = (np.abs(rv_fitness[0] - tv_fitness[0]),)
            solution.fitness_type = reference_version if rv_fitness[0] > tv_fitness[0] else test_version
            solution.is_violated = solution.fitness.values[0] > 0
        else:
            del solution.fitness.values
            solution.fitness_type = None
            solution.is_violated = False
        if cfg.CONFIG["search"]["constraint"]["enable"]:
            solution.similarity = (similarities[i * 2], similarities[i * 2 + 1])
        # if not rv_fitness and not tv_fitness:
        #     diff, typ = 0, None
        # else:
        #     if not rv_fitness:
        #         diff, typ = np.abs(tv_fitness[0]), test_version
        #     elif not tv_fitness:
        #         diff, typ = np.abs(rv_fitness[0]), reference_version
        #     else:
        #         diff = np.abs(rv_fitness[0] - tv_fitness[0])
        #         typ = reference_version if rv_fitness[0] > tv_fitness[0] else test_version
        # if cfg.CONFIG["search"]["constraint"]["enable"] or cfg.CONFIG["search"]["multi_objective"]["enable"]:
        #     solution.similarity = (similarities[i * 2], similarities[i * 2 + 1])
        # if cfg.CONFIG["search"]["multi_objective"]["enable"]:
        #     solution.fitness.values = (diff, min(solution.similarity))
        # else:
        #     solution.fitness.values = (diff,)
        # solution.fitness_type = typ
        # solution.is_violated = diff > 0

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
                if extent[0] >= cfg.CONFIG["violation"]["reevaluation"]["threshold"]:
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
    repeat = cfg.CONFIG["violation"]["reevaluation"]["repeat"]
    aggregation = cfg.CONFIG["violation"]["reevaluation"]["aggregation"]
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


def _penalize(fitness, similarity):
    """Penalize the fitness value according to the similarity.

    :param fitness: The fitness value.
    :param similarity: The similarity.
    :return: Penalized fitness value.
    """
    if similarity <= cfg.CONFIG["search"]["constraint"]["threshold"]:
        return (fitness[0],)
    penalty = cfg.CONFIG["search"]["constraint"]["penalty_base"] ** (
            cfg.CONFIG["search"]["constraint"]["penalty_amplifier"] *
            (similarity - cfg.CONFIG["search"]["constraint"]["threshold"])
    )
    return (fitness[0] / penalty,)


def _evaluate_individual(individual, complete_solutions):
    """Evaluate the individual fitness of a scenario or a sequence of perturbations.

    :param individual: The individual to be evaluated.
    :param complete_solutions: The list of complete solutions with fitness evaluated.
    :return: The individual with fitness evaluated.
    """
    index = 0 if str(type(individual)) == str(creator.Scenario) else 1
    involved = []
    for solution in complete_solutions:
        if solution[index] == individual and solution.fitness.valid:
            if cfg.CONFIG["search"]["constraint"]["enable"] or cfg.CONFIG["search"]["multi_objective"]["enable"]:
                involved.append((solution.fitness.values[0], solution.similarity[index]))
            else:
                involved.append(solution.fitness.values)
    if len(involved) > 0:
        sorted_involved = sorted(involved, key=lambda x: tuple(np.array(x) * -np.array(weights)))
        if cfg.CONFIG["search"]["constraint"]["enable"]:
            individual.fitness.values = _penalize(sorted_involved[0], np.min(sorted_involved, axis=0)[1])
        else:
            individual.fitness.values = sorted_involved[0]
    else:
        del individual.fitness.values
    return individual


if cfg.CONFIG["search"]["diff_testing"]:
    toolbox.register("evaluate_solutions", _evaluate_solutions_dt)
else:
    toolbox.register("evaluate_solutions", _evaluate_solutions)
toolbox.register("evaluate_individual", _evaluate_individual)
