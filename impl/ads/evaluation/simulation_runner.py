import itertools
import logging
import os
import pickle
import shutil
import subprocess
import sys
import json
import traceback
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from math import sqrt
from multiprocessing import Manager, Process
from types import SimpleNamespace
import numpy as np
import pandas as pd
from deap import tools, creator

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from impl import config as cfg
from impl.ads.utils.visualization import visualize_violation
from impl.ads.evaluation.exceptions import InvalidScenarioDefinitionError
from impl.ads.scenario.scenario_definition import ScenarioDefinition
from impl.ads.utils.carla_utils import initialize_carla
from impl.ads.utils.process_utils import run_silently
from impl.core.evaluation.base_evaluation import BaseEvaluator
from impl.core.scenario.base_scenario import AbstractScenarioDefinition

arguments = [
    ("SCENARIOS", "scenarios",
     os.path.join(cfg.CONFIG["interfuser"]["repo"], "leaderboard/data/scenarios/town05_all_scenarios.json")),
    ("ROUTES", "routes",
     os.path.join(cfg.CONFIG["interfuser"]["repo"], "leaderboard/data/training_routes/routes_town05_long.xml")),
    ("REPETITIONS", "repetitions", 1),
    ("CHALLENGE_TRACK_CODENAME", "track", "SENSORS"),
    # ("CHECKPOINT_ENDPOINT", "checkpoint", os.path.join(CONFIG["workspace"]["sim_result"], "checkpoint.json")),
    ("TEAM_AGENT", "agent", "impl/ads/agent/interfuser_agent.py"),
    ("TEAM_CONFIG", "agent_config", "impl/ads/agent/interfuser_config_v1.py"),
    ("DEBUG_CHALLENGE", "debug", 0),
    ("RESUME", "resume", False),
    ("SAVE_PATH", None, None),
    (None, "trafficManagerSeed", "1"),
    (None, "carlaProviderSeed", "2000"),
    (None, "record", ""),
    (None, "timeout", "600.0"),
]

for env, _, v in arguments:
    if env is not None and v is not None:
        os.environ[env] = str(v)

from impl.ads.evaluation.interfuser_scenario_evaluator import ScenarioEvaluator

logger = logging.getLogger(__name__)
config = type("", (object,), {arg: value for _, arg, value in arguments})()
evaluated_scenarios = Manager().dict()
carla_host = carla_port = tm_port = gpu_device = tag = None


def _init_carla(instance_configs):
    """
    Initialize CARLA connection based on provided instance configuration.

    This sets host, port, traffic manager port, and GPU device.
    Cleans previous CARLA actors and prepares environment.

    :param instance_configs: Queue providing instance settings.
    """
    global carla_host, carla_port, tm_port, gpu_device, tag
    carla_host, carla_port, tm_port, gpu_device, tag = instance_configs.get(timeout=10)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    CarlaDataProvider.cleanup()
    initialize_carla(carla_host, carla_port, tm_port, gpu_device, tag=tag)


def run_free_environments(confs):
    """
    Run multiple environment instances in parallel or sequentially.

    :param confs: List of configuration dictionaries for each environment run.
    :return: A list of results from each environment.
    """
    tag = os.environ.get("tag", "default")
    if cfg.CONFIG["runtime"]["parallel"]:
        process_configs = Manager().Queue()
        for instance in cfg.CONFIG["runtime"]["instances"]:
            process_configs.put(
                (instance["host"], instance["port"], instance["tm_port"], instance["gpu_device"], tag))
        with ProcessPoolExecutor(max_workers=len(cfg.CONFIG["runtime"]["instances"]),
                                 initializer=_init_carla, initargs=(process_configs,)) as executor:
            results = executor.map(run_environment, confs)
    else:
        global carla_host, carla_port, tm_port, gpu_device
        instance = cfg.CONFIG["runtime"]["instances"][0]
        carla_host, carla_port, tm_port, gpu_device = (instance["host"], instance["port"],
                                                       instance["tm_port"], instance["gpu_device"])
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_device)
        CarlaDataProvider.cleanup()
        initialize_carla(carla_host, carla_port, tm_port, gpu_device, tag=tag)
        results = map(run_environment, confs)
        results = zip(*results)
    return results


def run_environment(conf):
    """
    Run a single simulation environment with the given configuration.

    Sets environment variables and executes a subprocess to run the simulation.

    :param conf: Configuration dictionary for one environment instance.
    :return: Process return code (:data:`0` for success).
    """
    global carla_host, carla_port, tm_port

    cp_path, output_path, dataset_path = "", "", ""
    if conf.get("cp_root", None):
        cp_path = os.path.join(conf["cp_root"], f"weather-{conf['weather']}", f"{conf['route_name']}.json")
        os.makedirs(os.path.dirname(cp_path), exist_ok=True)

    if conf.get("output_root", None):
        output_path = os.path.join(conf["output_root"], f"weather-{conf['weather']}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if conf.get("dataset_path", None):
        dataset_path = conf["dataset_path"]
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

    scenarios = os.path.join(cfg.CONFIG["interfuser"]["repo"], "leaderboard", "data", conf["scenario"])
    routes = os.path.join(cfg.CONFIG["interfuser"]["repo"], "leaderboard", "data", conf["route"])

    child_env = os.environ.copy()

    # Set environment variables as in the bash script
    child_env.update({
        "DATA_ROOT": str(cfg.CONFIG["workspace"]["realtime_data"]),
        "CARLA_ROOT": os.path.join(cfg.CONFIG["interfuser"]["repo"], "carla"),
        "CARLA_SERVER": os.path.join(cfg.CONFIG["interfuser"]["repo"], "carla", "CarlaUE4.sh"),
        "CARLA_WEATHER": str(conf["weather"]),
        "LEADERBOARD_ROOT": os.path.join(cfg.CONFIG["interfuser"]["repo"], "leaderboard"),
        "CHECKPOINT_ENDPOINT": cp_path,
        "SAVE_PATH": output_path,
        "DATASET_PATH": dataset_path,
        "TRAFFIC_SEED": "2000",
        "CARLA_SEED": "2000",
        "SCENARIOS": scenarios,
        "ROUTES": routes,
        "TM_PORT": str(tm_port),
        "PORT": str(carla_port),
        "HOST": carla_host,
        "CHALLENGE_TRACK_CODENAME": "SENSORS",
        "DEBUG_CHALLENGE": "0",
        "REPETITIONS": "1",
        "TEAM_AGENT": conf["agent_path"],
        "RESUME": "True",
        "COLLECTION_DELAY_LOWER": conf.get("collection_delay_lower", "None"),
        "COLLECTION_DELAY_UPPER": conf.get("collection_delay_upper", "None"),
        "COLLECTION_DURATION": conf.get("collection_duration", "None"),
        "COLLECTION_INTERVAL": conf.get("collection_interval", "None"),
        "SUBMITION_ROUTE_LIMIT": conf.get("submition_route_limit", "None"),
        "DISCARD_ORIGINAL_FEATURES": conf.get("discard_original_features", "None"),
        "NOVELTY_STRATEGY": conf.get("novelty_strategy", "None"),
        # Add RECORD_PATH if needed
        # "RECORD_PATH": "path/to/record",
    })

    pp_parent = os.environ.get("PYTHONPATH", "")
    pp_added = os.pathsep.join(sys.path)
    child_env["PYTHONPATH"] = os.pathsep.join([p for p in [pp_parent, pp_added] if p])

    command = [
        # f"{sys.executable} {os.path.join(CONFIG['interfuser']['repo'], 'leaderboard/leaderboard/leaderboard_evaluator.py')}"
        f"{sys.executable}", f"{os.path.join('impl', 'ads', 'evaluation', 'leaderboard_evaluator.py')}",
        f"--scenarios", f"{scenarios}",
        f"--routes", f"{routes}",
        f"--repetitions", f"1",
        f"--track", f"SENSORS",
        f"--checkpoint", f"{cp_path}",
        f"--agent", f"{conf['agent_path']}",
        f"--agent-config", f"{conf['agent_config']}",
        f"--debug", "0",
        f"--resume", "True",
        f"--port", f"{carla_port}",
        f"--host", f"{carla_host}",
        f"--trafficManagerPort", f"{tm_port}",
        f"--carlaProviderSeed", "2000",
        f"--trafficManagerSeed", "2000",
    ]

    process = subprocess.run(command, env=child_env, check=True, shell=False, text=True, stdout=None, stderr=None)

    return process.returncode


def run_solutions(alg, project, mr_set, top: int = -1, verbose=True, agent_names=["v1"], record_video=False):
    """
    Load and run scenarios from a saved solution file.

    Each solution typically contains a source and a follow-up scenario.

    :param alg: Algorithm name.
    :param project: Project name.
    :param mr_set: Metamorphic relation set to evaluate violations.
    :param top: Number of top solutions to run (:data:`-1` runs all).
    :param verbose: If :data:`True`, run with output logs; otherwise silent.
    :param agent_names: Agents to use.
    :param record_video: Whether to record video outputs for the runs.
    """
    cfg.init_project_directory(alg, resume=False)
    solution_file = next((cfg.CONFIG["workspace"]["result"] / project / "solutions").rglob("solutions*"), None)
    if solution_file is None:
        logger.warning(f"No solution file found for {project}.")
        return
    print("solution_file:", solution_file, "exists:", solution_file.exists())
    print("parent:", solution_file.parent, "glob:", list(solution_file.parent.glob("*"))[:10])
    solutions = pickle.loads(solution_file.read_bytes())
    solutions = tools.selBest([ind for ind in solutions], top)
    additional_confs = None
    for i, sol in enumerate(solutions):
        fitness_data = {
            "old": {"diff": sol.fitness.values[0]},
            "new": {},
        }
        source, perturbations = sol[0], sol[1]
        solution_identifier = f"{project}-{i + 1}"
        solution_base = cfg.CONFIG["workspace"]["recordings"] / solution_identifier
        if record_video and already_recorded(solution_base, agent_names=agent_names):
            logger.info(f"Solution {solution_identifier} is already recorded, skipping...")
            continue
        for agent in agent_names:
            source_id = f"top{i + 1}_source_{agent}"
            follow_up_id = f"top{i + 1}_follow_up_{agent}"
            agent_data = getattr(sol, agent, None)
            fitness_data["old"][agent] = agent_data.fitness[0]
            if record_video:
                solution_path = cfg.CONFIG["workspace"]["recordings"] / solution_identifier / agent
                os.makedirs(solution_path, exist_ok=True)

                additional_confs = [
                    {"recording_save_path": solution_path / "source"},
                    {"recording_save_path": solution_path / "follow-up"},
                ]
                # Visualize Violation

                if agent_data is None:
                    logger.warning(f"Solution {solution_identifier} does not have data for agent {agent}.")
                else:
                    visualize_violation(agent_data.source, agent_data.follow_up, mr_set=mr_set,
                                        save_path=solution_path / "old_violation.png")

            if solution_path.exists():
                shutil.rmtree(solution_path)
            logger.info(f"Running solution {solution_identifier} for agent {agent}...")
            source.id_ = source_id
            follow_up = deepcopy(source)
            follow_up.id_ = follow_up_id
            perturbations.perturb(follow_up)
            logger.info(f"Running solution {solution_identifier}...")
            if verbose:
                run_process = Process(target=ADSEvaluator(mr_set=None).run_scenarios,
                                      args=([source, follow_up], agent, True, additional_confs))
            else:
                run_process = Process(target=run_silently,
                                      args=(ADSEvaluator(mr_set=None).run_scenarios, [source, follow_up], agent,
                                            True,
                                            additional_confs))
            run_process.start()
            run_process.join()

            result_dfs = []
            for scen_id in (source_id, follow_up_id):
                result_path = cfg.CONFIG["workspace"]["sim_result"] / f"{scen_id}.csv"
                result = pd.read_csv(result_path)
                result.set_index(result.columns[0], inplace=True)
                result_dfs.append(result)

            fitness_val = ADSEvaluator(mr_set=mr_set).fitness(result_dfs[0], result_dfs[1])[1]
            fitness_data["new"][agent] = fitness_val[0] if fitness_val is not None else np.nan
            visualize_violation(result_dfs[0], result_dfs[1], mr_set=mr_set,
                                save_path=solution_path / "new_violation.png")
        fitness_data["new"]["diff"] = np.abs(fitness_data["new"][agent_names[0]] - fitness_data["new"][agent_names[1]])

        fitness_path = cfg.CONFIG["workspace"]["recordings"] / solution_identifier / "fitness.json"
        with open(fitness_path, "w") as f:
            json.dump(fitness_data, f, indent=4)


def already_recorded(solution_path, agent_names):
    """
    Check if a recording already exists for a given solution path.

    :param solution_path: Path where the source and follow-up recordings are expected.
    :return: :data:`True` if already recorded, :data:`False` otherwise.
    """
    recorded = True
    for agent in agent_names:
        try:
            if not (len(os.listdir(solution_path / agent / "source")) > 0 and len(
                    os.listdir(solution_path / agent / "follow-up")) > 0):
                recorded = False
        except FileNotFoundError:
            recorded = False
    return recorded

    try:
        if len(os.listdir(solution_path / "source")) > 0 and len(os.listdir(solution_path / "follow-up")) > 0:
            return True
    except FileNotFoundError:
        pass
    return False


class ADSEvaluator(BaseEvaluator):

    def __init__(self, mr_set):

        super().__init__(mr_set)

    def evaluate_solutions(self, solutions):
        scenarios = self._get_scenarios(solutions)

        if cfg.CONFIG["search"]["constraint"]["enable"] or cfg.CONFIG["search"]["multi_objective"]["enable"]:
            from impl.ads.register import runtime_scenarios
            similarities = [ADSEvaluator._calculate_similarity(scenario, [creator.Scenario(scenario) for scenario in
                                                                          runtime_scenarios]) for scenario in
                            scenarios]
            for i, solution in enumerate(solutions):
                solution.similarity = (similarities[i * 2], similarities[i * 2 + 1])

        if cfg.CONFIG["search"]["diff_testing"]["enabled"]:
            return self._evaluate_solutions_dt(solutions, scenarios)
        else:
            return self._evaluate_solutions(solutions, scenarios)

    def evaluate_individual(self, individual, complete_solutions):
        """Evaluate the individual fitness of a scenario or a sequence of perturbations.

        :param individual: The individual to be evaluated.
        :param complete_solutions: The list of complete solutions with fitness evaluated.
        :return: The individual with fitness evaluated.
        """
        from impl.problem import weights
        index = 0 if isinstance(individual, AbstractScenarioDefinition) else 1
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
                individual.fitness.values = self._penalize(sorted_involved[0], np.min(sorted_involved, axis=0)[1])
            else:
                individual.fitness.values = sorted_involved[0]
        else:
            del individual.fitness.values
        return individual

    def run_scenarios(self, scenarios, agent_name="v1", rerun=False, additional_confs=None):
        """
        Run a batch of scenarios either sequentially or in parallel.

        :param scenarios: List of :class:`ScenarioDefinition` objects to simulate.
        :param agent_name: Name of agent configuration to use.
        :param rerun: Whether to rerun already evaluated scenarios.
        :param additional_confs: Optional additional configurations per scenario.
        :return: Tuple (List of results, number of successful runs).
        """
        tag = os.environ.get("tag", "default")
        if additional_confs is None:
            additional_confs = list(itertools.repeat(None, len(scenarios)))
        assert len(additional_confs) == len(scenarios)
        if cfg.CONFIG["simulation"]["parallel"]:
            process_configs = Manager().Queue()
            for instance in cfg.CONFIG["simulation"]["instances"]:
                process_configs.put(
                    (instance["host"], instance["port"], instance["tm_port"], instance["gpu_device"], tag))
            # FIXME: Traffic manager may cause memory leak.
            # https://github.com/carla-simulator/carla/issues/3584
            # https://github.com/carla-simulator/carla/issues/3540
            # https://github.com/carla-simulator/leaderboard/issues/81
            # https://github.com/carla-simulator/carla/issues/2781
            with ProcessPoolExecutor(max_workers=len(cfg.CONFIG["simulation"]["instances"]),
                                     initializer=_init_carla, initargs=(process_configs,)) as executor:
                results = executor.map(ADSEvaluator.run_scenario, scenarios,
                                       itertools.repeat(agent_name, len(scenarios)),
                                       itertools.repeat(rerun, len(scenarios)), additional_confs)
        else:
            global carla_host, carla_port, tm_port, gpu_device
            instance = cfg.CONFIG["simulation"]["instances"][0]
            carla_host, carla_port, tm_port, gpu_device = (instance["host"], instance["port"],
                                                           instance["tm_port"], instance["gpu_device"])
            results = map(ADSEvaluator.run_scenario, scenarios, itertools.repeat(agent_name, len(scenarios)),
                          itertools.repeat(rerun, len(scenarios)), additional_confs)

        results, is_executed = zip(*results)
        return results, is_executed.count(True)

    def fitness(self, source, follow_up):
        """Calculate the fitness value and check if it violates the metamorphic relations.

        :param source: Simulation results of the source scenario.
        :param follow_up: Simulation results of the follow-up scenario.
        :return: A tuple containing a bool value indicating whether it violates the relation and the fitness value.
        """
        if self.mr_set.relation.field == "velocity":
            func = lambda row: sqrt(row.velocity_x ** 2 + row.velocity_y ** 2)
            source[self.mr_set.relation.field] = source.apply(func, axis=1, result_type="reduce")
            follow_up[self.mr_set.relation.field] = follow_up.apply(func, axis=1, result_type="reduce")
        is_violated, extent = self.mr_set.is_violated(source, follow_up)
        if cfg.CONFIG["violation"]["clip_negative"] and extent is not None:
            extent = extent if extent > 0 else 0.0
        return is_violated, (extent,) if extent is not None else None

    @staticmethod
    def run_scenario(scenario: ScenarioDefinition, agent_name, rerun=False, additional_config=None):
        """Run a single :class:`ScenarioDefinition` through CARLA simulation.

        :param scenario: :class:`ScenarioDefinition` object to simulate.
        :param agent_name: Name of the agent configuration to use.
        :param rerun: Whether to force rerunning even if results exist.
        :param additional_config: Additional parameters to pass.
        :return: Tuple (Result :class:`Dataframe`, whether simulation was newly executed).
        """
        global carla_host, carla_port, tm_port, gpu_device
        assert carla_host is not None and carla_port is not None and tm_port is not None and gpu_device is not None

        agent_config = [conf["agent_config"] for conf in cfg.CONFIG["interfuser"]["versions"] if
                        conf["name"] == agent_name]
        if len(agent_config) == 0:
            raise ValueError(f"Agent not defined: \"{agent_name}\".")
        agent_config = agent_config[0]
        setattr(config, "agent_config", agent_config)

        if not rerun and agent_name in evaluated_scenarios:
            for evaluated_scenario, evaluated_result in evaluated_scenarios[agent_name]:
                if scenario == evaluated_scenario:
                    logger.debug(f"Scenario evaluated for agent {agent_name}: {evaluated_scenario}.")
                    return evaluated_result, False

        setattr(config, "host", carla_host)
        setattr(config, "port", carla_port)
        setattr(config, "trafficManagerPort", tm_port)
        setattr(config, "gpu_device", gpu_device)
        setattr(config, "additional_config", additional_config)
        logger.debug(f"Starting simulation, scenario id: {scenario.id_}, carla instance: {config.host}:{config.port}, "
                     f"traffic manager port: {config.trafficManagerPort} on cuda device {config.gpu_device}.")
        logger.debug(scenario)

        is_successful = False
        for _ in range(1 + cfg.CONFIG["simulation"]["retry_times"]):
            evaluator = None
            try:
                evaluator = ScenarioEvaluator(scenario, config)
                evaluator.run(config)
                is_successful = True
                break
            except InvalidScenarioDefinitionError as e:
                # logger.error(f"Scenario failed: {scenario}, message: {e}.")
                logger.error(f"Scenario failed, message: {e}.")
                is_successful = False
                break
            except Exception as e:
                # logger.error(f"Scenario failed: {scenario}, message: {e}.")
                logger.error(f"Scenario failed, message: {e}.")
                if cfg.CONFIG['debug']:
                    traceback.print_exc()
                is_successful = False
            finally:
                del evaluator
        if not is_successful:
            return None, False

        result_path = cfg.CONFIG["workspace"]["sim_result"] / f"{scenario.id_}.csv"
        if not result_path.exists():
            logger.warning(f"Scenario results cannot be found: {scenario.id_}.")
            return None, False

        try:
            result = pd.read_csv(result_path)
            result.set_index(result.columns[0], inplace=True)
        except Exception as e:
            logger.error(f"Scenario results cannot be read: {scenario.id_}, message: {e}.")
            return None, False
        if agent_name not in evaluated_scenarios:
            evaluated_scenarios[agent_name] = []
        evaluated_scenarios[agent_name].append((scenario, result))
        return result, True

    @staticmethod
    def _calculate_similarity(scenario, scenarios):
        """Calculate the heterogeneous distance between the given scenario and the list of scenarios.

        :param scenario: The given scenario.
        :param scenarios: A list of scenarios.
        :return: The heterogeneous distance.
        """
        return min([scenario.dist(scen) for scen in scenarios])

    def _evaluate_solutions(self, solutions, scenarios):
        """Evaluate the complete solutions.

        :param solutions: List of complete solutions.
        :return: Tuple (evaluated_solutions, number_of_simulations).
        """

        reeval = []
        results, sim_num = self.run_scenarios(scenarios)
        for solution, source, follow_up in zip(solutions, results[::2], results[1::2]):
            if source is not None and follow_up is not None:
                solution.source = source
                solution.follow_up = follow_up
                solution.is_violated, extent = self.fitness(source, follow_up)
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

        reeval_sim_num = self._reevaluate(reeval)
        return solutions, sim_num + reeval_sim_num

    def _evaluate_solutions_dt(self, solutions, scenarios):
        """Evaluate complete solutions using the Differential Testing approach.

        :param solutions: List of complete solutions.
        :return: Tuple (evaluated_solutions, number_of_simulations).
        """
        reference_version = cfg.CONFIG["search"]["diff_testing"]["reference"]
        test_version = cfg.CONFIG["search"]["diff_testing"]["test"]

        rv_solutions, rv_sim_num = self._perform_evaluation(solutions, scenarios, agent_name=reference_version)
        tv_solutions, tv_sim_num = self._perform_evaluation(rv_solutions, scenarios, agent_name=test_version)
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

    def _perform_evaluation(self, solutions, scenarios, agent_name):
        """Run simulations for scenarios and attach evaluation results to solutions.

        :param solutions: List of solutions to update.
        :param scenarios: Scenarios to simulate.
        :param agent_name: Name of the agent to run simulations with.
        :return: Tuple (updated_solutions, number_of_simulations).
        """
        reeval = []
        results, sim_num = self.run_scenarios(scenarios, agent_name=agent_name)
        for solution, source, follow_up in zip(solutions, results[::2], results[1::2]):
            eval_data = SimpleNamespace()
            if source is not None and follow_up is not None:
                eval_data.source = source
                eval_data.follow_up = follow_up
                eval_data.is_violated, extent = self.fitness(source, follow_up)
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

        reeval_sim_num = self._reevaluate_dt(reeval, agent_name)

        return solutions, sim_num + reeval_sim_num

    def _reevaluate(self, solutions):
        """Reevaluate selected solutions multiple times and aggregate the results.

        :param solutions: List of solutions flagged for reevaluation.
        :return: Total number of simulations performed during reevaluation.
        """
        if len(solutions) == 0:
            return 0
        repeat = cfg.CONFIG["violation"]["reevaluation"]["repeat"]
        aggregation = cfg.CONFIG["violation"]["reevaluation"]["aggregation"]
        scenarios = []
        for solution in solutions:
            solution.eval_history = [{
                "source": solution.source.copy(),
                "follow_up": solution.follow_up.copy(),
                "is_violated": solution.is_violated,
                "fitness": deepcopy(solution.fitness),
            }]
            solution.aggregation = aggregation
            for repetition in range(1, repeat):
                source = deepcopy(solution[0])
                source.assign_new_id()
                scenarios.append(source)
                follow_up = deepcopy(solution[0])
                follow_up.assign_new_id()
                solution[1].perturb(follow_up)
                scenarios.append(follow_up)
        assert len(scenarios) == len(solutions) * 2 * (repeat - 1)
        results, sim_num = self.run_scenarios(scenarios, rerun=True)

        for i, source, follow_up in zip(range(len(solutions) * (repeat - 1)), results[::2], results[1::2]):
            solution = solutions[int(i / (repeat - 1))]
            fitness = deepcopy(solution.fitness)
            if source is not None and follow_up is not None:
                is_violated, extent = self.fitness(source, follow_up)
                if extent:
                    fitness.values = extent
                else:
                    del fitness.values
            else:
                is_violated = False
                del fitness.values

            solution.eval_history.append({
                "source": source,
                "follow_up": follow_up,
                "is_violated": is_violated,
                "fitness": fitness,
            })

        for solution in solutions:
            fitnesses = [(ev["fitness"].values[0] if ev["fitness"].valid else np.nan) for ev in solution.eval_history]
            num_nan_fitnesses = len([f for f in fitnesses if np.isnan(f)])
            if num_nan_fitnesses > float(repeat) / 2:
                del solution.fitness.values
                solution.source = None
                solution.follow_up = None
                solution.is_violated = False
            else:
                aggregate_value = getattr(np, f"nan{aggregation}")(fitnesses)
                aggregation_arg = np.nanargmin(np.abs([f - aggregate_value for f in fitnesses]))
                selected_candidate = solution.eval_history[aggregation_arg]

                solution.fitness.values = (aggregate_value,)
                solution.source = selected_candidate["source"]
                solution.follow_up = selected_candidate["follow_up"]
                solution.is_violated = selected_candidate["is_violated"]

        return sim_num

    def _reevaluate_dt(self, solutions, agent_name):
        """Reevaluate selected solutions multiple times and aggregate the results.

        :param solutions: List of solutions flagged for reevaluation.
        :param agent_name: Name of the agent to run simulations with.
        :return: Total number of simulations performed during reevaluation.
        """
        if len(solutions) == 0:
            return 0
        repeat = cfg.CONFIG["violation"]["reevaluation"]["repeat"]
        aggregation = cfg.CONFIG["violation"]["reevaluation"]["aggregation"]
        scenarios = []
        for solution in solutions:
            eval_data = getattr(solution, agent_name)
            eval_data.eval_history = [{
                "source": eval_data.source.copy(),
                "follow_up": eval_data.follow_up.copy(),
                "is_violated": eval_data.is_violated,
                "fitness": deepcopy(eval_data.fitness),
            }]
            eval_data.aggregation = aggregation
            for repetition in range(1, repeat):
                source = deepcopy(solution[0])
                source.assign_new_id()
                scenarios.append(source)
                follow_up = deepcopy(solution[0])
                follow_up.assign_new_id()
                solution[1].perturb(follow_up)
                scenarios.append(follow_up)
        assert len(scenarios) == len(solutions) * 2 * (repeat - 1)
        results, sim_num = self.run_scenarios(scenarios, agent_name=agent_name, rerun=True)

        for i, source, follow_up in zip(range(len(solutions) * (repeat - 1)), results[::2], results[1::2]):
            solution = solutions[int(i / (repeat - 1))]
            eval_data = getattr(solution, agent_name)
            fitness = deepcopy(eval_data.fitness)
            if source is not None and follow_up is not None:
                is_violated, extent = self.fitness(source, follow_up)
                if extent:
                    fitness.values = extent
                else:
                    del fitness.values
            else:
                is_violated = False
                del fitness.values

            eval_data.eval_history.append({
                "source": source,
                "follow_up": follow_up,
                "is_violated": is_violated,
                "fitness": fitness,
            })

        for solution in solutions:
            eval_data = getattr(solution, agent_name)
            fitnesses = [(ev["fitness"].values[0] if ev["fitness"].valid else np.nan) for ev in eval_data.eval_history]
            num_nan_fitnesses = len([f for f in fitnesses if np.isnan(f)])
            if num_nan_fitnesses > float(repeat) / 2:
                eval_data.fitness = None
                eval_data.source = None
                eval_data.follow_up = None
                eval_data.is_violated = False
            else:
                aggregate_value = getattr(np, f"nan{aggregation}")(fitnesses)
                aggregation_arg = np.nanargmin(np.abs([f - aggregate_value for f in fitnesses]))
                selected_candidate = eval_data.eval_history[aggregation_arg]

                eval_data.fitness = (aggregate_value,)
                eval_data.source = selected_candidate["source"]
                eval_data.follow_up = selected_candidate["follow_up"]
                eval_data.is_violated = selected_candidate["is_violated"]
        return sim_num

    def _penalize(self, fitness, similarity):
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
