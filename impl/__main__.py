import argparse
import logging.config
import os
import sys

import argformat
import subprocess

from timm.data import create_dataset

from impl import problem
from impl.algorithm.ccea import CCEA
from impl.algorithm.ga import GeneticAlgorithm
from impl.algorithm.moccea import MOCCEA
from impl.algorithm.rs import RandomSearch
from impl.scenario.scenario_definition import ScenarioDefinition
from impl.scenario.simulation_runner import run_scenarios, run_solutions, run_free_environments
from impl.utils.docker_utils import cleanup_containers
from impl.utils.leaderboad_utils import get_enviroment_confs, make_yamls, create_dataset_index

from impl.config import CONFIG

logger = logging.getLogger("impl")


def search(algorithm: str, resume: bool):
    if algorithm == "ccea":
        solver = CCEA(toolbox=problem.toolbox, budget=problem.budget)
    elif algorithm == "moccea":
        solver = MOCCEA(toolbox=problem.toolbox, budget=problem.budget)
    elif algorithm == "rs":
        solver = RandomSearch(toolbox=problem.toolbox, budget=problem.budget)
    elif algorithm == "ga":
        solver = GeneticAlgorithm(toolbox=problem.toolbox, budget=problem.budget)
    elif algorithm == "gawa":
        solver = GeneticAlgorithm(toolbox=problem.toolbox, budget=problem.budget, keep_best=True)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}.")
    solver.solve(resume=resume)


def simulate(num: int, file: str):
    if file:
        logger.info(f"Loading solution file: {file}.")
        run_solutions(file, num)
    else:
        logger.info(f"Running random scenarios.")
        run_scenarios([ScenarioDefinition.generate_random_or_leaderboard() for _ in range(num)])


def collect_runtime_data(agent: str):
    logger.info(f"Running free simulation environment for agent {agent} to collect runtime data...")

    os.makedirs(CONFIG["workspace"]["train_data"], exist_ok=True)
    os.makedirs(CONFIG["workspace"]["runtime_data"], exist_ok=True)

    workspace_conf = {
        "cp_root": CONFIG["workspace"]["data_collection_checkpoint"],
        "output_root": CONFIG["workspace"]["runtime_data"]
    }
    agent_conf = CONFIG["interfuser"].copy()
    del agent_conf["versions"]
    version_conf = [_c for _c in CONFIG["interfuser"]["versions"] if _c["name"] == agent][0]
    environment_confs = get_enviroment_confs()
    for i in range(len(environment_confs)):
        environment_confs[i] = {
            **environment_confs[i],
            **agent_conf,
            **version_conf,
            **workspace_conf,
        }

    run_free_environments(environment_confs)


def generate_train_data():
    logger.info(f"Generating training data...")

    os.makedirs(CONFIG["workspace"]["data_gen_checkpoint"], exist_ok=True)
    os.makedirs(CONFIG["workspace"]["train_data"], exist_ok=True)

    workspace_conf = {
        "cp_root": CONFIG["workspace"]["data_gen_checkpoint"],
        "output_root": CONFIG["workspace"]["train_data"]
    }
    environment_confs = get_enviroment_confs()
    make_yamls()
    for i in range(len(environment_confs)):
        weather = environment_confs[i]["weather"]
        environment_confs[i] = {
            **environment_confs[i],
            **workspace_conf,
            **{
                "agent_path": os.path.join(CONFIG["interfuser"]["repo"], "leaderboard", "team_code", "auto_pilot.py"),
                "agent_config": os.path.join(CONFIG["data_collection"]["yaml_root"], f"weather-{weather}.yaml"),
            }
        }
    run_free_environments(environment_confs)


def train_interfuser():
    logger.info(f"Creating dataset index...")
    create_dataset_index(CONFIG["workspace"]["train_data"])
    logger.info(f"Dataset index created at {os.path.join(CONFIG['workspace']['train_data'], 'dataset_index.txt')}")
    logger.info("Training an Interfuser model...")
    child_env = os.environ.copy()
    child_env.update({
        "GPU_NUM": str(CONFIG["training"]["gpu_num"]),
        "DATASET_ROOT": CONFIG["workspace"]["train_data"],
    })
    child_env["PYTHONPATH"] = os.pathsep.join(sys.path)

    command = (
        f"{sys.executable} -m torch.distributed.launch --nproc_per_node={str(CONFIG['training']['gpu_num'])}"
        f" {os.path.join(CONFIG['interfuser']['repo'], 'interfuser', 'train.py')}"
        f" {CONFIG['workspace']['train_data']}"
        f" --dataset carla"
        f" --train-towns {' '.join([str(c) for c in CONFIG['training']['train_towns']])}"
        f" --val-towns {' '.join([str(c) for c in CONFIG['training']['val_towns']])}"
        f" --train-weathers {' '.join([str(c) for c in CONFIG['training']['train_weathers']])}"
        f" --val-weathers {' '.join([str(c) for c in CONFIG['training']['val_weathers']])}"
        f" --model interfuser_baseline"
        f" --sched cosine"
        f" --epochs {CONFIG['training']['epochs']}"
        f" --warmup-epochs {CONFIG['training']['warmup_epochs']}"
        f" --lr {CONFIG['training']['lr']}"
        f" --batch-size {CONFIG['training']['batch_size']}"
        f" -j {CONFIG['training']['j']}"
        f" --no-prefetcher"
        f" --eval-metric {CONFIG['training']['eval_metric']}"
        f" --opt {CONFIG['training']['opt']}"
        f" --opt-eps {CONFIG['training']['opt_eps']}"
        f" --weight-decay {CONFIG['training']['weight_decay']}"
        f" --scale {CONFIG['training']['scale']}"
        f" --saver-decreasing"
        f" --clip-grad {CONFIG['training']['clip_grad']}"
        f" --freeze-num {CONFIG['training']['freeze_num']}"
        f" --with-backbone-lr"
        f" --backbone-lr {CONFIG['training']['backbone_lr']}"
        f" --multi-view"
        f" --with-lidar"
        f" --multi-view-input-size 3 128 128"
        f" --experiment interfuser_baseline"
        f" --pretrained"
    )

    process = subprocess.run(command, env=child_env, check=True, shell=True, text=True, stdout=None, stderr=None)

    return process.returncode


class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        is_kv = ["=" in value for value in values]
        if all(is_kv):
            pairs = {}
            for value in values:
                k, v = value.split("=")
                pairs[k] = v.split(",")
            setattr(namespace, self.dest, pairs)
        elif not any(is_kv):
            setattr(namespace, self.dest, values)
        else:
            parser.error("expected consistent type of arguments")


if __name__ == "__main__":
    # Parse command line.
    parser = argparse.ArgumentParser(
        prog="mtcg",
        description="Test case generator for metamorphic testing.",
        formatter_class=argformat.StructuredFormatter
    )
    subparsers = parser.add_subparsers(
        title="subcommands",
        description="Valid subcommands",
        required=True,
        help="subcommand help"
    )

    parser_search = subparsers.add_parser("search", aliases=["srch"], help="start the search")
    parser_search.add_argument("-a", "--algorithm", choices=("ccea", "moccea", "rs", "ga", "gawa"), default="ccea",
                               help="choose the algorithm to use. "
                                    "ccea: cooperative co-evolutionary algorithm; "
                                    "moccea: multi-objective cooperative co-evolutionary algorithm; "
                                    "rs: random search algorithm; "
                                    "ga: standard genetic algorithm; "
                                    "gawa: genetic algorithm with archive strategy")
    parser_search.add_argument("-r", "--resume", action="store_true", help="resume the previous run")
    parser_search.set_defaults(func=lambda args: search(args.algorithm, args.resume))

    parser_sim = subparsers.add_parser("simulate", aliases=["sim"], help="run simulations")
    parser_sim.add_argument("-n", "--number", type=int, default=1, help="number of simulations to run")
    parser_sim.add_argument("-f", "--file", default=None, help="solution file")
    parser_sim.set_defaults(func=lambda args: simulate(args.number, args.file))

    parser_sim = subparsers.add_parser("collect_runtime_data",
                                       help="Execute free environments to collect runtime data.")
    parser_sim.add_argument("-a", "--agent", type=str, default="v1", help="Agent version or name")
    parser_sim.set_defaults(func=lambda args: collect_runtime_data(args.agent))

    parser_sim = subparsers.add_parser("generate_train_data", help="Generate training data using a rule-based agent")
    parser_sim.set_defaults(func=lambda args: generate_train_data())

    parser_sim = subparsers.add_parser("train", help="Train an Interfuser agent using generated data.")
    parser_sim.set_defaults(func=lambda args: train_interfuser())

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    arguments = parser.parse_args()
    arguments.func(arguments)
    cleanup_containers()
