import argparse
import logging.config
import os
import pickle
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import argformat
import torch

from impl import config as cfg
from impl.algorithm.ccea import CCEA
from impl.algorithm.ga import GeneticAlgorithm
from impl.algorithm.moccea import MOCCEA
from impl.algorithm.rs import RandomSearch
from impl.scenario.scenario_definition import ScenarioDefinition
from impl.scenario.simulation_runner import run_scenarios, run_solutions, run_free_environments
from impl.utils.docker_utils import cleanup_containers
from impl.utils.leaderboad_utils import get_enviroment_confs, make_yamls, create_dataset_index, vectorize_runtime_data

logger = logging.getLogger("impl")


def parse_list(_type, delimeter):
    def parse_func(arg_str):
        """Parse a comma-separated list of integers into a list."""
        try:
            arg_str = arg_str.strip()
            if arg_str.startswith('[') and arg_str.endswith(']'):
                arg_str = arg_str[1:-1]
            return [_type(str(w).strip()) for w in arg_str.split(delimeter)]
        except ValueError:
            raise argparse.ArgumentTypeError(f"args must be a list of '{_type}' separated by '{delimeter}'")

    return parse_func


def search(algorithm: str, resume: bool, folder_name: str):
    from impl import problem
    if resume:
        cfg.init_project_directory(folder_name, resume)
    else:
        cfg.init_project_directory(algorithm, resume)
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
    cfg.init_project_directory("sim")
    if file:
        logger.info(f"Loading solution file: {file}.")
        run_solutions(file, num)
    else:
        logger.info(f"Running random scenarios.")
        run_scenarios([ScenarioDefinition.generate_random_or_leaderboard() for _ in range(num)])


def collect_runtime_data(agent: str, output: str):
    if output is None:
        output = cfg.CONFIG["workspace"]["runtime_data"]
        cp_root = cfg.CONFIG["workspace"]["data_collection_checkpoint"]
    else:
        cp_root = os.path.join(output, 'checkpoints')
    logger.info(f"Running free simulation environment for agent {agent} to collect runtime data...")

    os.makedirs(cfg.CONFIG["workspace"]["train_data"], exist_ok=True)
    os.makedirs(output, exist_ok=True)

    agent_conf = cfg.CONFIG["interfuser"].copy()
    del agent_conf["versions"]
    version_conf = [_c for _c in cfg.CONFIG["interfuser"]["versions"] if _c["name"] == agent][0]

    workspace_conf = {
        "cp_root": cp_root,
        "output_root": output,
    }

    exec_conf = {
        "collection_delay_lower": str(cfg.CONFIG["runtime"]["collection_delay"][0]),
        "collection_delay_upper": str(cfg.CONFIG["runtime"]["collection_delay"][1]),
        "collection_duration": str(cfg.CONFIG["runtime"]["collection_duration"]),
        "collection_interval": str(cfg.CONFIG["runtime"]["collection_interval"]),
        "collection_route_limit": str(cfg.CONFIG["runtime"]["collection_route_limit"]),
    }

    environment_confs = get_enviroment_confs()
    for i in range(len(environment_confs)):
        environment_confs[i] = {
            **environment_confs[i],
            **agent_conf,
            **version_conf,
            **workspace_conf,
            **exec_conf,
        }

    run_free_environments(environment_confs)


def generate_train_data():
    logger.info(f"Generating training data...")

    os.makedirs(cfg.CONFIG["workspace"]["data_gen_checkpoint"], exist_ok=True)
    os.makedirs(cfg.CONFIG["workspace"]["train_data"], exist_ok=True)

    workspace_conf = {
        "cp_root": cfg.CONFIG["workspace"]["data_gen_checkpoint"],
        "output_root": cfg.CONFIG["workspace"]["train_data"]
    }
    environment_confs = get_enviroment_confs()
    make_yamls()
    for i in range(len(environment_confs)):
        weather = environment_confs[i]["weather"]
        environment_confs[i] = {
            **environment_confs[i],
            **workspace_conf,
            **{
                "agent_path": os.path.join(cfg.CONFIG["interfuser"]["repo"],
                                           "leaderboard", "team_code", "auto_pilot.py"),
                "agent_config": os.path.join(cfg.CONFIG["data_collection"]["yaml_root"], f"weather-{weather}.yaml"),
            }
        }
    run_free_environments(environment_confs)


def train_interfuser(args):
    logger.info(f"Creating dataset index...")
    create_dataset_index(cfg.CONFIG["workspace"]["train_data"],
                         weathers=args.train_weathers + args.val_weathers,
                         towns=args.train_towns + args.val_towns,
                         )
    logger.info(f"Dataset index created at {cfg.CONFIG['workspace']['train_data'] / 'dataset_index.txt'}")

    gpu_count = torch.cuda.device_count()
    if args.gpu_num > gpu_count:
        raise RuntimeError(f"Requested {args.gpu_num} GPUs, but only {gpu_count} are available.")
    output_base = args.output

    logger.info(f"Training an Interfuser model on {args.gpu_num} GPUs...")
    child_env = os.environ.copy()
    child_env.update({
        "GPU_NUM": str(args.gpu_num),  # TODO test
        "DATASET_ROOT": str(cfg.CONFIG["workspace"]["train_data"]),
    })
    child_env["PYTHONPATH"] = os.pathsep.join(sys.path)

    distributed_command = f"-m torch.distributed.launch --nproc_per_node={args.gpu_num}" if args.gpu_num > 1 else ""
    command = (
        f"{sys.executable} {distributed_command}"
        f" {os.path.join(cfg.CONFIG['interfuser']['repo'], 'interfuser', 'train.py')}"
        f" {cfg.CONFIG['workspace']['train_data']}"
        f" --dataset carla"
        f" --train-towns {' '.join([str(c) for c in args.train_towns])}"
        f" --val-towns {' '.join([str(c) for c in args.val_towns])}"
        f" --train-weathers {' '.join([str(c) for c in args.train_weathers])}"
        f" --val-weathers {' '.join([str(c) for c in args.val_weathers])}"
        f" --model {args.model}"
        f" --sched cosine"
        f" --epochs {args.epochs}"
        f" --warmup-epochs {args.warmup_epochs}"
        f" --lr {args.lr}"
        f" --batch-size {args.batch_size}"
        f" --workers {args.workers}"
        f" --no-prefetcher"
        f" --eval-metric {args.eval_metric}"
        f" --opt {args.opt}"
        f" --opt-eps {args.opt_eps}"
        f" --weight-decay {args.weight_decay}"
        f" --scale 0.9 1.1"
        f" --saver-decreasing"
        f" --clip-grad 10"
        f" --freeze-num -1"
        f" --with-backbone-lr"
        f" --backbone-lr {args.backbone_lr}"
        f" --multi-view"
        f" --with-lidar"
        f" --multi-view-input-size 3 128 128"
        f" --experiment interfuser_baseline"
        f" --pretrained"
        f" --output {args.output}"
    )
    if args.resume:
        runs = [d for d in os.listdir(output_base) if os.path.isdir(os.path.join(output_base, d))]
        if len(runs) == 0: raise RuntimeError("'--resume' was given but cannot find the last run.")

        last_cp_dir = sorted(runs, key=lambda d: os.path.getctime(os.path.join(output_base, d)), reverse=True)[
            0]
        last_cp = os.path.join(output_base, last_cp_dir, "last.pth.tar")
        command = command + f" --resume {last_cp}"

    if hasattr(args, "retrain") and args.retrain is not None:
        command = command + f" --resume {args.retrain}"

    process = subprocess.run(command, env=child_env, check=True, shell=True, text=True, stdout=None, stderr=None)

    return process.returncode


def convert2scenarios(directory: str, n: int, towns: List[str]):
    def vectorize(path):
        towns_lower = [name.lower() for name in towns]
        try:
            runtime_data = pickle.loads(path.read_bytes())
            if towns is not None and runtime_data["sim_data"]["town"].lower() not in towns_lower:
                logger.info("Scenario town is not in one of the selected towns. skipping...")
                return None
            return vectorize_runtime_data(runtime_data)
        except Exception as e:
            logger.error(f"Failed to vectorize runtime data from {path}, error message {e}.")
            return None

    scenarios = []
    if n < 1:
        for data_path in Path(directory).rglob("*.pkl"):
            scenario = vectorize(data_path)
            if scenario is None: continue
            scenarios.append(scenario)
    else:
        count = 0
        files = list(Path(directory).rglob("*.pkl"))
        random.shuffle(files)
        for data_path in files:
            if count < n:
                scenario = vectorize(data_path)
                if scenario is None: continue
                scenarios.append(scenario)
            else:
                i = random.randint(0, count)
                if i < n:
                    scenario = vectorize(data_path)
                    if scenario is None: continue
                    scenarios[i] = scenario
            count += 1

    (cfg.CONFIG["workspace"]["runtime_scenario"] /
     f"rt_scen_{str(int(round(time.time() * 1000)))}.pickle").write_bytes(pickle.dumps(scenarios))


if __name__ == "__main__":
    # Parse the command line.
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
    parser_search.add_argument("-f", "--folder", default=None, help="folder name to resume from")
    parser_search.set_defaults(func=lambda args: search(args.algorithm, args.resume, args.folder))

    parser_sim = subparsers.add_parser("simulate", aliases=["sim"], help="run simulations")
    parser_sim.add_argument("-n", "--number", type=int, default=1, help="number of simulations to run")
    parser_sim.add_argument("-f", "--file", default=None, help="solution file")
    parser_sim.set_defaults(func=lambda args: simulate(args.number, args.file))

    parser_crd = subparsers.add_parser("collect_runtime_data",
                                       help="Execute free environments to collect runtime data.")
    parser_crd.add_argument("-a", "--agent", type=str, default="v1", help="Agent version or name")
    parser_crd.add_argument("--output", type=str, default=None, help="Path of output directory")
    parser_crd.set_defaults(func=lambda args: collect_runtime_data(args.agent, args.output))

    parser_gtd = subparsers.add_parser("generate_train_data", help="Generate training data using a rule-based agent")
    parser_gtd.set_defaults(func=lambda args: generate_train_data())

    parser_train = subparsers.add_parser("train", help="Train an Interfuser agent using generated data.")

    parser_train.add_argument("--gpu-num", type=int,
                              default=cfg.CONFIG["training"]["gpu_num"],
                              help="Number of GPUS for training.")
    parser_train.add_argument("--train-weathers", type=parse_list(int, ","),
                              default=cfg.CONFIG["training"]["train_weathers"],
                              help="List of weathers for training, e.g. '0,1,2,3'")
    parser_train.add_argument("--train-towns", type=parse_list(int, ","),
                              default=cfg.CONFIG["training"]["train_towns"],
                              help="List of towns for training, e.g. '1,2,3'")
    parser_train.add_argument("--val-weathers", type=parse_list(int, ","),
                              default=cfg.CONFIG["training"]["val_weathers"],
                              help="List of weathers for validation, e.g. '0,1,2,3'")
    parser_train.add_argument("--val-towns", type=parse_list(int, ","),
                              default=cfg.CONFIG["training"]["val_towns"],
                              help="List of towns for validation, e.g. '1,2,3'")
    parser_train.add_argument("--model", type=str,
                              default="interfuser_baseline",
                              help="Model to train, default: 'interfuser_baseline'")
    parser_train.add_argument("--epochs", type=int,
                              default=cfg.CONFIG["training"]["epochs"],
                              help="Number of training epochs.")
    parser_train.add_argument("--warmup-epochs", type=int,
                              default=cfg.CONFIG["training"]["warmup_epochs"],
                              help="Number of warmup epochs.")
    parser_train.add_argument("--lr", type=float,
                              default=cfg.CONFIG["training"]["lr"],
                              help="Learning rate.")
    parser_train.add_argument("--batch-size", type=int,
                              default=cfg.CONFIG["training"]["batch_size"],
                              help="Size of batches.")
    parser_train.add_argument("--eval-metric", type=str,
                              default=cfg.CONFIG["training"]["eval_metric"],
                              help="Evaluation metric.")
    parser_train.add_argument("--opt", type=str,
                              default=cfg.CONFIG["training"]["opt"],
                              help="Optimization algorithm.")
    parser_train.add_argument("--opt-eps", type=float,
                              default=cfg.CONFIG["training"]["opt_eps"],
                              help="Optimization tolerance.")
    parser_train.add_argument("--weight-decay", type=float,
                              default=cfg.CONFIG["training"]["weight_decay"],
                              help="Weight decay regularization parameter.")
    parser_train.add_argument("--backbone-lr", type=float,
                              default=cfg.CONFIG["training"]["backbone_lr"],
                              help="Learning rate of backbone models.")
    parser_train.add_argument("--output", type=str,
                              default=cfg.CONFIG["workspace"].get("trained_models"),
                              help="Path to training output and results.")
    parser_train.add_argument("--workers", type=int,
                              default=cfg.CONFIG["training"]["workers"],
                              help="How many training processes to use.")
    group = parser_train.add_mutually_exclusive_group()
    group.add_argument("--resume", action="store_true",
                       help="Resume the last checkpoint.")
    group.add_argument("--retrain", type=str,
                       help="Fine-tune a specific model give the model's path")

    parser_train.set_defaults(func=lambda args: train_interfuser(args))

    parser_convert = subparsers.add_parser("convert", aliases=["conv"],
                                           help="convert runtime data to runtime scenarios")
    parser_convert.add_argument("-d", "--directory", required=True, help="directory of runtime data")
    parser_convert.add_argument("-n", "--number", type=int, default=100, help="number of runtime scenarios")
    parser_convert.add_argument("--towns", type=parse_list(str, ","), default=None, help="filter specific towns")
    parser_convert.set_defaults(func=lambda args: convert2scenarios(args.directory, args.number, args.towns))

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    arguments = parser.parse_args()
    arguments.func(arguments)
    cleanup_containers()
