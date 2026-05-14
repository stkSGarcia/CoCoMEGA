#!/usr/bin/env python3
"""Summarize how re-evaluation changes reported divergence alarms.

The script loads the experimental run lists from ``test/postprocess-new.ipynb``
and the serialized evaluated solutions from the corresponding result
directories. It never runs CARLA, calls evaluator APIs, or recomputes MR
fitness; it only reads values already stored in ``fitness.values`` and
``eval_history``. It reports two related quantities:

1. DT-level alarm stability: solutions whose initial v1-v2 divergence is above
   the re-evaluation threshold, and whether the final median-aggregated
   divergence remains above that threshold.
2. Version-level MR stability: individual v1/v2 MR violation measurements that
   triggered re-evaluation, and whether their median-aggregated value remains
   above the threshold.
"""

from __future__ import annotations

import argparse
import ast
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK = REPO_ROOT / "test" / "postprocess-new.ipynb"
DEFAULT_RESULTS_ROOT = Path("/data/hyous028/cocomega/results")


def install_pickle_compatibility_aliases() -> None:
    """Register module aliases required by older serialized solutions."""
    import impl.ads.scenario.scenario_definition as scen_def_module
    import impl.ads.mr.mr as base_mr_module
    import impl.core.algorithm as alg_module

    sys.modules["impl.scenario.scenario_definition"] = scen_def_module
    sys.modules["impl.mr.mr"] = base_mr_module
    sys.modules["impl.algorithm"] = alg_module


def load_project_lists(notebook_path: Path) -> Dict[str, Dict[str, List[str]]]:
    """Load ``mr1_data`` and ``mr2_data`` dictionaries from the notebook."""
    notebook = json.loads(notebook_path.read_text())
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if "mr1_data" not in source or "mr2_data" not in source:
            continue

        tree = ast.parse(source)
        projects: Dict[str, Dict[str, List[str]]] = {}
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"mr1_data", "mr2_data"}:
                    projects[target.id] = ast.literal_eval(node.value)
        return projects

    raise RuntimeError(f"Could not find mr1_data and mr2_data in {notebook_path}")


def find_pickle(run_dir: Path, prefix: str) -> Optional[Path]:
    files = sorted((run_dir / "solutions").glob(f"{prefix}*.pickle"))
    return files[-1] if files else None


def stored_fitness_value(fitness: Any) -> Optional[float]:
    """Read a previously stored scalar fitness value without recomputing it."""
    values = getattr(fitness, "values", None)
    if not values:
        return None
    return float(values[0])


def stored_eval_fitness_value(eval_data: Any) -> Optional[float]:
    """Read the final stored fitness value from a v1/v2 evaluation record."""
    if eval_data is None:
        return None
    return stored_fitness_value(getattr(eval_data, "fitness", None))


def stored_initial_eval_fitness_value(eval_data: Any) -> Optional[float]:
    """Read the first stored fitness value before median aggregation."""
    history = getattr(eval_data, "eval_history", None)
    if history:
        return stored_fitness_value(history[0].get("fitness"))
    return stored_eval_fitness_value(eval_data)


def safe_abs_diff(left: Optional[float], right: Optional[float]) -> Optional[float]:
    if left is None or right is None:
        return None
    return abs(left - right)


def get_run_threshold(run_dir: Path, fallback: float) -> float:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return fallback
    config = json.loads(config_path.read_text())
    return float(config["violation"]["reevaluation"]["threshold"])


def summarize_run(
    dataset: str,
    algorithm: str,
    run_name: str,
    run_dir: Path,
    default_threshold: float,
) -> Dict[str, Any]:
    solution_path = find_pickle(run_dir, "evaluated-")
    if solution_path is None:
        return {
            "dataset": dataset,
            "algorithm": algorithm,
            "run": run_name,
            "missing": True,
            "reason": "missing evaluated pickle",
        }

    threshold = get_run_threshold(run_dir, default_threshold)
    with solution_path.open("rb") as handle:
        solutions = pickle.load(handle)

    row: Dict[str, Any] = {
        "dataset": dataset,
        "algorithm": algorithm,
        "run": run_name,
        "missing": False,
        "threshold": threshold,
        "evaluated_solutions": len(solutions),
        "dt_initial_above": 0,
        "dt_final_above": 0,
        "dt_dropped": 0,
        "dt_initial_invalid": 0,
        "dt_final_invalid": 0,
        "version_initial_above": 0,
        "version_final_above": 0,
        "version_dropped": 0,
        "version_reevaluated": 0,
        "version_invalid_after": 0,
    }

    for solution in solutions:
        initial_vals = {
            agent: stored_initial_eval_fitness_value(getattr(solution, agent, None))
            for agent in ("v1", "v2")
        }
        final_vals = {
            agent: stored_eval_fitness_value(getattr(solution, agent, None))
            for agent in ("v1", "v2")
        }

        initial_diff = safe_abs_diff(initial_vals["v1"], initial_vals["v2"])
        final_diff = safe_abs_diff(final_vals["v1"], final_vals["v2"])

        if initial_diff is None:
            row["dt_initial_invalid"] += 1
        elif initial_diff >= threshold:
            row["dt_initial_above"] += 1
            if final_diff is not None and final_diff >= threshold:
                row["dt_final_above"] += 1
            else:
                row["dt_dropped"] += 1

        if final_diff is None:
            row["dt_final_invalid"] += 1

        for agent in ("v1", "v2"):
            eval_data = getattr(solution, agent, None)
            if not getattr(eval_data, "eval_history", None):
                continue
            row["version_reevaluated"] += 1

            initial_val = stored_initial_eval_fitness_value(eval_data)
            final_val = stored_eval_fitness_value(eval_data)
            if initial_val is not None and initial_val >= threshold:
                row["version_initial_above"] += 1
                if final_val is not None and final_val >= threshold:
                    row["version_final_above"] += 1
                else:
                    row["version_dropped"] += 1
            if final_val is None:
                row["version_invalid_after"] += 1

    return row


def rate(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def aggregate(rows: Iterable[Mapping[str, Any]], group_cols: List[str]) -> pd.DataFrame:
    df = pd.DataFrame([row for row in rows if not row.get("missing")])
    numeric_cols = [
        "evaluated_solutions",
        "dt_initial_above",
        "dt_final_above",
        "dt_dropped",
        "dt_initial_invalid",
        "dt_final_invalid",
        "version_initial_above",
        "version_final_above",
        "version_dropped",
        "version_reevaluated",
        "version_invalid_after",
    ]
    if group_cols:
        grouped = df.groupby(group_cols, as_index=False)[numeric_cols].sum()
    else:
        grouped = pd.DataFrame([{col: df[col].sum() for col in numeric_cols}])
    grouped["dt_instability_false_positive_rate"] = grouped.apply(
        lambda row: rate(row["dt_dropped"], row["dt_initial_above"]), axis=1
    )
    grouped["version_instability_false_positive_rate"] = grouped.apply(
        lambda row: rate(row["version_dropped"], row["version_initial_above"]), axis=1
    )
    return grouped


def write_outputs(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_df = pd.DataFrame(rows)
    run_df.to_csv(output_dir / "reevaluation_false_positive_by_run.csv", index=False)

    by_dataset_alg = aggregate(rows, ["dataset", "algorithm"])
    by_dataset_alg.to_csv(output_dir / "reevaluation_false_positive_by_dataset_algorithm.csv", index=False)

    by_dataset = aggregate(rows, ["dataset"])
    by_dataset.to_csv(output_dir / "reevaluation_false_positive_by_dataset.csv", index=False)

    overall = aggregate(rows, [])
    overall.to_csv(output_dir / "reevaluation_false_positive_overall.csv", index=False)

    print("\nOverall")
    print(overall.to_string(index=False))
    print("\nBy dataset")
    print(by_dataset.to_string(index=False))
    print("\nBy dataset and algorithm")
    print(by_dataset_alg.to_string(index=False))

    missing = run_df[run_df.get("missing", False) == True]  # noqa: E712
    if not missing.empty:
        print("\nMissing runs")
        print(missing[["dataset", "algorithm", "run", "reason"]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "analysis" / "reevaluation_false_positives")
    parser.add_argument("--default-threshold", type=float, default=0.2)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["mr1_data", "mr2_data"],
        choices=["mr1_data", "mr2_data"],
    )
    args = parser.parse_args()

    install_pickle_compatibility_aliases()
    projects = load_project_lists(args.notebook)

    rows: List[Dict[str, Any]] = []
    for dataset in args.datasets:
        for algorithm, runs in projects[dataset].items():
            for run_name in runs:
                rows.append(
                    summarize_run(
                        dataset=dataset,
                        algorithm=algorithm,
                        run_name=run_name,
                        run_dir=args.results_root / run_name,
                        default_threshold=args.default_threshold,
                    )
                )

    write_outputs(rows, args.output_dir)


if __name__ == "__main__":
    main()
