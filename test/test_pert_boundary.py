import logging
import os
import pickle
import sys
import time

import pandas as pd

from impl.algorithm import CCEA
import impl.config as config
from impl.mr.mr import Decreasing, MR, PerturbationFactory, MRSet
from impl import problem
from impl.scenario.scenario_definition import ScenarioDefinition, Boundary
from impl.scenario.simulation_runner import run_scenarios
from impl.utils.visualization import Visualizer

logger = logging.getLogger("impl")


class TestPertBoundary:
    def __init__(self, category, algorithm="ccea", conf=None):
        self.category = category
        self.algorithm = algorithm
        config.CONFIG = config.merge_dict(config.CONFIG, conf)
        self.base_dir = config.CONFIG["workspace"]["test_result"]
        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)
        self.out_dir = os.path.join(self.base_dir,
                                    f'pert_boundary_{int(round(time.time() * 1000))}')
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)

    def run(self):
        meta = pd.DataFrame()
        pert_boundaries = {
            "small":
                Boundary({
                    "x": [-180.0, -178.0],
                    "y": [89.0, 102.0],
                    "z": [0.5, 1.0],
                    "pitch": [0.0, 0.0],
                    "yaw": [-180.0, 180.0],
                    "roll": [0.0, 0.0],
                    "speed": [0.0, 10.0],
                    "model": [0, 14],
                }),
            # "medium":
            #     Boundary({
            #         "x": [-190.0, -178.0],
            #         "y": [89.0, 102.0],
            #         "z": [0.5, 1.0],
            #         "pitch": [0.0, 0.0],
            #         "yaw": [-180.0, 180.0],
            #         "roll": [0.0, 0.0],
            #         "speed": [0.0, 10.0],
            #         "model": [0, 14],
            #     }),
            # "large":
            #     Boundary({
            #         "x": [-198.0, -178.0],
            #         "y": [89.0, 102.0],
            #         "z": [0.5, 1.0],
            #         "pitch": [0.0, 0.0],
            #         "yaw": [-180.0, 180.0],
            #         "roll": [0.0, 0.0],
            #         "speed": [0.0, 10.0],
            #         "model": [0, 14],
            #     }),
        }
        for name, pert_boundary in pert_boundaries.items():
            pert_factory = PerturbationFactory(category=self.category, boundary=pert_boundary)
            problem.mr_set = MRSet([MR([pert_factory], Decreasing("velocity"))])
            statistics_path = self._test(problem)
            meta = pd.concat([meta, pd.DataFrame([{
                'name': name,
                'boundary': pert_boundary.boundary,
                'statistics_path': statistics_path
            }])], ignore_index=True)

        Visualizer.plot_pert_boundary_results(meta, out_dir=self.out_dir)

        return meta

    def _test(self, problem):
        if self.algorithm == "ccea":
            solver = CCEA(
                toolbox=problem.toolbox,
                budget=problem.budget,
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}.")
        _, statistics_path = solver.solve(resume=False)
        return statistics_path


if __name__ == '__main__':
    meta = TestPertBoundary(category="walker", conf={
        "max_iteration": 1,
        "scenario": {
            "pop_size": 1
        },
        "perturbation": {
            "pop_size": 1
        }
    }).run()
