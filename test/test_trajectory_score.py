from copy import deepcopy

import test
import os
import time
import pandas as pd

from impl.ads.evaluation.simulation_runner import ADSEvaluator
from impl.ads.mr.mr import PerturbationFactory, Operation
from impl.ads.utils.carla_utils import initialize_carla
from impl.ads.scenario.scenario_definition import Boundary, ScenarioDefinition
from impl.ads.utils.trajectory import single_trajectory_score

config = test.CONFIG


class TestTrajectoryScore:

    def __init__(self, pert_pop_size=10):
        self.pert_pop_size = pert_pop_size
        self.base_dir = config["workspace"]["test_result"]
        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)

        self.out_dir = os.path.join(self.base_dir, f'traj_score_{int(round(time.time() * 1000))}')

        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)

    def test(self):
        initialize_carla(2000)
        pert_fact = PerturbationFactory("walker", Boundary.Region.FOCUS, Operation.ADD, mark=True)
        scen = ScenarioDefinition.generate_random()
        pert_pop = [pert_fact.spawn() for _ in range(self.pert_pop_size)]
        perturbed = []
        results = pd.DataFrame()
        for i, pert in enumerate(pert_pop):
            traj_score = 0
            scenario = deepcopy(scen)
            scenario.assign_new_id()
            pert.perturb(scenario)
            perturbed.append(scenario)
            pert_route = scenario.build_actor_trajectory(pert.value, scenario_duration=20)
            traj_score += single_trajectory_score([t[0] for t in scenario.trajectory["route"]], pert_route)
            results = pd.concat([results, pd.DataFrame([{
                "scen_id": scenario.id_,
                "pert_idx": i,
                "traj_score": traj_score,
            }])], ignore_index=True)

        ADSEvaluator(mr_set=None).run_scenarios(perturbed)
        results.to_csv(os.path.join(self.out_dir, "results.csv"))


if __name__ == "__main__":
    tts = TestTrajectoryScore(pert_pop_size=10)
    tts.test()
