import logging
import os
import pickle
import sys
import time
from unittest import TestCase

import pandas as pd

from impl import config as cfg
from impl.problem import _fitness, mr_set, toolbox
from impl.scenario.scenario_definition import ScenarioDefinition
from impl.scenario.simulation_runner import run_scenarios
from impl.utils.visualization import Visualizer

logger = logging.getLogger("impl")


class TestIdenticalViolation:
    def __init__(self, mr_set, num_experiments=20, write_meta=True):
        super().__init__()
        self.mr_set = mr_set
        self.num_experiments = num_experiments
        self.write_meta = write_meta
        self.base_dir = cfg.CONFIG["workspace"]["test_result"]
        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)
        self.out_dir = os.path.join(self.base_dir,
                                    f'identical_violation_{int(round(time.time() * 1000))}')
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)

    def run(self):
        fitnesses, violations, meta = self._test()
        violation_rate = len([v for v in violations if v]) / len(violations)
        Visualizer.plot_histogram(
            stats=fitnesses,
            color=violations,
            out_dir=self.out_dir,
            show=True,
            name='identical_violation',
            verbose_name='Identical Violation',
        )
        Visualizer.plot_violation_monitor(
            stats=meta,
            out_dir=self.out_dir,
            show=True,
            name='violation_monitor',
            verbose_name='Violation Monitor',
        )
        return violation_rate

    def _test(self):
        i = 0
        meta = []
        fitnesses = []
        violations = []
        while i < self.num_experiments:
            source_scenarios = [ScenarioDefinition.generate_random() for _ in range(self.num_experiments - i)]
            scenarios = []
            for source in source_scenarios:
                follow_up = toolbox.clone(source)
                follow_up.assign_new_id()
                scenarios += [source, follow_up]
            results, sim_num = run_scenarios(scenarios, rerun=True)
            for source, follow_up, source_scenario, follow_up_scenario in zip(results[::2], results[1::2],
                                                                              scenarios[::2], scenarios[1::2]):
                if source is not None and follow_up is not None:
                    is_violated, fitness_value = _fitness(source, follow_up, mr_set=self.mr_set)
                    fitnesses.append(fitness_value[0])
                    violations.append(is_violated)
                    meta.append({
                        'source': source_scenario.id_,
                        'follow-up': follow_up_scenario.id_,
                        'is_violated': is_violated,
                        'fitness': fitness_value[0],
                    })
                    i += 1

        meta = pd.DataFrame(meta)
        if self.write_meta:
            meta.to_pickle(os.path.join(self.out_dir, 'meta.pkl'))
            # with open(os.path.join(self.out_dir, 'mr.pkl'), 'wb') as _file:
            #     pickle.dump(self.mr_set, _file)
        return fitnesses, violations, meta


if __name__ == '__main__':
    if len(sys.argv) > 1:
        num_experiments = int(sys.argv[1])
    else:
        num_experiments = 1

    print(f"Starting the test with {num_experiments} experiments...")
    violation_rate = TestIdenticalViolation(mr_set, num_experiments=num_experiments).run()
    print(
        f"""
        ##########################################
        Test Result:
        MR: {mr_set},
        Violation Rate: {violation_rate}
        ##########################################
        """
    )
