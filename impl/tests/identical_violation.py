import logging
import os.path
import sys

from impl.config import CONFIG
from impl.problem import _fitness, mr_set, toolbox
from impl.scenario.scenario_definition import ScenarioDefinition
from impl.scenario.simulation_runner import run_scenarios
from impl.utils.visualization import Visualizer


class IdenticalViolationTest:
    def __init__(self, mr_list, num_experiments=20):
        self.mr_list = mr_list
        self.num_experiments = num_experiments
        self.out_dir = os.path.join(CONFIG['workspace'], 'tests')

    def test(self):
        violation_rates = {}
        for mr in self.mr_list:
            fitnesses, violations = self._test_mr(mr)
            violation_rate = len([v for v in violations if v]) / len(violations)
            violation_rates[mr] = violation_rate
            Visualizer.plot_histogram(
                stats=fitnesses,
                color=violations,
                out_dir=self.out_dir,
                show=True,
                name='identical_violation',
                verbose_name='Identical Violation',
            )
        return violation_rates

    def _test_mr(self, mr):
        i = 0
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
            for source, follow_up in zip(results[::2], results[1::2]):
                if source is not None and follow_up is not None:
                    is_violated, fitness_value = _fitness(source, follow_up, mr_set=mr)
                    fitnesses.append(fitness_value[0])
                    violations.append(is_violated)
                    i += 1

        return fitnesses, violations


if __name__ == '__main__':
    if len(sys.argv) > 1:
        num_experiments = int(sys.argv[1])
    else:
        num_experiments = 10
    violation_rates = IdenticalViolationTest([mr_set], num_experiments=num_experiments).test()
    for mr, violation_rate in violation_rates.items():
        logging.info(
            f"""
            ##########################################
            Test Result:
            MR: {mr},
            Violation Rate: {violation_rate}
            ##########################################
            """
        )
