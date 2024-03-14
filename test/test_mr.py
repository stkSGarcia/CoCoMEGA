import os

os.chdir("..")
from impl import config

config.init_config()

import random
from copy import deepcopy
from unittest import TestCase

from impl.mr.mr import PerturbationFactory, Perturbation
from impl.scenario.scenario_definition import Boundary, ScenarioDefinition


class TestMR(TestCase):
    def test_perturbation(self):
        actor_factory = PerturbationFactory("vehicle", Boundary({
            "x": [-100.0, 100.0],
            "y": [-100.0, 100.0],
            "z": [-100.0, 100.0],
            "pitch": [-180.0, 180.0],
            "yaw": [-180.0, 180.0],
            "roll": [-180.0, 180.0],
            "speed": [20.0, 60.0],
            "model": [5, 20],
            "color": [0, 2],
            "autopilot": [0, 1],
        }))
        perturbation1 = actor_factory.spawn()
        print(perturbation1)

        env_factory = PerturbationFactory("weather", Boundary({"weather": [4, 10]}))
        perturbation2 = env_factory.spawn()
        print(perturbation2)

        scenario = ScenarioDefinition.generate_random()
        print(scenario)

        perturbation1.perturb(scenario)
        print(scenario)
        perturbation2.perturb(scenario)
        print(scenario)

    def test_dist(self):
        from impl.mr.predefined import mr_set1
        sequence1 = [random.choice(mr_set1.mrs).generate_perturbation() for _ in range(20)]
        sequence2 = [random.choice(mr_set1.mrs).generate_perturbation() for _ in range(10)]
        sequence1_origin = deepcopy(sequence1)
        print(sequence1)
        print(sequence2)

        scenario1 = Perturbation.squash(sequence1)
        scenario2 = Perturbation.squash(sequence2)
        scenario1_origin = Perturbation.squash(sequence1_origin)
        random.shuffle(scenario1_origin.walkers)
        random.shuffle(scenario1_origin.statics)
        print([a["id"] for a in scenario1.get_other_actors()])
        print([a["id"] for a in scenario1_origin.get_other_actors()])

        dist1 = scenario1.dist(scenario2)
        dist2 = scenario1.dist(scenario1_origin)
        print(dist1)
        print(dist2)
        self.assertEqual(dist2, 0.0)
