from copy import deepcopy
from unittest import TestCase

import test
from impl.mr.mr import PerturbationFactory, Operation, Decreasing, MR, MRSet, Perturbations
from impl.scenario.scenario_definition import Boundary, ScenarioDefinition

config = test.CONFIG


class TestMR(TestCase):
    def test_add(self):
        print("==========Perturbation: Add==========")
        for region in list(Boundary.Region) + [None]:
            print(f"**********Region: {region}**********")
            scenario = ScenarioDefinition.generate_random()
            print(scenario)
            origin_length = len(scenario.vehicles)
            factory = PerturbationFactory("vehicle", region, Operation.ADD)
            perturbation = factory.spawn()
            print(perturbation)
            if region is not None:
                self.assertEqual(perturbation.value.region, region)

            perturbation.perturb(scenario)
            print(scenario)
            self.assertEqual(origin_length + 1, len(scenario.vehicles))

    def test_remove(self):
        print("==========Perturbation: Remove==========")
        for region in list(Boundary.Region) + [None]:
            print(f"**********Region: {region}**********")
            scenario = ScenarioDefinition.generate_random()
            print(scenario)
            origin_length = len([v for v in scenario.vehicles if region is None or v.region == region])
            factory = PerturbationFactory("vehicle", region, Operation.REMOVE)
            perturbation = factory.spawn()
            print(perturbation)

            perturbation.perturb(scenario)
            print(scenario)
            self.assertEqual((origin_length - 1) if origin_length > 0 else origin_length,
                             len([v for v in scenario.vehicles if region is None or v.region == region]))

    def test_replace(self):
        print("==========Perturbation: Replace==========")
        for region in list(Boundary.Region) + [None]:
            print(f"**********Region: {region}**********")
            scenario = ScenarioDefinition.generate_random()
            print(scenario)
            origin_scenario = deepcopy(scenario)
            factory = PerturbationFactory("vehicle", region, Operation.REPLACE)
            perturbation = factory.spawn()
            print(perturbation)
            if region is not None:
                self.assertEqual(perturbation.value[0], region)
                self.assertEqual(perturbation.value[1].region, region)

            perturbation.perturb(scenario)
            print(scenario)
            origin_actors = [v for v in origin_scenario.vehicles if region is None or v.region == region]
            if len(origin_actors) > 0:
                self.assertEqual(len(origin_actors),
                                 len([v for v in scenario.vehicles if region is None or v.region == region]))
                self.assertNotEqual(scenario.dist(origin_scenario), 0.0)

    def test_dist(self):
        print("==========Perturbation: Dist==========")
        relation_slow = Decreasing("velocity")
        mr1 = MR([
            PerturbationFactory("vehicle", Boundary.Region.LEFT, Operation.ADD),
            PerturbationFactory("vehicle", Boundary.Region.FOCUS, Operation.REMOVE),
            PerturbationFactory("vehicle", Boundary.Region.RIGHT, Operation.REPLACE),
        ], relation_slow)
        mr2 = MR([
            PerturbationFactory("static", Boundary.Region.LEFT, Operation.REPLACE),
            PerturbationFactory("static", Boundary.Region.FOCUS, Operation.REMOVE),
            PerturbationFactory("static", Boundary.Region.RIGHT, Operation.ADD),
        ], relation_slow)
        mr_set = MRSet([mr1, mr2])

        sequence1 = Perturbations([mr_set.spawn() for _ in range(10)])
        sequence2 = Perturbations([mr_set.spawn() for _ in range(20)])
        print(sequence1)
        print(sequence2)

        dist = sequence1.dist(sequence2)
        print(dist)

    def test_same_dist(self):
        print("==========Perturbation: Dist zero==========")
        relation_slow = Decreasing("velocity")
        mr1 = MR([
            PerturbationFactory("vehicle", Boundary.Region.LEFT, Operation.ADD),
            PerturbationFactory("vehicle", Boundary.Region.FOCUS, Operation.REMOVE),
            PerturbationFactory("vehicle", Boundary.Region.RIGHT, Operation.REPLACE),
        ], relation_slow)
        mr2 = MR([
            PerturbationFactory("static", Boundary.Region.LEFT, Operation.REPLACE),
            PerturbationFactory("static", Boundary.Region.FOCUS, Operation.REMOVE),
            PerturbationFactory("static", Boundary.Region.RIGHT, Operation.ADD),
        ], relation_slow)
        mr_set = MRSet([mr1, mr2])

        sequence1 = Perturbations([mr_set.spawn() for _ in range(10)])
        sequence2 = deepcopy(sequence1)
        sequence2.insert(5, deepcopy(sequence2[5]))
        sequence2.insert(9, deepcopy(sequence2[9]))
        print(sequence1)
        print(sequence2)

        dist = sequence1.dist(sequence2)
        print(dist)
        self.assertEqual(dist, 0.0)
