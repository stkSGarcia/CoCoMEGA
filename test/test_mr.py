from copy import deepcopy
from unittest import TestCase

import test
from impl.mr.mr import PerturbationFactory, Operation
from impl.scenario.carla_utils import initialize_carla
from impl.scenario.scenario_definition import Boundary, ScenarioDefinition, Vehicle

config = test.CONFIG


class TestMR(TestCase):
    def setUp(self):
        initialize_carla()

    def test_add(self):
        print("==========Add==========")
        for region in list(Boundary.Region) + [None]:
            print(f"**********Region: {region}**********")
            scenario = ScenarioDefinition.generate_random()
            print(scenario)
            original_length = len(scenario.vehicles)
            factory = PerturbationFactory("vehicle", region, Operation.ADD, mark=True)
            perturbation = factory.spawn()
            print(perturbation)
            if region is not None:
                self.assertEqual(perturbation.value.region, region)
            perturbation.perturb(scenario)
            print(scenario)
            self.assertEqual(original_length + 1, len(scenario.vehicles))

    def test_remove(self):
        print("==========Remove==========")
        for region in list(Boundary.Region) + [None]:
            print(f"**********Region: {region}**********")
            scenario = ScenarioDefinition.generate_random()
            scenario.vehicles += [Vehicle.generate_random(region) for _ in range(3)]
            print(scenario)
            original_length = len([v for v in scenario.vehicles if region is None or v.region == region])
            factory = PerturbationFactory("vehicle", region, Operation.REMOVE, mark=True)
            perturbation = factory.spawn()
            print(perturbation)
            perturbation.perturb(scenario)
            print(scenario)
            self.assertEqual((original_length - 1) if original_length > 0 else original_length,
                             len([v for v in scenario.vehicles if region is None or v.region == region]))

    def test_replace(self):
        print("==========Replace==========")
        for region in list(Boundary.Region) + [None]:
            print(f"**********Region: {region}**********")
            scenario = ScenarioDefinition.generate_random()
            scenario.vehicles += [Vehicle.generate_random(region) for _ in range(3)]
            print(scenario)
            original_scenario = deepcopy(scenario)
            factory = PerturbationFactory("vehicle", region, Operation.REPLACE, mark=True)
            perturbation = factory.spawn()
            print(perturbation)
            if region is not None:
                self.assertEqual(perturbation.value[0], region)
                self.assertEqual(perturbation.value[1].region, region)
            perturbation.perturb(scenario)
            print(scenario)
            original_actors = [v for v in original_scenario.vehicles if region is None or v.region == region]
            if len(original_actors) > 0:
                self.assertEqual(len(original_actors),
                                 len([v for v in scenario.vehicles if region is None or v.region == region]))
                self.assertNotEqual(scenario.dist(original_scenario), 0.0)

    def test_actor_changes(self):
        print("==========Changes==========")
        factories = [PerturbationFactory("vehicle", Boundary.Region.FOCUS, Operation.REPLACE, mark=True),
                     PerturbationFactory("vehicle", Boundary.Region.FOCUS, Operation.REMOVE, mark=True),
                     PerturbationFactory("vehicle", Boundary.Region.FOCUS, Operation.ADD, mark=True)]
        scenario = ScenarioDefinition._generate_empty_scenario()
        original_actors = [Vehicle.generate_random(Boundary.Region.FOCUS) for _ in range(3)]
        farthest_actor = sorted(original_actors, key=lambda v: v.radius, reverse=True)[0]
        scenario.vehicles += deepcopy(original_actors)
        print(scenario)
        for factory in factories:
            factory.spawn().perturb(scenario)
        print(scenario)
        self.assertEqual(len(scenario.vehicles), 3)
        self.assertEqual(scenario.vehicles[0], farthest_actor)
