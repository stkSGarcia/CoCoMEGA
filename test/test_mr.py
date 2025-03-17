from copy import deepcopy
from unittest import TestCase

import test
from impl.mr.mr import PerturbationFactory, Operation
from impl.scenario.scenario_definition import Boundary, ScenarioDefinition, Vehicle
from impl.utils.carla_utils import initialize_carla

config = test.CONFIG


class TestMR(TestCase):
    def setUp(self):
        initialize_carla()

    def test_add_actor(self):
        for region in list(Boundary.Region) + [None]:
            scenario = ScenarioDefinition.generate_random()
            original_length = len(scenario.vehicles)
            perturbation = PerturbationFactory("vehicle", region, Operation.ADD, mark=True).spawn()
            if region is not None:
                self.assertEqual(perturbation.value.region, region)
            perturbation.perturb(scenario)
            self.assertEqual(original_length + 1, len(scenario.vehicles))

    def test_remove_actor(self):
        for region in list(Boundary.Region) + [None]:
            scenario = ScenarioDefinition.generate_random()
            scenario.vehicles += [Vehicle.generate_random(region) for _ in range(3)]
            original_length = len([v for v in scenario.vehicles if region is None or v.region == region])
            perturbation = PerturbationFactory("vehicle", region, Operation.REMOVE, mark=True).spawn()
            perturbation.perturb(scenario)
            self.assertEqual((original_length - 1) if original_length > 0 else original_length,
                             len([v for v in scenario.vehicles if region is None or v.region == region]))

    def test_replace_actor(self):
        for region in list(Boundary.Region) + [None]:
            scenario = ScenarioDefinition.generate_random()
            scenario.vehicles += [Vehicle.generate_random(region) for _ in range(3)]
            original_scenario = deepcopy(scenario)
            perturbation = PerturbationFactory("vehicle", region, Operation.REPLACE, mark=True).spawn()
            if region is not None:
                self.assertEqual(perturbation.value.region, region)
            perturbation.perturb(scenario)
            original_actors = [v for v in original_scenario.vehicles if region is None or v.region == region]
            if len(original_actors) > 0:
                self.assertEqual(len(original_actors),
                                 len([v for v in scenario.vehicles if region is None or v.region == region]))
                self.assertNotEqual(scenario.dist(original_scenario), 0.0)

    def test_actor_changes(self):
        factories = [PerturbationFactory("vehicle", Boundary.Region.FOCUS, Operation.REPLACE, mark=True),
                     PerturbationFactory("vehicle", Boundary.Region.FOCUS, Operation.REMOVE, mark=True),
                     PerturbationFactory("vehicle", Boundary.Region.FOCUS, Operation.ADD, mark=True)]
        scenario = ScenarioDefinition._generate_empty_scenario()
        scenario.set_trajectory(ScenarioDefinition._random_predefined_trajectory())
        original_actors = [Vehicle.generate_random(Boundary.Region.FOCUS) for _ in range(3)]
        farthest_actor = sorted(original_actors, key=lambda v: v.radius, reverse=True)[0]
        scenario.vehicles += deepcopy(original_actors)
        for factory in factories:
            factory.spawn().perturb(scenario)
        self.assertEqual(len(scenario.vehicles), 3)
        self.assertEqual(scenario.vehicles[0], farthest_actor)

    def test_ego(self):
        for category in ("speed", "model", "position"):
            scenario = ScenarioDefinition.generate_random()
            original_scenario = deepcopy(scenario)
            original = deepcopy(scenario.trajectory if category == "position" else scenario.ego_vehicle)
            perturbation = PerturbationFactory(("ego", category), Boundary({
                "speed": [0.0, 35.0],
                "model": [0, 22],
                "position": [100, 1000],
            })).spawn()
            perturbation.perturb(scenario)
            self.assertNotEqual(scenario, original_scenario)
            if category == "position":
                scenario.trajectory = original
                self.assertEqual(scenario, original_scenario)
            else:
                scenario.ego_vehicle = original
                self.assertEqual(scenario, original_scenario)

    def test_change_actor_attribute(self):
        for category in ("speed", "model"):
            scenario = ScenarioDefinition.generate_random()
            scenario.vehicles += [Vehicle.generate_random() for _ in range(3)]
            scenario.vehicles[0].mark = True
            original_scenario = deepcopy(scenario)
            marked_actor = deepcopy(scenario.vehicles[0])
            perturbation = PerturbationFactory(("vehicle", category), Boundary({
                "speed": [0.0, 35.0],
                "model": [0, 22],
            })).spawn()
            perturbation.perturb(scenario)
            self.assertNotEqual(scenario, original_scenario)
            scenario.vehicles[0] = marked_actor
            self.assertEqual(scenario, original_scenario)
