import random
from copy import deepcopy
from unittest import TestCase

from impl.ads.scenario.scenario_definition import ScenarioDefinition, Boundary
from impl.ads.scenario.scenario_definition import Vehicle, Walker, Static
from impl.ads.utils.carla_utils import initialize_carla


class TestActor(TestCase):
    def setUp(self):
        self.clazz = [Vehicle, Walker, Static]
        self.cxpb = 1.0
        self.mutpb = 1.0
        self.eta = 0.1

    def test_dist(self):
        for c in self.clazz:
            actor1 = c.generate_random()
            actor2 = c.generate_random()
            dist = actor1.dist(actor2)
            self.assertNotEqual(dist, 0.0)

    def test_mate(self):
        for c in self.clazz:
            actor1 = c.generate_random()
            actor2 = c.generate_random()
            original_actor1 = deepcopy(actor1)
            original_actor2 = deepcopy(actor2)
            actor1.mate(actor2, cxpb=self.cxpb)

            for attr in ["radius", "angle", "yaw", "model", "speed"]:
                self.assertEqual(getattr(actor1, attr, None), getattr(original_actor2, attr, None))
                self.assertEqual(getattr(actor2, attr, None), getattr(original_actor1, attr, None))
            self.assertEqual(getattr(actor1, "region"), getattr(original_actor2, "region"))
            self.assertEqual(getattr(actor2, "region"), getattr(original_actor1, "region"))

    def test_mutate(self):
        for c in self.clazz:
            actor = c.generate_random()
            original_actor = deepcopy(actor)
            actor.mutate(mutpb=self.mutpb, eta=self.eta)
            self.assertNotEqual(actor, original_actor)

    def test_region(self):
        for c in self.clazz:
            for region in Boundary.Region:
                actor = c.generate_random(region=region)
                self.assertEqual(actor.region, region)


class TestScenario(TestCase):
    def setUp(self):
        self.cxpb = 1.0
        self.mutpb = 0.95
        self.eta = 0.1
        initialize_carla()

    def test_dist(self):
        # Different scenarios.
        scenario1 = ScenarioDefinition.generate_random()
        scenario2 = ScenarioDefinition.generate_random()
        vehicle1 = Vehicle.generate_random()
        vehicle2 = deepcopy(vehicle1)
        scenario1.vehicles.append(vehicle1)
        scenario2.vehicles.append(vehicle2)
        scenario1.walkers = []
        scenario1.statics = []
        scenario2.statics = []
        dist = scenario1.dist(scenario2)
        self.assertNotEqual(dist, 0.0)

        # Same scenarios.
        scenario1 = ScenarioDefinition.generate_random()
        scenario3 = deepcopy(scenario1)
        dist = scenario1.dist(scenario3)
        self.assertEqual(dist, 0.0)

        for actors in (scenario3.vehicles, scenario3.walkers, scenario3.statics):
            random.shuffle(actors)
        dist = scenario1.dist(scenario3)
        self.assertEqual(dist, 0.0)

    def test_mate(self):
        scenario1 = ScenarioDefinition.generate_random()
        scenario2 = ScenarioDefinition.generate_random()
        original_scenario1 = deepcopy(scenario1)
        original_scenario2 = deepcopy(scenario2)
        scenario1.mate(scenario2, cxpb=self.cxpb)

        invariants = ["id_", "ego_vehicle", "trajectory"]
        for invariant in invariants:
            self.assertEqual(getattr(scenario1, invariant), getattr(original_scenario1, invariant))
            self.assertEqual(getattr(scenario2, invariant), getattr(original_scenario2, invariant))

        variants = ["vehicles", "walkers", "statics", "weather", "brightness"]
        for variant in variants:
            if isinstance(getattr(scenario1, variant), list):
                actors1 = getattr(scenario1, variant)
                actors2 = getattr(scenario2, variant)
                original_actors1 = getattr(original_scenario1, variant)
                original_actors2 = getattr(original_scenario2, variant)
                self.assertEqual(len(actors1), len(original_actors2))
                self.assertEqual(len(original_actors1), len(actors2))
                for i in range(len(actors1)):
                    self.assertEqual(actors1[i], original_actors2[i])
                for i in range(len(actors2)):
                    self.assertEqual(actors2[i], original_actors1[i])
            else:
                self.assertEqual(getattr(scenario1, variant), getattr(original_scenario2, variant))
                self.assertEqual(getattr(original_scenario1, variant), getattr(scenario2, variant))

    def test_mutate(self):
        original_scenario = ScenarioDefinition.generate_random()
        variants = ["vehicles", "walkers", "statics", "weather", "brightness"]
        for _ in range(10):
            scenario = deepcopy(original_scenario)
            scenario.mutate(mutpb=self.mutpb, eta=self.eta)
            if any(getattr(scenario, variant) != getattr(original_scenario, variant)
                   for variant in variants):
                break

        invariants = ["id_", "ego_vehicle", "trajectory"]
        for invariant in invariants:
            self.assertEqual(getattr(scenario, invariant), getattr(original_scenario, invariant))

        self.assertTrue(any(
            getattr(scenario, variant) != getattr(original_scenario, variant)
            for variant in variants
        ))

    def test_random_marked_actors(self):
        scenario = ScenarioDefinition.generate_random_with_marked_actors()
        marked_actors = [actor for actor in scenario.vehicles + scenario.walkers + scenario.statics if actor.mark]
        self.assertEqual(len(marked_actors), 1)
