import math
import random
from copy import deepcopy
from unittest import TestCase

import test
from impl.utils.carla_utils import initialize_carla
from impl.scenario.scenario_definition import ScenarioDefinition, Boundary
from impl.scenario.scenario_definition import Vehicle, Walker, Static

config = test.CONFIG


class TestActor(TestCase):
    def setUp(self):
        self.clazz = [Vehicle, Walker, Static]
        self.cxpb = 1.0
        self.mutpb = 1.0
        self.eta = 0.1

    def test_dist(self):
        for c in self.clazz:
            print(f"=========={c.__name__}: Dist==========")
            actor1 = c.generate_random()
            actor2 = c.generate_random()
            print(actor1)
            print(actor2)
            print(math.sqrt(actor1.dist(actor2)))

    def test_mate(self):
        for c in self.clazz:
            print(f"=========={c.__name__}: Mate==========")
            actor1 = c.generate_random()
            actor2 = c.generate_random()
            original_actor1 = deepcopy(actor1)
            original_actor2 = deepcopy(actor2)
            print(actor1)
            print(actor2)
            actor1.mate(actor2, cxpb=self.cxpb)
            print(actor1)
            print(actor2)

            for attr in ["radius", "angle", "yaw", "model", "speed", "autopilot"]:
                self.assertEqual(getattr(actor1, attr, None), getattr(original_actor2, attr, None))
                self.assertEqual(getattr(actor2, attr, None), getattr(original_actor1, attr, None))
            self.assertEqual(getattr(actor1, "region"), getattr(original_actor2, "region"))
            self.assertEqual(getattr(actor2, "region"), getattr(original_actor1, "region"))

    def test_mutate(self):
        for c in self.clazz:
            print(f"=========={c.__name__}: Mutate==========")
            actor = c.generate_random()
            original_actor = deepcopy(actor)
            print(actor)
            actor.mutate(mutpb=self.mutpb, eta=self.eta)
            print(actor)
            self.assertNotEqual(actor, original_actor)

    def test_region(self):
        for c in self.clazz:
            print(f"=========={c.__name__}: Region==========")
            for region in Boundary.Region:
                actor = c.generate_random(region=region)
                print(actor)
                self.assertEqual(actor.region, region)


class TestScenario(TestCase):
    def setUp(self):
        self.cxpb = 1.0
        self.mutpb = 0.95
        self.eta = 0.1
        initialize_carla()

    def test_dist(self):
        print("==========Dist==========")
        scenario1 = ScenarioDefinition.generate_random()
        scenario2 = ScenarioDefinition.generate_random()
        vehicle1 = Vehicle.generate_random()
        vehicle2 = deepcopy(vehicle1)
        scenario1.vehicles.append(vehicle1)
        scenario2.vehicles.append(vehicle2)
        scenario1.walkers = []
        scenario1.statics = []
        scenario2.statics = []
        print(scenario1)
        print(scenario2)
        dist = scenario1.dist(scenario2)
        print(dist)

    def test_same_dist(self):
        print("==========Dist zero==========")
        scenario1 = ScenarioDefinition.generate_random()
        scenario2 = deepcopy(scenario1)
        print(scenario1)
        dist = scenario1.dist(scenario2)
        print(dist)
        self.assertEqual(dist, 0.0)

        random.shuffle(scenario2.vehicles)
        random.shuffle(scenario2.walkers)
        random.shuffle(scenario2.statics)
        print(scenario2)
        dist = scenario1.dist(scenario2)
        print(dist)
        self.assertEqual(dist, 0.0)

    def test_mate(self):
        print("==========Mate==========")
        scenario1 = ScenarioDefinition.generate_random()
        scenario2 = ScenarioDefinition.generate_random()
        original_scenario1 = deepcopy(scenario1)
        original_scenario2 = deepcopy(scenario2)
        print(scenario1)
        print(scenario2)
        scenario1.mate(scenario2, cxpb=self.cxpb)
        print(scenario1)
        print(scenario2)

        invariants = ["id_", "ego_vehicle", "trajectory"]
        for invariant in invariants:
            self.assertEqual(getattr(scenario1, invariant), getattr(original_scenario1, invariant))
            self.assertEqual(getattr(scenario2, invariant), getattr(original_scenario2, invariant))

        variants = ["vehicles", "walkers", "statics", "weather"]
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
        print("==========Mutate==========")
        scenario = ScenarioDefinition.generate_random()
        original_scenario = deepcopy(scenario)
        print(scenario)
        scenario.mutate(mutpb=self.mutpb, eta=self.eta)
        print(scenario)

        invariants = ["id_", "ego_vehicle", "trajectory"]
        for invariant in invariants:
            self.assertEqual(getattr(scenario, invariant), getattr(original_scenario, invariant))

        variants = ["vehicles", "walkers", "statics", "weather"]
        for variant in variants:
            if ((isinstance(getattr(scenario, variant), list) and len(getattr(scenario, variant)) != 0) or
                    not isinstance(getattr(scenario, variant), list)):
                self.assertNotEqual(getattr(scenario, variant), getattr(original_scenario, variant))
