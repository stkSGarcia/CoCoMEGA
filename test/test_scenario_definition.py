import math
import random
from copy import deepcopy
from unittest import TestCase

import test
from impl.scenario.scenario_definition import ScenarioDefinition, Boundary
from impl.scenario.scenario_definition import Vehicle, Walker, Static


class TestActor(TestCase):
    def setUp(self):
        self.clazz = [Vehicle, Walker, Static]
        test.CONFIG["scenario"]["cxpb"] = 1.0
        test.CONFIG["scenario"]["mutpb"] = 1.0
        test.CONFIG["scenario"]["eta"] = 0.1

    def test_dist(self):
        for c in self.clazz:
            print(f"=========={c.__name__}: Dist==========")
            a = c.generate_random()
            b = c.generate_random()
            print(a)
            print(b)
            print(math.sqrt(a.dist(b)))

    def test_mate(self):
        for c in self.clazz:
            print(f"=========={c.__name__}: Mate==========")
            a = c.generate_random()
            b = c.generate_random()
            a_origin = deepcopy(a)
            b_origin = deepcopy(b)
            print(a)
            print(b)
            a.mate(b)
            print(a)
            print(b)

            for attr in ["radius", "angle", "yaw", "model", "speed", "autopilot"]:
                self.assertEqual(getattr(a, attr, None), getattr(b_origin, attr, None))
                self.assertEqual(getattr(b, attr, None), getattr(a_origin, attr, None))
            self.assertEqual(getattr(a, "region"), getattr(b_origin, "region"))
            self.assertEqual(getattr(b, "region"), getattr(a_origin, "region"))

    def test_mutate(self):
        for c in self.clazz:
            print(f"=========={c.__name__}: Mutate==========")
            a = c.generate_random()
            print(a)
            a.mutate()
            print(a)

    def test_region(self):
        for c in self.clazz:
            print(f"=========={c.__name__}: Region==========")
            for region in Boundary.Region:
                a = c.generate_random(region=region)
                print(a)
                self.assertEqual(a.region, region)


class TestScenarioDefinition(TestCase):
    def setUp(self):
        test.CONFIG["scenario"]["cxpb"] = 1.0
        test.CONFIG["scenario"]["mutpb"] = 1.0
        test.CONFIG["scenario"]["eta"] = 0.1

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
        scenario1_origin = deepcopy(scenario1)
        scenario2_origin = deepcopy(scenario2)
        print(scenario1)
        print(scenario2)
        scenario1.mate(scenario2)
        print(scenario1)
        print(scenario2)

        invariants = ["id_", "ego_vehicle", "trajectory"]
        for invariant in invariants:
            self.assertEqual(getattr(scenario1, invariant), getattr(scenario1_origin, invariant))
            self.assertEqual(getattr(scenario2, invariant), getattr(scenario2_origin, invariant))

        variants = ["vehicles", "walkers", "statics", "weather"]
        for variant in variants:
            if isinstance(getattr(scenario1, variant), list):
                actors1 = getattr(scenario1, variant)
                actors2 = getattr(scenario2, variant)
                actors1_origin = getattr(scenario1_origin, variant)
                actors2_origin = getattr(scenario2_origin, variant)
                self.assertEqual(len(actors1), len(actors2_origin))
                self.assertEqual(len(actors1_origin), len(actors2))
                for i in range(len(actors1)):
                    self.assertEqual(actors1[i], actors2_origin[i])
                for i in range(len(actors2)):
                    self.assertEqual(actors2[i], actors1_origin[i])
            else:
                self.assertEqual(getattr(scenario1, variant), getattr(scenario2_origin, variant))
                self.assertEqual(getattr(scenario1_origin, variant), getattr(scenario2, variant))

    def test_mutate(self):
        print("==========Mutate==========")
        scenario = ScenarioDefinition.generate_random()
        scenario_origin = deepcopy(scenario)
        print(scenario)
        scenario.mutate()
        print(scenario)

        invariants = ["id_", "ego_vehicle", "trajectory"]
        for invariant in invariants:
            self.assertEqual(getattr(scenario, invariant), getattr(scenario_origin, invariant))

        variants = ["vehicles", "walkers", "statics", "weather"]
        for variant in variants:
            if ((isinstance(getattr(scenario, variant), list) and len(getattr(scenario, variant)) != 0) or
                    not isinstance(getattr(scenario, variant), list)):
                self.assertNotEqual(getattr(scenario, variant), getattr(scenario_origin, variant))
