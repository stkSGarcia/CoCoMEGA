import os

os.chdir("..")
from impl import config

config.init_config()

from copy import deepcopy
from unittest import TestCase

from impl.scenario.scenario_definition import ScenarioDefinition
from impl.scenario.scenario_definition import Vehicle, Walker, Static


class TestActor(TestCase):
    def setUp(self):
        self.clazz = [Vehicle, Walker, Static]
        config.CONFIG["scenario"]["cxpb"] = 1.0
        config.CONFIG["scenario"]["mutpb"] = 1.0

    def test_dist(self):
        for c in self.clazz:
            print(f"=========={c.__name__}: Dist==========")
            a = c.generate_random()
            b = c.generate_random()
            print(a)
            print(b)
            print(a.dist(b))

    def test_mate(self):
        for c in self.clazz:
            print(f"=========={c.__name__}: Mate==========")
            a = c.generate_random()
            b = c.generate_random()
            print(a)
            print(b)
            a.mate(b)
            print(a)
            print(b)

    def test_mutate(self):
        for c in self.clazz:
            print(f"=========={c.__name__}: Mutate==========")
            a = c.generate_random()
            b = c.generate_random()
            print(a)
            print(b)
            a.mutate()
            b.mutate()
            print(a)
            print(b)


class TestScenarioDefinition(TestCase):
    def setUp(self):
        config.CONFIG["scenario"]["cxpb"] = 1.0
        config.CONFIG["scenario"]["mutpb"] = 1.0
        config.CONFIG["scenario"]["eta"] = 0.1

    def test_mock(self):
        print("==========Mock==========")
        print(ScenarioDefinition.mock())

    def test_dist(self):
        print("==========Dist==========")
        vehicle1 = Vehicle.generate_random()
        vehicle2 = deepcopy(vehicle1)
        vehicle2.mutate()
        scenario1 = ScenarioDefinition.generate_random()
        scenario2 = ScenarioDefinition.generate_random()
        scenario1.vehicles.append(vehicle1)
        scenario2.vehicles.append(vehicle2)
        print(scenario1)
        print(scenario2)
        dist = scenario1.dist(scenario2)
        print(dist)
        self.assertLessEqual(dist, 1.0)
        self.assertGreaterEqual(dist, 0.0)

    def test_same_dist(self):
        print("==========Dist zero==========")
        scenario1 = ScenarioDefinition.mock()
        scenario2 = ScenarioDefinition.mock()
        print(scenario1)
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

        variants = ["vehicles", "walkers", "statics", "weather", "darkness"]
        for variant in variants:
            if isinstance(getattr(scenario1, variant), list):
                actors1 = getattr(scenario1, variant)
                actors2 = getattr(scenario2, variant)
                actors1_origin = getattr(scenario1_origin, variant)
                actors2_origin = getattr(scenario2_origin, variant)
                self.assertEqual(len(actors1), len(actors2_origin))
                self.assertEqual(len(actors1_origin), len(actors2))
                size = min(len(actors1), len(actors2), len(actors1_origin), len(actors2_origin))
                for i in range(size):
                    self.assertEqual(actors1[i].id_, actors1_origin[i].id_)
                    self.assertEqual(actors2[i].id_, actors2_origin[i].id_)
                    actors1[i].id_ = actors1_origin[i].id_ = actors2[i].id_ = actors2_origin[i].id_ = "temp"
                    self.assertEqual(actors1[i], actors2_origin[i])
                    self.assertEqual(actors1_origin[i], actors2[i])
            else:
                self.assertEqual(getattr(scenario1, variant), getattr(scenario2_origin, variant))
                self.assertEqual(getattr(scenario1_origin, variant), getattr(scenario2, variant))

    def test_mutate(self):
        print("==========Mutate==========")
        scenario1 = ScenarioDefinition.generate_random()
        scenario2 = ScenarioDefinition.generate_random()
        scenario1_origin = deepcopy(scenario1)
        scenario2_origin = deepcopy(scenario2)
        print(scenario1)
        print(scenario2)
        scenario1.mutate()
        scenario2.mutate()
        print(scenario1)
        print(scenario2)

        invariants = ["id_", "ego_vehicle", "trajectory"]
        for invariant in invariants:
            self.assertEqual(getattr(scenario1, invariant), getattr(scenario1_origin, invariant))
            self.assertEqual(getattr(scenario2, invariant), getattr(scenario2_origin, invariant))

        variants = ["vehicles", "walkers", "statics", "weather", "darkness"]
        for variant in variants:
            self.assertNotEqual(getattr(scenario1, variant), getattr(scenario1_origin, variant))
            self.assertNotEqual(getattr(scenario2, variant), getattr(scenario2_origin, variant))
