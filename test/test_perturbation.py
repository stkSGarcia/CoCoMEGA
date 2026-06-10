import random
from copy import deepcopy
from unittest import TestCase

from impl.ads.mr.mr import PerturbationFactory, Operation, Decreasing
from impl.core.mr.base_mr import MRSet, MR
from impl.ads.scenario.scenario_definition import Boundary, ScenarioDefinition


class TestPerturbation(TestCase):
    def setUp(self):
        self.cxpb = 1.0
        self.mutpb = 1.0
        self.eta = 0.1
        regions = list(Boundary.Region) + [None]
        self.factories = [
            PerturbationFactory("vehicle", random.choice(regions), Operation.ADD),
            PerturbationFactory("vehicle", random.choice(regions), Operation.REMOVE),
            PerturbationFactory("vehicle", random.choice(regions), Operation.REPLACE),

            PerturbationFactory("walker", random.choice(regions), Operation.ADD),
            PerturbationFactory("walker", random.choice(regions), Operation.REMOVE),
            PerturbationFactory("walker", random.choice(regions), Operation.REPLACE),

            PerturbationFactory("weather", Boundary({"weather": [7, 11]})),
            PerturbationFactory("brightness", Boundary({"brightness": [0, 2]})),

            PerturbationFactory(("vehicle", "model"), Boundary({"model": [0, 22]})),
            PerturbationFactory(("vehicle", "speed"), Boundary({"speed": [0.0, 35.0]})),

            PerturbationFactory(("ego", "speed"), Boundary({"speed": [0.0, 35.0]})),
            PerturbationFactory(("ego", "model"), Boundary({"model": [0, 22]})),
            PerturbationFactory(("ego", "position"), Boundary({"position": [10, 30]})),
        ]

    def test_mate(self):
        # Different categories and operations.
        for i in range(len(self.factories)):
            for j in range(i + 1, len(self.factories)):
                perturbation1, perturbation2 = self.factories[i].spawn(), self.factories[j].spawn()
                original_perturbation1, original_perturbation2 = deepcopy(perturbation1), deepcopy(perturbation2)
                perturbation1.mate(perturbation2, cxpb=self.cxpb)
                self.assertEqual(perturbation1, original_perturbation1)
                self.assertEqual(perturbation2, original_perturbation2)

        # Same categories and operations.
        for factory in self.factories:
            perturbation1, perturbation2 = factory.spawn(), factory.spawn()
            original_perturbation1, original_perturbation2 = deepcopy(perturbation1), deepcopy(perturbation2)
            perturbation1.mate(perturbation2, cxpb=self.cxpb)
            self.assertEqual(perturbation1, original_perturbation2)
            self.assertEqual(perturbation2, original_perturbation1)

    def test_mutate(self):
        for factory in self.factories:
            perturbation = factory.spawn()
            original_perturbation = deepcopy(perturbation)
            perturbation.mutate(mutpb=self.mutpb, eta=self.eta)
            self.assertNotEqual(perturbation, original_perturbation)

    def test_dist(self):
        # Different categories and operations.
        for i in range(len(self.factories)):
            for j in range(i + 1, len(self.factories)):
                perturbation1, perturbation2 = self.factories[i].spawn(), self.factories[j].spawn()
                dist = perturbation1.dist(perturbation2)
                self.assertEqual(dist, 1.0)

        # Same categories and operations.
        zeros = ones = 0
        for factory in self.factories:
            perturbation1, perturbation2 = factory.spawn(), factory.spawn()
            dist = perturbation1.dist(perturbation2)
            if dist == 0: zeros += 1
            if dist == 1: ones += 1
        self.assertLess(zeros, len(self.factories))
        self.assertLess(ones, len(self.factories))

        # Same perturbations.
        for factory in self.factories:
            perturbation1 = factory.spawn()
            perturbation2 = deepcopy(perturbation1)
            dist = perturbation1.dist(perturbation2)
            self.assertEqual(dist, 0.0)


class TestPerturbations(TestCase):
    def setUp(self):
        self.cxpb = 1.0
        self.mutpb = 1.0
        self.eta = 0.1
        relation = Decreasing("velocity")
        self.mr_set = MRSet([
            MR([
                PerturbationFactory("vehicle", Boundary.Region.LEFT, Operation.ADD),
                PerturbationFactory("vehicle", Boundary.Region.FOCUS, Operation.REMOVE),
                PerturbationFactory("vehicle", Boundary.Region.RIGHT, Operation.REPLACE),
            ], relation),
            MR([
                PerturbationFactory("static", Boundary.Region.LEFT, Operation.REPLACE),
                PerturbationFactory("static", Boundary.Region.FOCUS, Operation.REMOVE),
                PerturbationFactory("static", Boundary.Region.RIGHT, Operation.ADD),
            ], relation),
            MR([
                PerturbationFactory("weather", Boundary({"weather": [7, 11]})),
                PerturbationFactory("brightness", Boundary({"brightness": [0, 2]})),
            ], relation),
            MR([
                PerturbationFactory(("ego", "speed"), Boundary({"speed": [0.0, 35.0]})),
                PerturbationFactory(("ego", "model"), Boundary({"model": [0, 22]})),
            ], relation),
            MR([
                PerturbationFactory(("vehicle", "model"), Boundary({"model": [0, 22]})),
                PerturbationFactory(("vehicle", "speed"), Boundary({"speed": [0.0, 35.0]})),
            ], relation),
        ], source_gen_func=ScenarioDefinition.generate_random)

    def test_mate(self):
        sequence1 = self.mr_set.generate_perturbation()
        sequence2 = self.mr_set.generate_perturbation()
        original_sequence1 = deepcopy(sequence1)
        original_sequence2 = deepcopy(sequence2)
        sequence1.mate(sequence2, cxpb=self.cxpb)
        self.assertEqual(sequence1, original_sequence2)
        self.assertEqual(sequence2, original_sequence1)

    def test_mutate(self):
        sequence = self.mr_set.generate_perturbation()
        original_sequence = deepcopy(sequence)
        sequence.mutate(mutpb=self.mutpb, eta=self.eta)
        self.assertNotEqual(sequence, original_sequence)

    def test_dist(self):
        # Different sequences.
        sequence1 = self.mr_set.generate_perturbation()
        sequence2 = self.mr_set.generate_perturbation()
        dist = sequence1.dist(sequence2)
        self.assertNotEqual(dist, 0.0)

        # Same sequences.
        sequence1 = self.mr_set.generate_perturbation()
        sequence2 = deepcopy(sequence1)
        dist = sequence1.dist(sequence2)
        self.assertEqual(dist, 0.0)
