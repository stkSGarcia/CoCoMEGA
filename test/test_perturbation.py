import random
from copy import deepcopy
from unittest import TestCase

import test
from impl.mr.mr import PerturbationFactory, Operation, Decreasing, MR, MRSet
from impl.scenario.scenario_definition import Boundary

config = test.CONFIG


class TestPerturbation(TestCase):
    def setUp(self):
        self.cxpb = 1.0
        self.mutpb = 1.0
        self.eta = 0.1
        Boundary.Region.__repr__ = lambda x: x.name
        self.factories = [
            PerturbationFactory("vehicle", random.choice(list(Boundary.Region)), Operation.ADD),
            PerturbationFactory("vehicle", random.choice(list(Boundary.Region)), Operation.REMOVE),
            PerturbationFactory("vehicle", random.choice(list(Boundary.Region)), Operation.REPLACE),
            PerturbationFactory("walker", random.choice(list(Boundary.Region)), Operation.ADD),
            PerturbationFactory("walker", random.choice(list(Boundary.Region)), Operation.REMOVE),
            PerturbationFactory("walker", random.choice(list(Boundary.Region)), Operation.REPLACE),
        ]
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
            ], relation)
        ])

    def test_mate(self):
        print("==========Mate==========")
        print("**********Different**********")
        for i in range(len(self.factories)):
            for j in range(len(self.factories)):
                if i == j: continue
                perturbation1, perturbation2 = self.factories[i].spawn(), self.factories[j].spawn()
                original_perturbation1, original_perturbation2 = deepcopy(perturbation1), deepcopy(perturbation2)
                print(perturbation1)
                print(perturbation2)
                perturbation1.mate(perturbation2, cxpb=self.cxpb)
                print(perturbation1)
                print(perturbation2)
                self.assertEqual(perturbation1, original_perturbation1)
                self.assertEqual(perturbation2, original_perturbation2)

        print("**********Same**********")
        for factory in self.factories:
            perturbation1, perturbation2 = factory.spawn(), factory.spawn()
            original_perturbation1, original_perturbation2 = deepcopy(perturbation1), deepcopy(perturbation2)
            print(perturbation1)
            print(perturbation2)
            perturbation1.mate(perturbation2, cxpb=self.cxpb)
            print(perturbation1)
            print(perturbation2)
            self.assertEqual(perturbation1, original_perturbation2)
            self.assertEqual(perturbation2, original_perturbation1)

    def test_mutate(self):
        print("==========Mutate==========")
        for factory in self.factories:
            perturbation = factory.spawn()
            original_perturbation = deepcopy(perturbation)
            print(perturbation)
            perturbation.mutate(mutpb=self.mutpb, eta=self.eta)
            print(perturbation)
            self.assertNotEqual(perturbation, original_perturbation)

    def test_dist(self):
        print("==========Dist==========")
        sequence1 = self.mr_set.initialize()
        sequence2 = self.mr_set.initialize()
        print(sequence1)
        print(sequence2)
        dist = sequence1.dist(sequence2)
        print(dist)

    def test_same_dist(self):
        print("==========Dist zero==========")
        sequence1 = self.mr_set.initialize()
        sequence2 = deepcopy(sequence1)
        print(sequence1)
        print(sequence2)
        dist = sequence1.dist(sequence2)
        print(dist)
        self.assertEqual(dist, 0.0)


class TestPerturbations(TestCase):
    def setUp(self):
        self.cxpb = 1.0
        self.mutpb = 1.0
        self.eta = 0.1
        Boundary.Region.__repr__ = lambda x: x.name
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
            ], relation)
        ])

    def test_mate(self):
        print("==========Mate==========")
        sequence1 = self.mr_set.initialize()
        sequence2 = self.mr_set.initialize()
        original_sequence1 = deepcopy(sequence1)
        original_sequence2 = deepcopy(sequence2)
        print(sequence1)
        print(sequence2)
        sequence1.mate(sequence2, cxpb=self.cxpb)
        print(sequence1)
        print(sequence2)
        self.assertEqual(sequence1, original_sequence2)
        self.assertEqual(sequence2, original_sequence1)

    def test_mutate(self):
        print("==========Mutate==========")
        sequence = self.mr_set.initialize()
        original_sequence = deepcopy(sequence)
        print(sequence)
        sequence.mutate(mutpb=self.mutpb, eta=self.eta)
        print(sequence)
        self.assertNotEqual(sequence, original_sequence)

    def test_dist(self):
        print("==========Dist==========")
        sequence1 = self.mr_set.initialize()
        sequence2 = self.mr_set.initialize()
        print(sequence1)
        print(sequence2)
        dist = sequence1.dist(sequence2)
        print(dist)

    def test_same_dist(self):
        print("==========Dist zero==========")
        sequence1 = self.mr_set.initialize()
        sequence2 = deepcopy(sequence1)
        print(sequence1)
        dist = sequence1.dist(sequence2)
        print(dist)
        self.assertEqual(dist, 0.0)
