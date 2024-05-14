from impl.mr.mr import Decreasing, Invariance, MR, MRSet, PerturbationFactory, Operation
from impl.scenario.scenario_definition import Boundary

__all__ = ["mr_set1"]

relation_slow = Decreasing("velocity")
relation_slow_30 = Decreasing("velocity", threshold=0.3)
relation_steer = Invariance("steering")

vehicle_pert_factory = PerturbationFactory("vehicle", None, Operation.ADD)

walker_pert_factory = PerturbationFactory("walker", "right", Operation.ADD)

static_pert_factory = PerturbationFactory("static", "right", Operation.ADD)

mr1 = MR([walker_pert_factory], relation_slow)
mr2 = MR([static_pert_factory], relation_slow)
mr3 = MR([walker_pert_factory], relation_slow_30)
mr7 = MR([PerturbationFactory("weather", Boundary({"weather": [1, 3]}))], relation_slow)
mr8 = MR([PerturbationFactory("weather", Boundary({"weather": [0, 14]}))], relation_steer)

# mr_set1 = MRSet([mr1, mr2, mr7])
mr_set1 = MRSet([mr1])
