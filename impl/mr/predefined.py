from impl.mr.mr import Decreasing, Invariance, MR, ActorPerturbationFactory, EnvPerturbationFactory, MRSet

__all__ = ["mr_set1"]

relation_slow = Decreasing("speed")
relation_slow_30 = Decreasing("speed", threshold=0.3)
relation_steer = Invariance("steering")

pedestrian_pert_factory = ActorPerturbationFactory(
    "pedestrian",
    [0.0, 10.0],
    [0.0, 10.0],
    [-90.0, 90.0],
    [-90.0, 90.0],
    [0.0, 20.0],
    [0.0, 20.0],
    [0, 1],
    [0.0, 100.0],
)

vehicle_pert_factory = ActorPerturbationFactory(
    "vehicle",
    [0.0, 50.0],
    [0.0, 50.0],
    [-90.0, 90.0],
    [-90.0, 90.0],
    [0.0, 100.0],
    [0.0, 100.0],
    [0, 5],
    [0.0, 100.0],
    [0.0, 20.0],
    [0.0, 20.0],
)

object_pert_factory = ActorPerturbationFactory(
    "object",
    [0.0, 50.0],
    [0.0, 50.0],
    [-90.0, 90.0],
    [-90.0, 90.0],
)

mr1 = MR([pedestrian_pert_factory], relation_slow)
mr2 = MR([object_pert_factory], relation_slow)
mr3 = MR([pedestrian_pert_factory], relation_slow_30)
mr7 = MR([EnvPerturbationFactory("darkness", [1, 1])], relation_slow)
mr8 = MR([EnvPerturbationFactory("weather", [5, 9])], relation_steer)

mr_set1 = MRSet([mr1, mr2, mr7])
