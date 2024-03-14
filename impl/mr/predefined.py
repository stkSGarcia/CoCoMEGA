from impl.mr.mr import Decreasing, Invariance, MR, MRSet, PerturbationFactory

__all__ = ["mr_set1"]

from impl.scenario.scenario_definition import Boundary

relation_slow = Decreasing("speed")
relation_slow_30 = Decreasing("speed", threshold=0.3)
relation_steer = Invariance("steering")

vehicle_pert_factory = PerturbationFactory("vehicle", Boundary({
    "x": [-50.0, 50.0],
    "y": [-50.0, 50.0],
    "z": [0.0, 2.0],
    "pitch": [-90.0, 90.0],
    "yaw": [-90.0, 90.0],
    "roll": [-90.0, 90.0],
    "speed": [0.0, 100.0],
    "model": [0, 26],
    "color": [0, 2],
    "autopilot": [0, 1],
}))

walker_pert_factory = PerturbationFactory("walker", Boundary({
    "x": [-10.0, 10.0],
    "y": [-10.0, 10.0],
    "z": [0.0, 2.0],
    "pitch": [-90.0, 90.0],
    "yaw": [-90.0, 90.0],
    "roll": [-90.0, 90.0],
    "speed": [0.0, 10.0],
    "model": [0, 14],
}))

static_pert_factory = PerturbationFactory("static", Boundary({
    "x": [-50.0, 50.0],
    "y": [-50.0, 50.0],
    "z": [0.0, 5.0],
    "pitch": [-90.0, 90.0],
    "yaw": [-90.0, 90.0],
    "roll": [-90.0, 90.0],
    "model": [0, 89],
    "size": [0, 50],
}))

mr1 = MR([walker_pert_factory], relation_slow)
mr2 = MR([static_pert_factory], relation_slow)
mr3 = MR([walker_pert_factory], relation_slow_30)
mr7 = MR([PerturbationFactory("darkness", Boundary({"darkness": [1, 3]}))], relation_slow)
mr8 = MR([PerturbationFactory("weather", Boundary({"weather": [0, 14]}))], relation_steer)

mr_set1 = MRSet([mr1, mr2, mr7])
