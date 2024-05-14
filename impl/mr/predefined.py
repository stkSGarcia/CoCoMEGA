from impl.mr.mr import Decreasing, Invariance, MR, MRSet, PerturbationFactory

__all__ = ["mr_set1"]

from impl.scenario.scenario_definition import Boundary

relation_slow = Decreasing("velocity")
relation_slow_30 = Decreasing("velocity", threshold=0.3)
relation_steer = Invariance("steering")

vehicle_pert_factory = PerturbationFactory("vehicle", Boundary({
    "x": [-198.0, -178.0],
    "y": [89.0, 102.0],
    "z": [1.0, 1.0],
    "pitch": [0.0, 0.0],
    "yaw": [-180.0, 180.0],
    "roll": [0.0, 0.0],
    "speed": [0.0, 100.0],
    "model": [0, 26],
    "color": [0, 2],
    "autopilot": [0, 1],
}))

walker_pert_factory = PerturbationFactory("walker", Boundary({
    "x": [-198.0, -178.0],
    "y": [89.0, 102.0],
    "z": [0.5, 1.0],
    "pitch": [0.0, 0.0],
    "yaw": [-180.0, 180.0],
    "roll": [0.0, 0.0],
    "speed": [0.0, 10.0],
    "model": [0, 14],
}), "walker_roadside1")

static_pert_factory = PerturbationFactory("static", Boundary({
    "x": [-198.0, -178.0],
    "y": [89.0, 102.0],
    "z": [0.5, 1.0],
    "pitch": [0.0, 0.0],
    "yaw": [-180.0, 180.0],
    "roll": [0.0, 0.0],
    "model": [69, 71],
    "size": [0, 0],
}), "static_roadside1")

mr1 = MR([walker_pert_factory], relation_slow)
mr2 = MR([static_pert_factory], relation_slow)
mr3 = MR([walker_pert_factory], relation_slow_30)
mr7 = MR([PerturbationFactory("weather", Boundary({"weather": [1, 3]}))], relation_slow)
mr8 = MR([PerturbationFactory("weather", Boundary({"weather": [0, 14]}))], relation_steer)

# mr_set1 = MRSet([mr1, mr2, mr7])
mr_set1 = MRSet([mr1])
