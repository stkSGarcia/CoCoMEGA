from impl.mr.mr import Decreasing, Invariance, MR, MRSet, PerturbationFactory, Operation
from impl.scenario.scenario_definition import Boundary

__all__ = ["mr_set1", "mr_set2"]

slow = Decreasing("velocity", threshold=0.2)
slow_30 = Decreasing("velocity", threshold=0.3)
steer_keep = Invariance("steer", threshold=0.1, percentage=False)

vehicle_add_left = PerturbationFactory("vehicle", Boundary.Region.LEFT, Operation.ADD, mark=True)
vehicle_add_focus = PerturbationFactory("vehicle", Boundary.Region.FOCUS, Operation.ADD, mark=True)
vehicle_add_right = PerturbationFactory("vehicle", Boundary.Region.RIGHT, Operation.ADD, mark=True)

walker_add_left = PerturbationFactory("walker", Boundary.Region.LEFT, Operation.ADD, mark=True)
walker_add_focus = PerturbationFactory("walker", Boundary.Region.FOCUS, Operation.ADD, mark=True)
walker_add_right = PerturbationFactory("walker", Boundary.Region.RIGHT, Operation.ADD, mark=True)

weather_fog = PerturbationFactory("weather", Boundary({"weather": [7, 11]}))
weather_rain = PerturbationFactory("weather", Boundary({"weather": [4, 6]}))
brightness_night = PerturbationFactory("brightness", Boundary({"brightness": [0, 2]}))

mr1 = MR([walker_add_left, walker_add_right], slow)
mr2 = MR([brightness_night], slow)
mr3 = MR([vehicle_add_focus], slow)
mr4 = MR([walker_add_focus], slow)
mr5 = MR([weather_rain], slow)
mr6 = MR([weather_fog], steer_keep)
mr7 = MR([weather_rain], steer_keep)

mr_set1 = MRSet([mr1, mr2, mr3, mr4, mr5])
mr_set2 = MRSet([mr6, mr7])
