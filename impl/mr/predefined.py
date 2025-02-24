from impl.mr.mr import Decreasing, Invariance, MR, MRSet, PerturbationFactory, Operation
from impl.scenario.scenario_definition import Boundary

__all__ = ["mr_set1", "mr_set2", "mr_set3"]

slow = Decreasing("velocity", threshold=0.2)
slow_30 = Decreasing("velocity", threshold=0.3)
steer_keep = Invariance("steer", threshold=0.1, percentage=False)

ego_speed = PerturbationFactory(("ego", "speed"), Boundary({"speed": [0.0, 35.0]}))
ego_model = PerturbationFactory(("ego", "model"), Boundary({"model": [0, 22]}))
ego_position = PerturbationFactory(("ego", "position"), Boundary({"position": [10, 30]}))

vehicle_add_left = PerturbationFactory("vehicle", Boundary.Region.LEFT, Operation.ADD, mark=True)
vehicle_add_focus = PerturbationFactory("vehicle", Boundary.Region.FOCUS, Operation.ADD, mark=True)
vehicle_add_right = PerturbationFactory("vehicle", Boundary.Region.RIGHT, Operation.ADD, mark=True)
vehicle_change_size = PerturbationFactory(("vehicle", "model"), Boundary({"model": [0, 22]}))
vehicle_change_speed = PerturbationFactory(("vehicle", "speed"), Boundary({"speed": [0.0, 35.0]}))

walker_add_left = PerturbationFactory("walker", Boundary.Region.LEFT, Operation.ADD, mark=True)
walker_add_focus = PerturbationFactory("walker", Boundary.Region.FOCUS, Operation.ADD, mark=True)
walker_add_right = PerturbationFactory("walker", Boundary.Region.RIGHT, Operation.ADD, mark=True)
walker_change_size = PerturbationFactory(("walker", "model"), Boundary({"model": [0, 14]}))
walker_change_speed = PerturbationFactory(("walker", "speed"), Boundary({"speed": [0.0, 4.0]}))

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
mr8 = MR([ego_speed], steer_keep)
mr9 = MR([ego_model], steer_keep)
mr10 = MR([vehicle_change_size, walker_change_size], steer_keep)
mr11 = MR([ego_position], steer_keep)
mr12 = MR([vehicle_change_speed, walker_change_speed], steer_keep)
mr13 = MR([vehicle_add_left, vehicle_add_focus, vehicle_add_right,
           walker_add_left, walker_add_focus, walker_add_right], steer_keep)

mr_set1 = MRSet([mr1, mr2, mr3, mr4, mr5])
mr_set2 = MRSet([mr6, mr7])
mr_set3 = MRSet([mr8, mr9, mr10, mr11, mr12, mr13])
