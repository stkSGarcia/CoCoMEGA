from srunner.scenariomanager.scenarioatomics.atomic_criteria import Criterion
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
import py_trees
import csv


class VehicleMeasurementTest(Criterion):
    """
    This class contains an atomic test for Measuring Vehicle Behavior.

    Important parameters:
    - actor: CARLA actor to be used for this test
    - measures: List of measures to measure. subset of [throttle, brake, steer, hand_brake, manual_gear_shift, reverse]
    - measurement_interval: This variable determines the interval, in ticks, at which measurements are taken
    - optional [optional]: If True, the result is not considered for an overall pass/fail result
    """

    def __init__(self, actor, measures, measurement_interval, scenario_def_id, optional=False,
                 name="VehicleMeasurement"):
        """
        Setup actor and measures
        """
        self.measures = measures
        self.measurement_interval = measurement_interval
        self.scenario_def_id = scenario_def_id
        self.values = []
        self.ticks = 0
        super(VehicleMeasurementTest, self).__init__(name, actor, 1, None, optional)

    def update(self):
        """
        Check velocity
        """
        new_status = py_trees.common.Status.RUNNING

        if self.actor is None:
            return new_status

        self.ticks += 1
        if self.ticks % self.measurement_interval == 0:
            control = self.actor.get_control()
            self.values.append(
                {'tick': self.ticks, **{measure: getattr(control, measure, None) for measure in self.measures}})

        # throttle = control.throttle
        # steer = control.steer
        # brake = control.brake
        # hand_brake = control.hand_brake
        # manual_gear_shift = control.manual_gear_shift
        # reverse = control.reverse

        # self.actual_value = max(velocity, self.actual_value)
        #
        # if velocity > self.expected_value_success:
        #     self.test_status = "FAILURE"
        # else:
        #     self.test_status = "SUCCESS"
        #
        # if self._terminate_on_failure and (self.test_status == "FAILURE"):
        #     new_status = py_trees.common.Status.FAILURE
        #
        # self.logger.debug("%s.update()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

        return new_status

    def terminate(self, new_status):
        self._write_to_file()
        super().terminate(new_status)

    def _write_to_file(self):
        keys = self.values[0].keys()
        with open(f'results/{self.scenario_def_id}.csv', 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(self.values)
