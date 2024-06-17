import numpy as np
from srunner.scenariomanager.scenarioatomics.atomic_criteria import Criterion
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
import carla
import py_trees
import csv
import os
import math

from impl.config import CONFIG
from impl.scenario.exceptions import EarlyTerminationException


def _distance(actor1, actor2):
    loc1 = actor1.get_location()
    loc2 = actor2.get_location()
    return math.sqrt((loc1.x - loc2.x) ** 2 + (loc1.y - loc2.y) ** 2)


def _angle_between_vectors(v1, v2):
    dot_product = v1.x * v2.x + v1.y * v2.y
    magnitude_v1 = math.sqrt(v1.x ** 2 + v1.y ** 2)
    magnitude_v2 = math.sqrt(v2.x ** 2 + v2.y ** 2)
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    angle = math.acos(cos_angle)
    return math.degrees(angle)


class VehicleMeasurementTest(Criterion):
    """
    This class contains an atomic test for Measuring Vehicle Behavior.

    Important parameters:
    - actor: CARLA actor to be used for this test
    - measures: List of measures to measure. subset of [throttle, brake, steer, hand_brake, manual_gear_shift, reverse]
    - measurement_interval: This variable determines the interval, in ticks, at which measurements are taken
    - optional [optional]: If True, the result is not considered for an overall pass/fail result
    """

    def __init__(self, actor, other_actors, measures, measurement_interval, scenario_def_id, optional=False,
                 name="VehicleMeasurement"):
        """
        Setup actor and measures
        """

        self.other_actors = other_actors
        self.measures = measures
        self.measurement_interval = measurement_interval
        self.scenario_def_id = scenario_def_id
        self.values = []
        self.ticks = 0
        self.fov = 100
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
            measure_dict = {
                'tick': self.ticks,
                **{measure: getattr(control, measure, None) for measure in self.measures if
                   hasattr(control, measure)},
            }
            if 'velocity' in self.measures:
                velocity = self.actor.get_velocity()
                measure_dict['velocity_x'] = velocity.x
                measure_dict['velocity_y'] = velocity.y

            if len(self.other_actors) > 0:
                measure_dict[f"ego-nearest-distance"] = min(
                    [_distance(self.actor, other_actor) for other_actor in self.other_actors])

                fov_distances = [_distance(self.actor, other_actor) \
                                 for other_actor in self.other_actors if self._isin_fov(other_actor)]
                measure_dict[f"fov-nearest-distance"] = min(fov_distances) if len(fov_distances) > 0 else np.inf
                role_names = set([other_actor.attributes['role_name'] for other_actor in self.other_actors])
                for role_name in role_names:
                    measure_dict[f"ego-nearest-distance-{role_name}"] = min(
                        [_distance(self.actor, other_actor) for other_actor in self.other_actors if
                         other_actor.attributes['role_name'] == role_name])

                    fov_rolename_distances = [_distance(self.actor, other_actor) \
                                              for other_actor in self.other_actors \
                                              if other_actor.attributes['role_name'] == role_name \
                                              and self._isin_fov(other_actor)]
                    measure_dict[f"fov-nearest-distance-{role_name}"] = min(fov_rolename_distances) \
                        if len(fov_rolename_distances) > 0 else np.inf

            self.values.append(measure_dict)

        return new_status

    def terminate(self, new_status):
        if len(self.values) > 0:
            self._write_to_file()
        # raise EarlyTerminationException("Scenario has terminated right after start.")
        super().terminate(new_status)

    def _write_to_file(self):
        keys = self.values[0].keys()
        with open(os.path.join(CONFIG["workspace"]["sim_result"], f"{self.scenario_def_id}.csv"),
                  'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(self.values)

    def _isin_fov(self, other_actor):
        ego_transform = self.actor.get_transform()
        ego_location = ego_transform.location
        forward_vector = ego_transform.rotation.get_forward_vector()
        actor_location = other_actor.get_location()

        vector_to_actor = carla.Vector2D(actor_location.x - ego_location.x,
                                         actor_location.y - ego_location.y, )
        angle = _angle_between_vectors(forward_vector, vector_to_actor)

        return angle <= self.fov / 2
