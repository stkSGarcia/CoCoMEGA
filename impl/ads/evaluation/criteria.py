import numpy as np
from srunner.scenariomanager.scenarioatomics.atomic_criteria import Criterion
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
import carla
import py_trees
import csv
import math

from impl import config as cfg
from impl.ads.utils.carla_utils import distance
from impl.ads.utils.math_utils import angle_between_vectors


class VehicleMeasurementTest(Criterion):
    """Atomic test to record control and positional data of a vehicle during simulation."""

    def __init__(self, actor, other_actors, measures, measurement_interval, scenario_def_id, optional=False,
                 name="VehicleMeasurement"):
        """Initialize the test with ego actor, measures, and output options.

        :param actor: The main actor (vehicle) to monitor.
        :param other_actors: List of other actors to consider for distance measurements.
        :param measures: List of control or state attributes to log.
        :param measurement_interval: How often to log data (in ticks).
        :param scenario_def_id: Unique ID for scenario (used for CSV output).
        :param optional: Whether this test affects overall pass/fail.
        :param name: Name for the criterion.
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
        """Update test status and record data at configured intervals.

        :return: Always returns :class:`py_trees.common.Status.RUNNING`.
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
            if 'position' in self.measures:
                position = self.actor.get_transform().location
                measure_dict['position_x'] = position.x
                measure_dict['position_y'] = position.y

            if 'velocity' in self.measures:
                velocity = self.actor.get_velocity()
                measure_dict['velocity_x'] = velocity.x
                measure_dict['velocity_y'] = velocity.y

            if len(self.other_actors) > 0:
                measure_dict[f"ego-nearest-distance"] = min(
                    [distance(self.actor, other_actor) for other_actor in self.other_actors])

                fov_distances = [distance(self.actor, other_actor) \
                                 for other_actor in self.other_actors if self._isin_fov(other_actor)]
                measure_dict[f"fov-nearest-distance"] = min(fov_distances) if len(fov_distances) > 0 else np.inf
                role_names = set([other_actor.attributes['role_name'] for other_actor in self.other_actors])
                for role_name in role_names:
                    measure_dict[f"ego-nearest-distance-{role_name}"] = min(
                        [distance(self.actor, other_actor) for other_actor in self.other_actors if
                         other_actor.attributes['role_name'] == role_name])

                    fov_rolename_distances = [distance(self.actor, other_actor) \
                                              for other_actor in self.other_actors \
                                              if other_actor.attributes['role_name'] == role_name \
                                              and self._isin_fov(other_actor)]
                    measure_dict[f"fov-nearest-distance-{role_name}"] = min(fov_rolename_distances) \
                        if len(fov_rolename_distances) > 0 else np.inf

            self.values.append(measure_dict)

        return new_status

    def terminate(self, new_status):
        """Terminate the test and save collected data to a CSV file."""
        if len(self.values) > 0:
            self._write_to_file()
        # raise EarlyTerminationException("Scenario has terminated right after start.")
        super().terminate(new_status)

    def _write_to_file(self):
        """Write recorded measurements to a CSV file in the ``results`` directory."""
        keys = self.values[0].keys()
        with (cfg.CONFIG["workspace"]["sim_result"] / f"{self.scenario_def_id}.csv").open("w", newline="") as f:
            dict_writer = csv.DictWriter(f, keys)
            dict_writer.writeheader()
            dict_writer.writerows(self.values)

    def _isin_fov(self, actor):
        """Check whether an actor is within the ego vehicle's field of view (FOV).

        :param actor: The actor.
        :return: True if within FOV, False otherwise.
        """
        ego_transform = self.actor.get_transform()
        ego_location = ego_transform.location
        forward_vector = ego_transform.rotation.get_forward_vector()
        actor_location = actor.get_location()

        vector_to_actor = carla.Vector2D(actor_location.x - ego_location.x,
                                         actor_location.y - ego_location.y, )
        angle = angle_between_vectors(forward_vector, vector_to_actor)

        return angle <= self.fov / 2
