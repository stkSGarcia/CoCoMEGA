import importlib
import math
from collections.abc import Sequence

from impl import config as cfg


class TrajectorySolver:

    @staticmethod
    def solve(traj1, traj2):
        """
        Check whether two trajectories intersect.

        :param traj1: First trajectory (list of points).
        :param traj2: Second trajectory (list of points).
        :return: True if any segments intersect, else False.
        """
        return TrajectorySolver._check_trajectory_collision(traj1, traj2)

    @staticmethod
    def _check_trajectory_collision(traj1, traj2):
        """
        Iterate through trajectory segments to detect collisions.

        :param traj1: First trajectory.
        :param traj2: Second trajectory.
        :return: True if any segment pairs intersect.
        """
        for i in range(len(traj1) - 1):
            for j in range(len(traj2) - 1):
                if TrajectorySolver._check_intersect(traj1[i], traj1[i + 1], traj2[j], traj2[j + 1]):
                    return True
        return False

    @staticmethod
    def _check_intersect(p1, p2, q1, q2):
        """
        Determine if two line segments (p1-p2 and q1-q2) intersect.

        :param p1: Start of first segment.
        :param p2: End of first segment.
        :param q1: Start of second segment.
        :param q2: End of second segment.
        :return: True if they intersect.
        """

        # Find the 4 orientations needed for the general and special cases
        o1 = TrajectorySolver._orientation(p1, p2, q1)
        o2 = TrajectorySolver._orientation(p1, p2, q2)
        o3 = TrajectorySolver._orientation(q1, q2, p1)
        o4 = TrajectorySolver._orientation(q1, q2, p2)

        # General case
        if o1 != o2 and o3 != o4:
            return True

        # Special Cases
        # p1, p2, q1 are collinear and q1 lies on segment p1p2
        if o1 == 0 and TrajectorySolver._on_segment(p1, q1, p2):
            return True

        # p1, p2, q2 are collinear and q2 lies on segment p1p2
        if o2 == 0 and TrajectorySolver._on_segment(p1, q2, p2):
            return True

        # q1, q2, p1 are collinear and p1 lies on segment q1q2
        if o3 == 0 and TrajectorySolver._on_segment(q1, p1, q2):
            return True

        # q1, q2, p2 are collinear and p2 lies on segment q1q2
        if o4 == 0 and TrajectorySolver._on_segment(q1, p2, q2):
            return True

        return False

    @staticmethod
    def _orientation(p, q, r):
        """
        Compute orientation of three ordered points (p, q, r).

        :return: 0 if collinear, 1 if clockwise, 2 if counterclockwise.
        """
        val = (float(q["y"] - p["y"]) * (r["x"] - q["x"])) - (float(q["x"] - p["x"]) * (r["y"] - q["y"]))
        if val > 0:
            return 1
        elif val < 0:
            return 2
        else:
            return 0

    @staticmethod
    def _on_segment(p, q, r):
        """
        Check if point q lies on segment pr.

        :return: True if on segment.
        """
        if min(p["x"], r["x"]) <= q["x"] <= max(p["x"], r["x"]) and min(p["y"], r["y"]) <= q["y"] <= max(p["y"],
                                                                                                         r["y"]):
            return True
        return False


def trajectory_score(pert, pop_scen):
    """
    Calculate average trajectory alignment score for perturbed actor trajectories.

    :param pert: List of perturbation objects (with actor configs).
    :param pop_scen: Population of scenarios.
    :return: Average score over all scenarios.
    """
    total_score = 0
    for p_element in pert:
        module = importlib.import_module("impl.scenario.scenario_definition")
        if (p_element.category in module.ScenarioDefinition.ATTRIBUTES or
                isinstance(p_element.category, Sequence)): continue
        for scenario in pop_scen:
            pert_route = scenario.build_actor_trajectory(p_element.value)
            total_score += single_trajectory_score([t[0] for t in scenario.trajectory["route"]], pert_route)
    return total_score / len(pop_scen)


def single_trajectory_score(scen_route, pert_route):
    """
    Score a single trajectory based on how closely it follows the ego path.

    :param scen_route: Waypoints along the ego trajectory.
    :param pert_route: Waypoints of the perturbed actor trajectory.
    :return: An integer score.
    """
    score = 0
    for i in range(len(scen_route)):
        if i < len(scen_route) - 1:
            forward_vector = distance_vector(scen_route[i + 1], scen_route[i])
        else:
            forward_vector = distance_vector(scen_route[i], scen_route[i - 1])
        if vector_size(forward_vector) == 0: continue
        for pp in pert_route:
            vector_to_actor = distance_vector(pp, scen_route[i])
            angle = angle_between_vectors(forward_vector, vector_to_actor)
            if ((-30 <= angle <= 30) or (330 <= angle <= 390)) and vector_size(vector_to_actor) <= \
                    cfg.CONFIG["violation"]["max_ego_distance"]:
                score += 1
    return score


def distance_vector(v1, v2):
    """
    Compute vector difference between v1 and v2.

    :param v1: Dict with 'x', 'y'.
    :param v2: Dict with 'x', 'y'.
    :return: Vector difference as a dictionary.
    """
    return {'x': v1['x'] - v2['x'], 'y': v1['y'] - v2['y']}


def angle_between_vectors(v1, v2):
    """
    Compute angle in degrees between two 2D vectors.

    :param v1: First vector.
    :param v2: Second vector.
    :return: Angle in degrees.
    """
    dot_product = v1['x'] * v2['x'] + v1['y'] * v2['y']
    magnitude_v1 = math.sqrt(v1['x'] ** 2 + v1['y'] ** 2)
    magnitude_v2 = math.sqrt(v2['x'] ** 2 + v2['y'] ** 2)
    if magnitude_v1 * magnitude_v2 == 0:
        return 0
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    cos_angle = clip(cos_angle, -1, 1)
    angle = math.acos(cos_angle)

    return math.degrees(angle)


def vector_size(v):
    """
    Compute the Euclidean norm of a vector.

    :param v: A 2D vector.
    :return: Magnitude of the vector.
    """
    return math.sqrt(v['x'] ** 2 + v['y'] ** 2)


def rotate_vector(v, degree):
    """
    Rotate a 2D vector by a specified angle.

    :param v: Vector to rotate.
    :param degree: Angle in degrees.
    :return: Rotated vector as a dictionary.
    """
    theta = math.radians(degree)
    return {'x': v['x'] * math.cos(theta) - v['y'] * math.sin(theta),
            'y': v['x'] * math.sin(theta) + v['y'] * math.cos(theta)}


def clip(value, _min, _max):
    """
    Clamp a value between a min and max.

    :param value: Input value.
    :param _min: Minimum allowed value.
    :param _max: Maximum allowed value.
    :return: Clamped value.
    """
    if value < _min:
        value = _min
    elif value > _max:
        value = _max
    return value
