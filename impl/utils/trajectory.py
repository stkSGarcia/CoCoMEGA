import importlib
import math
from collections.abc import Sequence

from impl import config


class TrajectorySolver:
    @staticmethod
    def solve(traj1, traj2):
        return TrajectorySolver._check_trajectory_collision(traj1, traj2)

    @staticmethod
    def _check_trajectory_collision(traj1, traj2):
        """ Check for collision between two trajectories. """
        for i in range(len(traj1) - 1):
            for j in range(len(traj2) - 1):
                if TrajectorySolver._check_intersect(traj1[i], traj1[i + 1], traj2[j], traj2[j + 1]):
                    return True
        return False

    @staticmethod
    def _check_intersect(p1, p2, q1, q2):
        """ Check if line segment p1-p2 and q1-q2 intersect. """

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
        """ Return the orientation of the triplet (p, q, r).
            0 -> p, q, r are collinear
            1 -> Clockwise
            2 -> Counterclockwise
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
        """ Given collinear points p, q, r, check if point q lies on segment pr """
        if min(p["x"], r["x"]) <= q["x"] <= max(p["x"], r["x"]) and min(p["y"], r["y"]) <= q["y"] <= max(p["y"],
                                                                                                         r["y"]):
            return True
        return False


def trajectory_score(pert, pop_scen):
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
                    config.CONFIG["violation"]["max_ego_distance"]:
                score += 1
    return score


def distance_vector(v1, v2):
    return {'x': v1['x'] - v2['x'], 'y': v1['y'] - v2['y']}


def angle_between_vectors(v1, v2):
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
    return math.sqrt(v['x'] ** 2 + v['y'] ** 2)


def rotate_vector(v, degree):
    theta = math.radians(degree)
    return {'x': v['x'] * math.cos(theta) - v['y'] * math.sin(theta),
            'y': v['x'] * math.sin(theta) + v['y'] * math.cos(theta)}


def clip(value, _min, _max):
    if value < _min:
        value = _min
    elif value > _max:
        value = _max
    return value
