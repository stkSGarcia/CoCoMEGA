import importlib
from collections.abc import Sequence

from impl import config as cfg
from impl.ads.utils.math_utils import vector_size, angle_between_vectors_dict, distance_vector


class TrajectorySolver:

    @staticmethod
    def solve(traj1, traj2):
        """
        Check whether two trajectories intersect.

        :param traj1: First trajectory (list of points).
        :param traj2: Second trajectory (list of points).
        :return: :data:`True` if any segments intersect, else :data:`False`.
        """
        return TrajectorySolver._check_trajectory_collision(traj1, traj2)

    @staticmethod
    def _check_trajectory_collision(traj1, traj2):
        """
        Iterate through trajectory segments to detect collisions.

        :param traj1: First trajectory.
        :param traj2: Second trajectory.
        :return: :data:`True` if any segment pairs intersect.
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
        :return: :data:`True` if they intersect.
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

        :return: :data:`0` if collinear, :data:`1` if clockwise, :data:`2` if counterclockwise.
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

        :return: :data:`True` if on segment.
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
            angle = angle_between_vectors_dict(forward_vector, vector_to_actor)
            if ((-30 <= angle <= 30) or (330 <= angle <= 390)) and vector_size(vector_to_actor) <= \
                    cfg.CONFIG["violation"]["max_ego_distance"]:
                score += 1
    return score
