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
        val = (float(q[1] - p[1]) * (r[0] - q[0])) - (float(q[0] - p[0]) * (r[1] - q[1]))
        if val > 0:
            return 1
        elif val < 0:
            return 2
        else:
            return 0

    @staticmethod
    def _on_segment(p, q, r):
        """ Given collinear points p, q, r, check if point q lies on segment pr """
        if min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1]):
            return True
        return False
