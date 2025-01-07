import math


def cartesian_to_polar(x, y):
    """
    Convert Cartesian coordinates to polar coordinates.

    Args:
        x (float): The x-coordinate.
        y (float): The y-coordinate.

    Returns:
        tuple: A tuple containing the radius (r) and the angle (theta) in degrees.
    """
    r = math.sqrt(x ** 2 + y ** 2)
    theta = math.degrees(math.atan2(y, x))
    return r, theta


def polar_to_cartesian(radius, angle_degrees):
    """
        Convert polar coordinates to Cartesian coordinates.

        Args:
            radius (float): The polar radius.
            angle_degrees (float): The polar angle in degrees.

        Returns:
            tuple: A tuple (x, y) containing the Cartesian coordinates.
        """
    x = radius * math.cos(math.radians(angle_degrees))
    y = radius * math.sin(math.radians(angle_degrees))
    return x, y


def vector_norm(vec):
    return math.pow(sum([el ** 2 for el in vec]), 1 / len(vec))
