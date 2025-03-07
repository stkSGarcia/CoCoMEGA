import math
import numpy as np
import pandas as pd


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

def calculate_ds_improvements(mean_df):
    comparison_df = mean_df.pivot_table(index=['fitness_threshold', 'distance_threshold'],
                                       columns='alg',
                                       values='mean_ds').reset_index()

    comparison_df['ccea/ga'] = (comparison_df['ccea'] - comparison_df['ga']) / comparison_df['ga'] * 100
    comparison_df['ccea/rs'] = (comparison_df['ccea'] - comparison_df['rs']) / comparison_df['rs'] * 100

    print(f"Average improvement of CoCoMEGA compared to SGA: {comparison_df['ccea/ga'].mean():.2f}%")
    print(f"Average improvement of CoCoMEGA compared to RS: {comparison_df['ccea/rs'].mean():.2f}%")


def area_under_curve(x, y):
    non_nan_indices = (~np.isnan(y)) & (~np.isnan(x))
    return np.trapz(y[non_nan_indices], x[non_nan_indices])


def calculate_auc_improvements(auc_df):
    comparison_df = auc_df.pivot_table(index=['fitness_threshold', 'distance_threshold'],
                 columns='alg',
                 values='auc').reset_index()

    comparison_df['ccea/ga'] = (comparison_df['ccea'] - comparison_df['ga']) / comparison_df['ga'] * 100
    comparison_df['ccea/rs'] = (comparison_df['ccea'] - comparison_df['rs']) / comparison_df['rs'] * 100

    print(f"Average improvement of CoCoMEGA compared to SGA: {comparison_df['ccea/ga'].mean():.2f}%")
    print(f"Average improvement of CoCoMEGA compared to RS: {comparison_df['ccea/rs'].mean():.2f}%")


