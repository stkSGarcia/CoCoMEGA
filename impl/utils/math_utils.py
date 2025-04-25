import math

import numpy as np


def cartesian_to_polar(x, y):
    """
    Convert Cartesian coordinates to polar coordinates.

    :param x: The x-coordinate.
    :param y: The y-coordinate.
    :return: A tuple containing the radius (r) and the angle (theta) in degrees.
    """
    r = math.sqrt(x ** 2 + y ** 2)
    theta = math.degrees(math.atan2(y, x))
    return r, theta


def polar_to_cartesian(radius, angle_degrees):
    """
    Convert polar coordinates to Cartesian coordinates.

    :param radius: The polar radius.
    :param angle_degrees: The polar angle in degrees.
    :return: A tuple (x, y) containing the Cartesian coordinates.
    """
    x = radius * math.cos(math.radians(angle_degrees))
    y = radius * math.sin(math.radians(angle_degrees))
    return x, y


def vector_norm(vec):
    """
    Calculate the norm (length) of a vector.

    :param vec: A list or array of numbers representing the vector.
    :return: The vector's norm (L2 norm).
    """
    return math.pow(sum([el ** 2 for el in vec]), 1 / len(vec))


def calculate_ds_improvements(mean_df):
    """
    Calculate and print the average improvement in DS (Distinct Solutions) between CoCoMEGA and other algorithms.

    :param mean_df: A DataFrame containing the mean DS values for different algorithms and thresholds.
    """
    comparison_df = mean_df.pivot_table(index=['fitness_threshold', 'distance_threshold'],
                                        columns='alg',
                                        values='mean_ds').reset_index()
    for alg in ('ga', 'rs', 'ccea-d'):
        if alg in comparison_df.columns:
            comparison_df[f'ccea/{alg}'] = (comparison_df['ccea'] - comparison_df[alg]) / comparison_df[alg] * 100
            print(f"Average improvement of CoCoMEGA compared to {alg}: {comparison_df[f'ccea/{alg}'].mean():.2f}%")


def area_under_curve(x, y):
    """
    Calculate the area under a curve defined by x and y using the trapezoidal rule.

    :param x: An array of x-values (independent variable).
    :param y: An array of y-values (dependent variable).
    :return: The area under the curve.
    """
    non_nan_indices = (~np.isnan(y)) & (~np.isnan(x))
    return np.trapz(y[non_nan_indices], x[non_nan_indices])


def calculate_auc_improvements(auc_df):
    """
    Calculate and print the average improvement in AUC (Area Under the Curve) for CoCoMEGA compared to other algorithms.

    :param auc_df: A DataFrame containing the AUC values for different algorithms and thresholds.
    """
    comparison_df = auc_df.pivot_table(index=['fitness_threshold', 'distance_threshold'],
                                       columns='alg',
                                       values='auc').reset_index()

    for alg in ('ga', 'rs', 'ccea-d'):
        if alg in comparison_df.columns:
            comparison_df[f'ccea/{alg}'] = (comparison_df['ccea'] - comparison_df[alg]) / comparison_df[alg] * 100
            print(f"Average improvement of CoCoMEGA compared to {alg}: {comparison_df[f'ccea/{alg}'].mean():.2f}%")
