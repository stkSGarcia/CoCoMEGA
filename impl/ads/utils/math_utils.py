import math

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, combine_pvalues, wilcoxon


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


def mannwhitneyu_fisher(stat_data, alg_pairs, value_col):
    """
    Perform pairwise Mann-Whitney U tests between algorithms for each combination of fitness and distance thresholds,
    and aggregate the p-values using Fisher's method.
    :param stat_data: A dictionary where keys are algorithm names and values are DataFrames containing statistics.
    :param alg_pairs: A list of tuples, each containing a pair of algorithm names to compare.
    :param value_col: The name of the column in the DataFrames to use for the statistical test.
    :return: A list of tuples, each containing the test statistic and combined p-value for each algorithm pair.
    """
    algorithms = set([alg for alg_pair in alg_pairs for alg in alg_pair])
    threshs_df = next(iter(stat_data.values()))[["fitness_threshold", "distance_threshold"]].drop_duplicates()
    records = []
    for (f_thresh, d_thresh) in list(zip(threshs_df["fitness_threshold"], threshs_df["distance_threshold"])):
        alg_data = {
            a: stat_data[a].query(f"fitness_threshold == {f_thresh} & distance_threshold == {d_thresh}")[
                value_col].values
            for a in algorithms
        }
        # Pairwise tests
        for (a1, a2) in alg_pairs:
            data1 = alg_data[a1]
            data2 = alg_data[a2]

            # stat, pval = mannwhitneyu(data1, data2, alternative='greater')
            greater = np.mean(data1) > np.mean(data2)
            stat, pval = mannwhitneyu(data1, data2, alternative='greater' if greater else 'less')
            # stat, pval = ttest_ind(data1, data2, equal_var=False)
            records.append({
                "fitness_threshold": f_thresh,
                "distance_threshold": d_thresh,
                "alg1": a1,
                "alg2": a2,
                "p_value": pval
            })

    individual_results = pd.DataFrame(records)

    # Fisher's method for aggregating multiple tests
    test_results = []
    for (a1, a2) in alg_pairs:
        statistic, combined_p = combine_pvalues(individual_results.query(f"alg1 == '{a1}' & alg2 == '{a2}'")["p_value"],
                                                method='fisher')
        test_results.append((statistic, combined_p))

    return test_results


def paired_wilcoxon(stat_df, alg_pairs):
    """
    Perform paired Wilcoxon signed-rank tests between pairs of algorithms.
    :param stat_df: A DataFrame where each column corresponds to an algorithm and contains the values to compare.
    :param alg_pairs: A list of tuples, each containing a pair of algorithm names to compare.
    :return: A list of tuples, each containing the test statistic and one-sided p-value for each algorithm pair.
    """
    results = []
    for (a1, a2) in alg_pairs:
        data1 = stat_df[a1]
        data2 = stat_df[a2]

        stat, pval = wilcoxon(data1, data2)

        differences = [x - y for x, y in zip(data1, data2)]
        if sum(d > 0 for d in differences) > sum(d < 0 for d in differences):
            pval_one_sided = pval / 2
        else:
            pval_one_sided = 1 - pval / 2

        results.append((stat, pval_one_sided))

    return results


def calculate_ds_improvements(data, mean_df, comparison_algs):
    """
    Calculate and print the average improvement in DS (Distinct Solutions) between algorithms.
    :param data: A dictionary where keys are algorithm names and values are DataFrames containing statistics.
    :param mean_df: A DataFrame containing the mean DS values for different algorithms and thresholds.
    :param comparison_algs: A dictionary with keys 'algs' and 'baselines' containing lists of algorithm names to compare.
    """
    algs = comparison_algs["algs"]
    baselines = comparison_algs["baselines"]

    alg_pairs = [(alg, baseline) for alg in algs for baseline in baselines]

    comparison_df = mean_df.pivot_table(index=["fitness_threshold", "distance_threshold"],
                                        columns="alg",
                                        values="mean_ds").reset_index()

    significance = mannwhitneyu_fisher(data, alg_pairs=alg_pairs, value_col="distinct_solution_num")
    p_values = [significance[i][1] for i in range(len(significance))]

    for pairs, p_val in zip(alg_pairs, p_values):
        alg, baseline = pairs[0], pairs[1]
        comparison_df[f'{alg}/{baseline}'] = (comparison_df[alg] - comparison_df[baseline]) / comparison_df[
            baseline] * 100
        print(
            f"Average improvement of {alg} compared to {baseline}: {comparison_df[f'{alg}/{baseline}'].mean():.2f}% (p-value = {p_val:.1e})")


def area_under_curve(x, y):
    """
    Calculate the area under a curve defined by x and y using the trapezoidal rule.

    :param x: An array of x-values (independent variable).
    :param y: An array of y-values (dependent variable).
    :return: The area under the curve.
    """
    non_nan_indices = (~np.isnan(y)) & (~np.isnan(x))
    return np.trapz(y[non_nan_indices], x[non_nan_indices])


def calculate_auc_improvements(auc_df, comparison_algs):
    """
    Calculate and print the average improvement in AUC (Area Under the Curve) for different algorithms.

    :param auc_df: A `DataFrame` containing the AUC values for different algorithms and thresholds.
    :param comparison_algs: A dictionary with keys 'algs' and 'baselines containing lists of algorithm names to compare.
    """

    algs = comparison_algs["algs"]
    baselines = comparison_algs["baselines"]
    alg_pairs = [(alg, baseline) for alg in algs for baseline in baselines]

    comparison_df = auc_df.pivot_table(index=["fitness_threshold", "distance_threshold"],
                                       columns="alg",
                                       values="auc").reset_index()

    significance = paired_wilcoxon(comparison_df, alg_pairs=alg_pairs)
    p_values = [significance[i][1] for i in range(len(significance))]

    for (alg, baseline), p_val in zip(alg_pairs, p_values):
        comparison_df[f"{alg}/{baseline}"] = (comparison_df[alg] - comparison_df[baseline]) / comparison_df[
            baseline] * 100
        print(
            f"Average improvement of {alg} compared to {baseline}: {comparison_df[f'{alg}/{baseline}'].mean():.2f}% (p-value = {p_val:.1e})")


def calculate_improvements(data, comparison_algs, metric_name, higher_is_better=True):
    """
    Calculate and print the average improvement in a specified metric between algorithms.
    :param data: A dictionary where keys are algorithm names and values are DataFrames containing statistics.
    :param comparison_algs: A dictionary with keys 'algs' and 'baselines containing lists of algorithm names to compare.
    :param metric_name: The name of the metric to compare (e.g., 'fitness', 'AED').
    :param higher_is_better: A boolean indicating whether higher values of the metric are better.
    """

    algs = comparison_algs["algs"]
    baselines = comparison_algs["baselines"]

    alg_pairs = [(alg, baseline) for alg in algs for baseline in baselines]

    for alg, baseline in alg_pairs:
        stat, pval = mannwhitneyu(data[alg], data[baseline], alternative="greater" if higher_is_better else "less")
        improvement = (np.mean(data[alg]) - np.mean(data[baseline])) / np.mean(data[baseline]) * 100
        if not higher_is_better:
            improvement = -improvement
        print(
            f"Average {metric_name} improvement of {alg} ({np.mean(data[alg]):.2f}) compared to {baseline} ({np.mean(data[baseline]):.2f}): {improvement:.2f}% (p-value = {pval:.1e})")
