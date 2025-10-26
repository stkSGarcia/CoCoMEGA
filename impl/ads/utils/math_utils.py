import math
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

    :param auc_df: A :class:`DataFrame` containing the AUC values for different algorithms and thresholds.
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


def lvd_from_embeddings(X, eps=1e-8):
    """X: (k,d) selected embeddings (not necessarily unit)."""
    Xn = l2_normalize_rows(X)
    K = Xn @ Xn.T
    # Cholesky for stability; det(K+epsI) = prod(diag(L))^2
    L = np.linalg.cholesky(K + eps * np.eye(K.shape[0]))
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    k = X.shape[0]
    return float(np.exp(logdet / k))  # LVD in (0,1]


def closest_pair_angle_deg(X):
    Xn = l2_normalize_rows(X)
    K = Xn @ Xn.T
    np.fill_diagonal(K, -np.inf)  # ignore self
    max_cos = np.max(K)  # closest pair = largest cosine
    max_cos = np.clip(max_cos, -1.0, 1.0)
    return float(np.degrees(np.arccos(max_cos)))


import numpy as np


def l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Row-wise L2 normalize the entire vector.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    eps : float
        Small value to avoid division by zero.

    Returns
    -------
    np.ndarray
        Row-normalized matrix; all-zero rows remain zero.
    """
    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    mask = norms > eps
    Xn = X.copy()
    if np.any(mask):
        Xn[mask[:, 0]] = X[mask[:, 0]] / norms[mask].reshape(-1, 1)
    return Xn


def block_normalize_rows(X: np.ndarray, block_sizes, eps: float = 1e-12) -> np.ndarray:
    """
    First L2-normalize each contiguous block per row, then L2-normalize the whole row.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    block_sizes : list[int]
        Sizes of contiguous feature blocks; must sum to n_features.
    eps : float

    Returns
    -------
    np.ndarray
        Matrix with per-block normalization followed by full-row L2 normalization.
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    if sum(block_sizes) != d:
        raise ValueError(f"block_sizes must sum to {d}, got {sum(block_sizes)}")

    Xb = X.copy()
    start = 0
    for size in block_sizes:
        end = start + size
        if size > 0:
            B = Xb[:, start:end]
            norms = np.linalg.norm(B, axis=1, keepdims=True)
            mask = norms > eps
            if np.any(mask):
                Xb[mask[:, 0], start:end] = B[mask[:, 0]] / norms[mask].reshape(-1, 1)
        start = end

    return l2_normalize_rows(Xb, eps=eps)


def d2_to_angle_deg(d2):
    d2 = float(np.clip(d2, 0.0, 1.0))
    return np.degrees(np.arcsin(np.sqrt(d2)))


def as_unit_matrix(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=np.float32)
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return M / norms


def solve_lower(RT: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve (R^T) w = b, where R is upper-triangular → R^T is lower-triangular."""
    L = RT  # lower-triangular
    b = b.astype(np.float32, copy=False)
    n = L.shape[0]
    w = np.empty_like(b)
    for i in range(n):
        s = b[i]
        if i > 0:
            s -= np.dot(L[i, :i], w[:i])
        w[i] = s / (L[i, i] if L[i, i] != 0 else 1e-12)
    return w


def angle_between_vectors(v1, v2):
    """Calculate angle between two 2D vectors in degrees.

    :param v1: First vector.
    :param v2: Second vector.
    :return: Angle in degrees.
    """
    dot_product = v1.x * v2.x + v1.y * v2.y
    magnitude_v1 = math.sqrt(v1.x ** 2 + v1.y ** 2)
    magnitude_v2 = math.sqrt(v2.x ** 2 + v2.y ** 2)
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    angle = math.acos(cos_angle)
    return math.degrees(angle)


def distance_vector(v1, v2):
    """
    Compute vector difference between v1 and v2.

    :param v1: Dict with ``x``, ``y``.
    :param v2: Dict with ``x``, ``y``.
    :return: Vector difference as a dictionary.
    """
    return {'x': v1['x'] - v2['x'], 'y': v1['y'] - v2['y']}


def angle_between_vectors_dict(v1, v2):
    """
    Compute angle in degrees between two 2D vectors (dictionary version).

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
