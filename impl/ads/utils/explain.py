import logging
import time
from copy import deepcopy

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from imodels import RuleFitRegressor
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from impl import config as cfg
from impl.ads.scenario.scenario_definition import ScenarioDefinition

logger = logging.getLogger(__name__)

CATEGORIES = {
    "town": ["town01", "town02", "town03", "town04", "town05", "town06", "town07", "town10"],
    "weather": list(range(0, 12)),
    "traj_direction": ["forward", "left", "right"],
    "model": list(range(0, 23)),
}


def vectorize(solutions, mode="stats"):
    """Vectorize the solutions.

    :param solutions: Solutions to vectorize.
    :param mode: The way to encode actors. Options are :data:`stats` for encoding statistics
        or :data:`padding` for padding shorter lists of actors (default: :data:`stats`).
    :return: A tuple of vectors, target values (solution fitnesses, v1 fitnesses, and v2 fitnesses),
        and corresponding feature names.
    """
    fitnesses = []
    fitnesses_v1 = []
    fitnesses_v2 = []
    source_scens = []
    follow_up_scens = []
    max_actors = 0
    for solution in solutions:
        fitnesses.append(solution.fitness.values[0]
                         if solution.fitness_type == cfg.CONFIG["search"]["diff_testing"]["test"]
                         else -solution.fitness.values[0])
        fitnesses_v1.append(solution.v1.fitness[0])
        fitnesses_v2.append(solution.v2.fitness[0])
        source = solution[0]
        source_scens.append(source)
        follow_up = deepcopy(source)
        perturbation = solution[1]
        perturbation.perturb(follow_up)
        follow_up_scens.append(follow_up)
        if mode == "padding":
            for attr in ScenarioDefinition.DYNAMIC:
                source_actors = getattr(source, f"{attr}s")
                follow_up_actors = getattr(follow_up, f"{attr}s")
                max_actors = max(max_actors, len(source_actors), len(follow_up_actors))

    if mode == "padding":
        vector_dfs = [pd.concat([
            source.vectorize(max_actors, prefix="source", mode="padding"),
            follow_up.vectorize(max_actors, prefix="follow_up", mode="padding")
        ], axis=1) for source, follow_up in zip(source_scens, follow_up_scens)]
    else:
        vector_dfs = [pd.concat([
            source.vectorize(max_actors, prefix="source", mode="stats"),
            follow_up.vectorize(max_actors, prefix="follow_up", mode="stats"),
        ], axis=1) for source, follow_up in zip(source_scens, follow_up_scens)]
    raw_vectors = pd.concat(vector_dfs, ignore_index=True)
    vectors, features = _preprocess_vectors(raw_vectors)
    return vectors, fitnesses, fitnesses_v1, fitnesses_v2, features


def _preprocess_vectors(raw_vectors):
    categorical_cols, categories_list = [], []
    for col in raw_vectors.columns:
        if all(key not in col for key in CATEGORIES.keys()): continue
        categorical_cols.append(col)
        for keyword in CATEGORIES.keys():
            if keyword in col:
                categories_list.append(CATEGORIES[keyword])
                break
        else:
            categories_list.append("auto")  # Fallback: let encoder auto-detect categories for this column.

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(categories=categories_list, sparse=False), categorical_cols),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    transformed_array = preprocessor.fit_transform(raw_vectors)
    features = preprocessor.get_feature_names_out()
    vectors = pd.DataFrame(transformed_array, columns=features).fillna(-999)
    return vectors, features


def model_fit(X, y):
    """Train a tree-based model to fit the data.

    :param X: Vectors.
    :param y: Target values.
    :return: The best model.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, np.array(y), test_size=0.2, random_state=42)

    logger.info(f"\n{'=' * 50}\nRANDOMIZED SEARCH\n{'=' * 50}")
    param_dist = {
        "n_estimators": randint(50, 501),
        "max_depth": randint(3, 9),
        "learning_rate": uniform(0.01, 0.19),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_weight": randint(1, 8),
        "reg_alpha": uniform(0, 1.0),
        "reg_lambda": uniform(0.5, 1.5),
    }

    logger.info("Using RandomizedSearchCV with 100 iterations...")
    random_search = RandomizedSearchCV(
        estimator=xgb.XGBRegressor(objective="reg:squarederror", n_jobs=-1, random_state=42),
        param_distributions=param_dist,
        n_iter=100,
        scoring="neg_mean_squared_error",
        cv=5,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
        random_state=42,
    )

    logger.info("Starting randomized search...")
    random_search.fit(X_train, y_train)

    logger.info("Randomized search completed!")
    logger.info(f"Best CV RMSE: {np.sqrt(-random_search.best_score_):.6f}")
    log_str = "\nBest parameters:"
    for param, value in random_search.best_params_.items():
        log_str += f"\n\t{param:<20}: {value}"
    logger.info(log_str)

    logger.info(f"\n{'=' * 50}\nRANDOMIZED SEARCH ANALYSIS\n{'=' * 50}")
    results_df = pd.DataFrame(random_search.cv_results_)

    # Best scores.
    log_str = "\nTop 5 parameter combinations:"
    top_results = results_df.nlargest(5, "mean_test_score")[["mean_test_score", "params"]]
    for i, (idx, row) in enumerate(top_results.iterrows(), 1):
        rmse = np.sqrt(-row["mean_test_score"])
        log_str += f"\n\t{i}. RMSE: {rmse:.6f} - {row['params']}"
    logger.info(log_str)

    # Parameter impact analysis.
    log_str = "\nParameter impact on performance:"
    for param in param_dist.keys():
        param_col = f"param_{param}"
        if param_col in results_df.columns:
            param_performance = results_df.groupby(param_col)["mean_test_score"].mean()
            best_value = param_performance.idxmax()
            best_score = np.sqrt(-param_performance.max())
            log_str += f"\n\t{param:<20}: best value = {best_value:<10}, best RMSE = {best_score:.6f}"
    logger.info(log_str)

    logger.info(f"\n{'=' * 50}\nBEST MODEL EVALUATION\n{'=' * 50}")
    best_model = random_search.best_estimator_
    y_pred_best = best_model.predict(X_test)

    # Calculate metrics.
    best_mse = mean_squared_error(y_test, y_pred_best)
    best_rmse = np.sqrt(best_mse)
    best_mae = mean_absolute_error(y_test, y_pred_best)
    best_r2 = r2_score(y_test, y_pred_best)
    logger.info(
        "\nBest Model Performance:"
        f"\n\tMSE : {best_mse:.6f}"
        f"\n\tRMSE: {best_rmse:.6f}"
        f"\n\tMAE : {best_mae:.6f}"
        f"\n\tR²  : {best_r2:.6f}"
    )

    # Cross-validation with the best model.
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
    cv_rmse = np.sqrt(-cv_scores.mean())
    cv_std = np.sqrt(cv_scores.std())
    logger.info(f"Cross-validation RMSE: {cv_rmse:.6f} (+/- {cv_std:.6f})")

    # Save the best model.
    result_path = cfg.CONFIG["workspace"]["exp_result"] / str(int(round(time.time() * 1000)))
    result_path.mkdir(exist_ok=True)
    model_filename = result_path / "best_model.pkl"
    joblib.dump(best_model, model_filename)
    logger.info(f"✓ Best model saved as '{model_filename}'.")

    # Save grid search results for analysis.
    result_filename = result_path / "search_results.pkl"
    joblib.dump(random_search, result_filename)
    logger.info(f"✓ Randomized search results saved as '{result_filename}'.")

    logger.info(f"\n{'=' * 50}\nMODEL VISUALIZATION\n{'=' * 50}")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Actual vs. Predicted
    axes[0, 0].scatter(y_test, y_pred_best, alpha=0.6, color="b")
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    axes[0, 0].set_xlabel("Actual Values")
    axes[0, 0].set_ylabel("Predicted Values")
    axes[0, 0].set_title("Actual vs Predicted")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].text(0.05, 0.95, f"R² = {best_r2:.4f}", transform=axes[0, 0].transAxes,
                    bbox=dict(boxstyle="round", facecolor="w", alpha=0.8), verticalalignment="top")

    # 2. Residuals Plot
    residuals = y_test - y_pred_best
    axes[0, 1].scatter(y_pred_best, residuals, alpha=0.6, color="g")
    axes[0, 1].axhline(y=0, color="r", linestyle="--")
    axes[0, 1].set_xlabel("Predicted Values")
    axes[0, 1].set_ylabel("Residuals")
    axes[0, 1].set_title("Residual Plot")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Feature Importance
    feature_importance = best_model.feature_importances_
    importance_df = pd.DataFrame({
        "feature_name": X_test.columns,
        "importance": feature_importance
    }).sort_values("importance", ascending=False)
    top_features = importance_df.head(10)
    axes[1, 0].barh(range(len(top_features)), top_features["importance"], color="orange")
    axes[1, 0].set_yticks(range(len(top_features)))
    axes[1, 0].set_yticklabels([name for name in top_features["feature_name"]])
    axes[1, 0].set_xlabel("Importance")
    axes[1, 0].set_title("Top 10 Feature Importances")
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Residuals Distribution
    axes[1, 1].hist(residuals, bins=30, alpha=0.7, color="c", edgecolor="k")
    axes[1, 1].axvline(x=0, color="r", linestyle="--", linewidth=2)
    axes[1, 1].axvline(x=residuals.mean(), color="orange", linestyle="-", linewidth=2,
                       label=f"Mean: {residuals.mean():.4f}")
    axes[1, 1].set_xlabel("Residuals")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Distribution of Residuals")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    stats_text = f"Std: {residuals.std():.4f}\nSkew: {stats.skew(residuals):.4f}"
    axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                    bbox=dict(boxstyle="round", facecolor="w", alpha=0.8), verticalalignment="top")

    plt.tight_layout()
    plt.show()

    return best_model


def generate_rules(X, y, features, base_model=None, max_rules=10):
    """Train the rule model.

    :param X: Vectors.
    :param y: Target values.
    :param features: Feature names.
    :param base_model: Tree-based model.
    :param max_rules: Maximum number of rules to generate.
    :return: Rule model.
    """
    model = RuleFitRegressor(max_rules=max_rules, tree_generator=base_model, exp_rand_tree_size=False) \
        if base_model else RuleFitRegressor(max_rules=max_rules)
    model.fit(X, y, feature_names=features)
    return model


def explain(model, X):
    """Explain the trained model and vectors using SHAP.

    :param model: The trained model.
    :param X: The data.
    :return: :class:`Explanation` object.
    """
    explainer = shap.TreeExplainer(model)
    return explainer(X)


def visualize_explanation(explanation, dependence_plot=False):
    """Visualize the explanation.

    :param explanation: :class:`Explanation` object to visualize.
    :param dependence_plot: Whether to show the dependence plots.
    """
    shap.plots.initjs()
    plt.figure()
    shap.plots.beeswarm(explanation, max_display=20, show=False)
    plt.savefig(cfg.CONFIG["workspace"]["visualization"] / "summary.png", bbox_inches="tight")
    plt.figure()
    shap.plots.bar(explanation, max_display=20, show=False)
    plt.savefig(cfg.CONFIG["workspace"]["visualization"] / "importance.png", bbox_inches="tight")

    if dependence_plot:
        top_inds = np.argsort(-np.sum(np.abs(explanation.values), 0))
        for i in range(10):
            shap.plots.scatter(explanation[:, top_inds[i]], color=explanation)


def scoring(model_v1, model_v2, X, clip=2.0, alpha=0.5, epsilon=1e-6):
    """Scoring the updated model.

    :param model_v1: Original model.
    :param model_v2: Updated model.
    :param X: The data.
    :param clip: Sets the symmetric clipping bound for the adjusted ratio, ensuring it never drops below 1/x or exceeds x.
    :param alpha: Controls the strength of the standard deviation adjustment by raising the raw ratio to this power.
        Values closer to zero weaken the effect.
    :param epsilon: A tiny positive constant added to standard deviations to prevent division by zero when forming ratios.
    :return: Score of the difference between the original and updated model, difference matrix, weight matrix.
    """
    explanation_v1 = explain(model_v1, X)
    explanation_v2 = explain(model_v2, X)
    A, B = explanation_v1.values, explanation_v2.values

    D = A - B
    abs_sum = np.abs(A) + np.abs(B)
    W = abs_sum / abs_sum.sum(axis=1, keepdims=True)

    f_raw = (np.std(A, axis=0) + epsilon) / (np.std(B, axis=0) + epsilon)
    f_decay = np.power(f_raw, alpha)
    f_clipped = np.clip(f_decay, 1 / clip, clip)
    sign_D = np.sign(D)
    G = np.power(f_clipped, sign_D)

    return np.sum(W * D), D, W


def vectorize_scenario(scenarios, rules):
    raw_vectors = pd.concat([scenario.vectorize(-1, prefix="source", mode="stats") for scenario in scenarios],
                            ignore_index=True)
    vectors, features = _preprocess_vectors(raw_vectors)

    for i, row in rules[(rules.coef != 0) & (rules.type != "linear")].iterrows():
        rule = row["rule"]
        vectors[rule] = 0

        conditions = [cond.strip() for cond in rule.split("and")]
        filtered_conditions = []
        for cond in conditions:
            feature = cond.split()[0]
            if feature in features:
                filtered_conditions.append(cond)
        if not filtered_conditions: continue
        filtered_rule = " and ".join(filtered_conditions)

        match_idx = vectors.query(filtered_rule).index
        vectors.loc[match_idx, rule] = 1
    return vectors
