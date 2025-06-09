from copy import deepcopy

import pandas as pd
from imodels import RuleFitRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from impl.scenario.scenario_definition import ScenarioDefinition


def vectorize(solutions, mode="stats"):
    """Vectorize the solutions.

    :param solutions: Solutions to vectorize.
    :param mode: The way to encode actors. Options are :data:`stats` for encoding statistics
        or :data:`padding` for padding shorter lists of actors (default: :data:`stats`).
    :return: A tuple of vectors, target values, and corresponding feature names.
    """
    fitnesses = []
    source_scens = []
    follow_up_scens = []
    max_actors = 0
    for solution in solutions:
        fitnesses.append(solution.fitness.values[0])
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

    categorical_cols = [col for col in raw_vectors.columns
                        if any(key in col for key in ("town", "weather", "traj_direction", "model"))]
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(sparse=False), categorical_cols),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    transformed_array = preprocessor.fit_transform(raw_vectors)
    features = preprocessor.get_feature_names_out()
    vectors = pd.DataFrame(transformed_array, columns=features).fillna(-999)
    return vectors, fitnesses, features


def generate_rules(x, y, features):
    """Train the rule model.

    :param x: Vectors.
    :param y: Target values.
    :param features: Feature names.
    :return: Rule model.
    """
    model = RuleFitRegressor(max_rules=12)
    model.fit(x, y, feature_names=features)
    return model
