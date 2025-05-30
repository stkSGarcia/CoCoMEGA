from copy import deepcopy

import pandas as pd
from imodels import RuleFitRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from impl.scenario.scenario_definition import ScenarioDefinition


def vectorize(solutions, encode_stats=False):
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
        if not encode_stats:
            for attr in ScenarioDefinition.DYNAMIC:
                source_actors = getattr(source, f"{attr}s")
                follow_up_actors = getattr(follow_up, f"{attr}s")
                max_actors = max(max_actors, len(source_actors), len(follow_up_actors))

    if encode_stats:
        vector_dfs = [pd.concat([
            source.vectorize(max_actors, prefix="source", encode_stats=True),
            follow_up.vectorize(max_actors, prefix="follow_up", encode_stats=True),
        ], axis=1) for source, follow_up in zip(source_scens, follow_up_scens)]
    else:
        vector_dfs = [pd.concat([
            source.vectorize(max_actors, prefix="source"),
            follow_up.vectorize(max_actors, prefix="follow_up")
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
    rulefit = RuleFitRegressor(max_rules=10)
    rulefit.fit(x, y, feature_names=features)
    return rulefit
