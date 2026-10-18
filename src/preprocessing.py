"""Fold-local preprocessing and a deliberately small classifier set."""

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, SEED


def build_preprocessor(scale_numeric=False):
    """Learn imputation/encoding inside each fit; only linear models need scaling."""
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_models():
    """Return unfitted pipelines, including a no-skill reference."""
    classifiers = {
        "Dummy": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=SEED, n_jobs=1
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=160,
            learning_rate=0.1,
            max_leaf_nodes=15,
            class_weight="balanced",
            early_stopping=False,
            random_state=SEED,
        ),
    }
    return {
        name: Pipeline(
            [
                (
                    "preprocess",
                    build_preprocessor(scale_numeric=name == "Logistic Regression"),
                ),
                ("classifier", estimator),
            ]
        )
        for name, estimator in classifiers.items()
    }


def tuning_grid(model_name):
    """Eight combinations, including the comparison model's configuration."""
    grids = {
        "Logistic Regression": {
            "classifier__C": [0.1, 1.0, 10.0, 100.0],
            "classifier__class_weight": [None, "balanced"],
        },
        "Random Forest": {
            "classifier__max_depth": [None, 12],
            "classifier__min_samples_leaf": [1, 3],
            "classifier__class_weight": [None, "balanced"],
        },
        "HistGradientBoosting": {
            "classifier__max_leaf_nodes": [15, 31],
            "classifier__l2_regularization": [0.0, 1.0],
            "classifier__class_weight": [None, "balanced"],
        },
    }
    return grids[model_name]
