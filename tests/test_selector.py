import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from agraspqfs import AGraspQFeatureSelector


def _toy_data(as_frame=False):
    X, y = make_classification(
        n_samples=120,
        n_features=12,
        n_informative=4,
        n_redundant=2,
        n_classes=2,
        random_state=7,
    )
    if as_frame:
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    return X, y


def test_fit_transform_dataframe_and_support():
    X, y = _toy_data(as_frame=True)
    selector = AGraspQFeatureSelector(
        estimator=GaussianNB(),
        preset="fast",
        min_features=2,
        max_features=6,
        rcl_size=8,
        random_state=42,
    )

    Xt = selector.fit_transform(X, y)

    assert isinstance(Xt, pd.DataFrame)
    assert 2 <= selector.n_features_selected_ <= 6
    assert Xt.shape[1] == selector.n_features_selected_
    assert selector.get_support().dtype == bool
    assert len(selector.get_support(indices=True)) == selector.n_features_selected_
    assert list(selector.get_feature_names_out()) == selector.selected_features_
    assert not selector.history_.empty


def test_budget_is_respected():
    X, y = _toy_data(as_frame=False)
    selector = AGraspQFeatureSelector(
        preset="fast",
        min_features=1,
        max_features=5,
        rcl_size=7,
        max_evaluations=20,
        random_state=1,
    )
    selector.fit(X, y)

    assert selector.n_evaluations_ <= 20
    assert selector.n_features_selected_ >= 1


def test_pipeline_compatibility():
    X, y = _toy_data(as_frame=False)
    pipe = Pipeline(
        steps=[
            (
                "fs",
                AGraspQFeatureSelector(
                    estimator=GaussianNB(),
                    preset="fast",
                    min_features=2,
                    max_features=5,
                    rcl_size=8,
                    random_state=2,
                ),
            ),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    pipe.fit(X, y)
    preds = pipe.predict(X[:5])

    assert preds.shape == (5,)
