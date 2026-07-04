"""Using A-GRASPQ-FS inside a scikit-learn Pipeline."""

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

from agraspqfs import AGraspQFeatureSelector

X, y = make_classification(
    n_samples=400,
    n_features=25,
    n_informative=6,
    n_redundant=5,
    random_state=42,
)

pipe = Pipeline(
    steps=[
        (
            "feature_selection",
            AGraspQFeatureSelector(
                estimator=GaussianNB(),
                preset="fast",
                min_features=2,
                max_features=12,
                random_state=42,
            ),
        ),
        ("classifier", LogisticRegression(max_iter=2000)),
    ]
)

scores = cross_val_score(pipe, X, y, cv=3, scoring="f1_weighted")
print("CV F1 scores:", scores)
print("Mean CV F1:", scores.mean())
