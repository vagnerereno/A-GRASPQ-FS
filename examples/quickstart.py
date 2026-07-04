"""Minimal A-GRASPQ-FS example."""

from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from agraspqfs import AGraspQFeatureSelector

X, y = make_classification(
    n_samples=500,
    n_features=30,
    n_informative=8,
    n_redundant=6,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

selector = AGraspQFeatureSelector(
    estimator=GaussianNB(),
    preset="fast",
    min_features=2,
    max_features=15,
    random_state=42,
)

X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

model = GaussianNB().fit(X_train_selected, y_train)
y_pred = model.predict(X_test_selected)

print("Selected feature indices:", selector.selected_indices_)
print("Number of selected features:", selector.n_features_selected_)
print("Internal best score:", round(selector.best_score_, 4))
print("Holdout F1:", round(f1_score(y_test, y_pred, average="weighted"), 4))
