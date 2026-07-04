"""Small reproducible benchmark comparing all features and A-GRASPQ-FS."""

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from agraspqfs import AGraspQFeatureSelector


def main() -> None:
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    baseline = GaussianNB().fit(X_train, y_train)
    baseline_f1 = f1_score(y_test, baseline.predict(X_test))

    selector = AGraspQFeatureSelector(
        estimator=GaussianNB(),
        preset="fast",
        min_features=2,
        max_features=15,
        max_evaluations=120,
        random_state=42,
    )
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)

    selected_model = GaussianNB().fit(X_train_sel, y_train)
    selected_f1 = f1_score(y_test, selected_model.predict(X_test_sel))

    print("All features")
    print(f"  n_features: {X_train.shape[1]}")
    print(f"  F1-score:   {baseline_f1:.4f}")
    print()
    print("A-GRASPQ-FS")
    print(f"  n_features: {selector.n_features_selected_}")
    print(f"  features:   {list(selector.selected_features_)}")
    print(f"  F1-score:   {selected_f1:.4f}")
    print(f"  evaluations:{selector.n_evaluations_}")
    print(f"  cache hits: {selector.n_cache_hits_}")


if __name__ == "__main__":
    main()
